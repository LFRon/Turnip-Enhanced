/*
 * Copyright (C) 2026
 * SPDX-License-Identifier: MIT
 *
 * Optional runtime gralloc metadata bridges for standalone Android builds.
 *
 * android-stub deliberately avoids build-time dependencies on Android's
 * private platform libraries.  Modern gralloc implementations, however, do
 * not expose enough information through the legacy gralloc module to import
 * private camera formats safely.  Resolve narrow metadata ABIs at runtime so
 * a single Vulkan HAL can use authoritative plane layouts when they are
 * available, while retaining a fail-closed fallback on other systems.
 *
 * QTI devices are tried through the gralloc::GetPlaneLayout() entry point in
 * libgrallocutils first.  Unlike libui, that library is part of the same
 * vendor gralloc stack as the in-process mapper HAL and does not require the
 * Vulkan client to access Android's private GraphicBufferMapper singleton.
 * The libui bridge remains as a generic fallback for platforms where it is
 * accessible.
 */

#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <new>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>

#include <string>
#include <vector>

#include "drm-uapi/drm_fourcc.h"
#include "util/log.h"

#include "u_gralloc_internal.h"

/* These are the data members of the VINTF-stable graphics-common AIDL
 * parcelables used by the QTI helper and GraphicBufferMapper.  Their exported
 * symbols contain the fully-qualified PlaneLayout type, so failure to find an
 * exact symbol also acts as the ABI/version gate for these optional bridges.
 */
namespace aidl::android::hardware::graphics::common {
struct ExtendableType {
   std::string name;
   int64_t value = 0;
};

struct PlaneLayoutComponent {
   ExtendableType type;
   int64_t offsetInBits = 0;
   int64_t sizeInBits = 0;
};

struct PlaneLayout {
   std::vector<PlaneLayoutComponent> components;
   int64_t offsetInBytes = 0;
   int64_t sampleIncrementInBits = 0;
   int64_t strideInBytes = 0;
   int64_t widthInSamples = 0;
   int64_t heightInSamples = 0;
   int64_t totalSizeInBytes = 0;
   int64_t horizontalSubsampling = 0;
   int64_t verticalSubsampling = 0;
};
} /* namespace aidl::android::hardware::graphics::common */

using aidl::android::hardware::graphics::common::ExtendableType;
using aidl::android::hardware::graphics::common::PlaneLayout;
using aidl::android::hardware::graphics::common::PlaneLayoutComponent;

/* The exact exported QTI/libui symbols name these libc++ AIDL types, but the
 * mangled name does not encode their data-member layout.  Keep the verified
 * Android arm64 ABI as an additional compile-time guard.  Other architectures
 * do not use the QTI bridge below.
 */
#if UINTPTR_MAX == UINT64_MAX
static_assert(sizeof(ExtendableType) == 32);
static_assert(sizeof(PlaneLayoutComponent) == 48);
static_assert(sizeof(PlaneLayout) == 88);
#endif

namespace {

constexpr size_t MAX_MAPPER_PLANE_COUNT = 8;
constexpr size_t MAX_COMPONENTS_PER_PLANE = 8;

constexpr int QTI_HANDLE_NUM_FDS = 2;
constexpr int QTI_HANDLE_NUM_INTS = 24;
constexpr int32_t QTI_HANDLE_MAGIC =
   ('g' << 24) | ('m' << 16) | ('s' << 8) | 'm';

/* QTI's private 8-bit NV12 UBWC format.  This is a gralloc software-format
 * ABI gate, not a GPU-model gate: any QTI stack using this format still has
 * to pass all of the plane geometry and allocation-layout checks below.
 */
constexpr int32_t QTI_HAL_PIXEL_FORMAT_YCBCR_420_SP_VENUS_UBWC =
   INT32_C(0x7fa30c06);

constexpr char STANDARD_PLANE_COMPONENT_TYPE[] =
   "android.hardware.graphics.common.PlaneLayoutComponentType";
constexpr int64_t PLANE_COMPONENT_Y = INT64_C(1) << 0;
constexpr int64_t PLANE_COMPONENT_CB = INT64_C(1) << 1;
constexpr int64_t PLANE_COMPONENT_CR = INT64_C(1) << 2;
constexpr int64_t PLANE_COMPONENT_R = INT64_C(1) << 10;
constexpr int64_t PLANE_COMPONENT_G = INT64_C(1) << 11;
constexpr int64_t PLANE_COMPONENT_B = INT64_C(1) << 12;
constexpr int64_t PLANE_COMPONENT_A = INT64_C(1) << 30;

constexpr char QTI_GET_PLANE_LAYOUT_SYMBOL[] =
   "_ZN7gralloc14GetPlaneLayoutEPN10qtigralloc16private_handle_tEPNSt3__16vectorIN4aidl7android8hardware8graphics6common11PlaneLayoutENS3_9allocatorISA_EEEE";

using HasMapperInstance = bool (*)();
using GetFourcc = int32_t (*)(void *, const native_handle_t *, uint32_t *);
using GetModifier = int32_t (*)(void *, const native_handle_t *, uint64_t *);
using GetAllocationSize =
   int32_t (*)(void *, const native_handle_t *, uint64_t *);
using GetPlaneLayouts = int32_t (*)(void *, const native_handle_t *,
                                   std::vector<PlaneLayout> *);
using GetQtiPlaneLayouts =
   int32_t (*)(native_handle_t *, std::vector<PlaneLayout> *);

struct libui_gralloc {
   struct u_gralloc base;
   void *libui;
   void **mapper_instance;
   HasMapperInstance has_mapper_instance;
   GetFourcc get_fourcc;
   GetModifier get_modifier;
   GetAllocationSize get_allocation_size;
   GetPlaneLayouts get_plane_layouts;
};

struct qti_metadata_gralloc {
   struct u_gralloc base;
   void *grallocutils;
   GetQtiPlaneLayouts get_plane_layouts;
   struct u_gralloc *fallback;
};

template <typename T>
bool
load_function(void *library, const char *name, T *out)
{
   static_assert(sizeof(T) == sizeof(void *));
   void *symbol = dlsym(library, name);
   if (!symbol)
      return false;

   memcpy(out, &symbol, sizeof(symbol));
   return true;
}

static bool
qti_handle_is_compatible(const native_handle_t *handle)
{
   /* This is the private_handle_t ABI validated by the supplied SM8550 QTI
    * gralloc stack.  It is a runtime software-ABI gate, not a GPU-model gate:
    * no QTI entry point is called unless the incoming handle matches it.
    */
   return sizeof(void *) == 8 && handle &&
          handle->version == sizeof(native_handle_t) &&
          handle->numFds == QTI_HANDLE_NUM_FDS &&
          handle->numInts == QTI_HANDLE_NUM_INTS &&
          handle->data[handle->numFds] == QTI_HANDLE_MAGIC;
}

static bool
get_dma_buf_allocation_size(const native_handle_t *handle,
                            uint64_t *allocation_size)
{
   struct stat statbuf = {};

   if (!handle || handle->numFds < 1 ||
       fstat(handle->data[0], &statbuf) != 0 || statbuf.st_size <= 0)
      return false;

   *allocation_size = static_cast<uint64_t>(statbuf.st_size);
   return true;
}

static bool
is_standard_component(const PlaneLayoutComponent &component, int64_t value)
{
   return component.type.name == STANDARD_PLANE_COMPONENT_TYPE &&
          component.type.value == value;
}

static bool
component_fits_sample(const PlaneLayoutComponent &component,
                      int64_t sample_increment)
{
   return sample_increment > 0 && component.offsetInBits >= 0 &&
          component.sizeInBits > 0 &&
          component.offsetInBits <= sample_increment &&
          component.sizeInBits <= sample_increment - component.offsetInBits;
}

static bool
is_nv12_luma_plane(const PlaneLayout &layout,
                   bool allow_omitted_components)
{
   if (layout.sampleIncrementInBits != 8 ||
       layout.horizontalSubsampling != 1 ||
       layout.verticalSubsampling != 1)
      return false;

   /* Some QTI gralloc versions, including the supplied SM8550 stack, return
    * authoritative geometry for this private NV12 UBWC format but omit the
    * standard Y component because their component-offset helper does not
    * handle that private HAL format.  The exact format and geometry remain
    * sufficient to identify the plane; never accept this omission for an
    * unknown or generic YUV format.
    */
   if (layout.components.empty())
      return allow_omitted_components;

   if (layout.components.size() != 1)
      return false;

   const PlaneLayoutComponent &component = layout.components[0];
   return is_standard_component(component, PLANE_COMPONENT_Y) &&
          component.offsetInBits == 0 && component.sizeInBits == 8;
}

static bool
is_nv12_chroma_plane(const PlaneLayout &layout,
                     bool allow_omitted_components)
{
   if (layout.sampleIncrementInBits != 16 ||
       layout.horizontalSubsampling != 2 ||
       layout.verticalSubsampling != 2)
      return false;

   if (layout.components.empty())
      return allow_omitted_components;

   if (layout.components.size() != 2)
      return false;

   bool found_cb = false;
   bool found_cr = false;
   for (const PlaneLayoutComponent &component : layout.components) {
      if (is_standard_component(component, PLANE_COMPONENT_CB) &&
          component.offsetInBits == 0 && component.sizeInBits == 8) {
         found_cb = true;
      } else if (is_standard_component(component, PLANE_COMPONENT_CR) &&
                 component.offsetInBits == 8 &&
                 component.sizeInBits == 8) {
         found_cr = true;
      } else {
         return false;
      }
   }

   return found_cb && found_cr;
}

static bool
is_standard_rgb_plane(const PlaneLayout &layout)
{
   if (layout.components.size() < 3 || layout.components.size() > 4 ||
       layout.sampleIncrementInBits <= 0 ||
       layout.horizontalSubsampling != 1 ||
       layout.verticalSubsampling != 1)
      return false;

   bool found_r = false;
   bool found_g = false;
   bool found_b = false;
   bool found_a = false;

   for (const PlaneLayoutComponent &component : layout.components) {
      if (!component_fits_sample(component, layout.sampleIncrementInBits))
         return false;

      bool *found = nullptr;
      if (is_standard_component(component, PLANE_COMPONENT_R))
         found = &found_r;
      else if (is_standard_component(component, PLANE_COMPONENT_G))
         found = &found_g;
      else if (is_standard_component(component, PLANE_COMPONENT_B))
         found = &found_b;
      else if (is_standard_component(component, PLANE_COMPONENT_A))
         found = &found_a;
      else
         return false;

      if (*found)
         return false;
      *found = true;
   }

   return found_r && found_g && found_b;
}

static bool
layout_value_fits_int(int64_t value)
{
   return value >= 0 && value <= INT_MAX;
}

static bool
valid_layout(const PlaneLayout &layout, uint64_t allocation_size)
{
   if (!layout_value_fits_int(layout.offsetInBytes) ||
       !layout_value_fits_int(layout.strideInBytes) ||
       layout.totalSizeInBytes <= 0 ||
       layout.widthInSamples <= 0 || layout.heightInSamples <= 0 ||
       layout.components.size() > MAX_COMPONENTS_PER_PLANE)
      return false;

   return static_cast<uint64_t>(layout.offsetInBytes) < allocation_size &&
          static_cast<uint64_t>(layout.totalSizeInBytes) <= allocation_size;
}

static bool
layout_range_fits(const PlaneLayout &layout, uint64_t allocation_size)
{
   const uint64_t offset = static_cast<uint64_t>(layout.offsetInBytes);
   const uint64_t size = static_cast<uint64_t>(layout.totalSizeInBytes);
   return offset <= allocation_size && size <= allocation_size - offset;
}

static bool
is_data_plane(const PlaneLayout &layout)
{
   return layout.sampleIncrementInBits > 0 && layout.strideInBytes > 0;
}

static bool
normalize_qti_nv12_layouts(const std::vector<PlaneLayout> &layouts,
                           int32_t hal_format,
                           std::vector<PlaneLayout> *normalized,
                           bool *compressed)
{
   const bool allow_omitted_components =
      hal_format == QTI_HAL_PIXEL_FORMAT_YCBCR_420_SP_VENUS_UBWC;

   /* Only accept the progressive, two-data-plane representation of this
    * private NV12 UBWC format when component labels are omitted.  Its QTI
    * PlaneLayouts vector must contain exactly two data and two metadata
    * planes.  Reject interlaced/batched/private variants instead of extending
    * the component omission exception to them.
    */
   if (allow_omitted_components && layouts.size() != 4)
      return false;

   size_t luma_index = layouts.size();
   size_t chroma_index = layouts.size();
   size_t data_count = 0;
   size_t metadata_count = 0;

   for (size_t i = 0; i < layouts.size(); i++) {
      if (is_data_plane(layouts[i])) {
         data_count++;
         if (is_nv12_luma_plane(layouts[i], allow_omitted_components) &&
             luma_index == layouts.size())
            luma_index = i;
         else if (is_nv12_chroma_plane(layouts[i],
                                       allow_omitted_components) &&
                  chroma_index == layouts.size())
            chroma_index = i;
         else
            return false;
      } else if (layouts[i].sampleIncrementInBits == 0) {
         metadata_count++;
      } else {
         return false;
      }
   }

   if (data_count != 2 || luma_index == layouts.size() ||
       chroma_index == layouts.size() ||
       (metadata_count != 0 && metadata_count != data_count) ||
       (allow_omitted_components && metadata_count != 2))
      return false;

   normalized->clear();
   normalized->reserve(layouts.size());
   normalized->push_back(layouts[luma_index]);
   normalized->push_back(layouts[chroma_index]);
   for (const PlaneLayout &layout : layouts) {
      if (!is_data_plane(layout))
         normalized->push_back(layout);
   }

   *compressed = metadata_count != 0;
   return true;
}

static bool
layouts_describe_standard_rgb(const std::vector<PlaneLayout> &layouts)
{
   size_t data_count = 0;

   for (const PlaneLayout &layout : layouts) {
      if (!is_data_plane(layout))
         continue;

      data_count++;
      if (!is_standard_rgb_plane(layout))
         return false;
   }

   return data_count == 1;
}

static int
copy_linear_layout(const native_handle_t *handle,
                   const std::vector<PlaneLayout> &layouts,
                   uint64_t allocation_size,
                   struct u_gralloc_buffer_basic_info *out)
{
   if (layouts.empty() || layouts.size() > ARRAY_SIZE(out->fds))
      return -EINVAL;

   int fd_index = 0;
   for (size_t i = 0; i < layouts.size(); i++) {
      const PlaneLayout &layout = layouts[i];
      if (!is_data_plane(layout) ||
          !layout_range_fits(layout, allocation_size))
         return -EINVAL;

      /* Gralloc4 has no fd-index field.  A zero offset on a subsequent
       * logical plane is the standard convention for a new dma-buf.  Extra
       * private-handle fds after the described planes (for example QCOM's
       * shared metadata fd) are intentionally ignored.
       */
      if (i > 0 && layout.offsetInBytes == 0)
         fd_index++;

      if (fd_index >= handle->numFds)
         return -EINVAL;

      out->fds[i] = handle->data[fd_index];
      out->offsets[i] = static_cast<int>(layout.offsetInBytes);
      out->strides[i] = static_cast<int>(layout.strideInBytes);
   }

   out->num_planes = static_cast<int>(layouts.size());
   return 0;
}

static int
copy_qcom_ubwc_layout(const native_handle_t *handle,
                      const std::vector<PlaneLayout> &layouts,
                      uint64_t allocation_size,
                      struct u_gralloc_buffer_basic_info *out)
{
   size_t data_indices[MAX_MAPPER_PLANE_COUNT];
   size_t metadata_indices[MAX_MAPPER_PLANE_COUNT];
   size_t data_count = 0;
   size_t metadata_count = 0;

   if (handle->numFds < 1 || layouts.empty() ||
       layouts.size() > MAX_MAPPER_PLANE_COUNT)
      return -EINVAL;

   for (size_t i = 0; i < layouts.size(); i++) {
      if (is_data_plane(layouts[i]))
         data_indices[data_count++] = i;
      else if (layouts[i].sampleIncrementInBits == 0)
         metadata_indices[metadata_count++] = i;
      else
         return -EINVAL;
   }

   if (data_count == 0 || data_count > ARRAY_SIZE(out->fds))
      return -EINVAL;

   /* QCOM RGB UBWC is reported as one logical data plane whose offset skips
    * over the leading UBWC metadata.  Vulkan's QCOM modifier plane starts at
    * that metadata, while Turnip computes the primary-data offset itself.
    */
   if (data_count == 1 && metadata_count == 0) {
      const PlaneLayout &data = layouts[data_indices[0]];
      if (data.offsetInBytes <= 0 ||
          static_cast<uint64_t>(data.totalSizeInBytes) != allocation_size)
         return -EINVAL;

      out->num_planes = 1;
      out->fds[0] = handle->data[0];
      out->offsets[0] = 0;
      out->strides[0] = static_cast<int>(data.strideInBytes);
      out->modifier_plane_layouts[0] = {
         .data_offset = static_cast<uint64_t>(data.offsetInBytes),
         .data_size = allocation_size -
                      static_cast<uint64_t>(data.offsetInBytes),
         .metadata_size = static_cast<uint64_t>(data.offsetInBytes),
         .metadata_row_pitch = 0,
      };
      out->has_explicit_modifier_layout = true;
      return 0;
   }

   /* QCOM YUV UBWC exposes physical planes in data/data/meta/meta order.
    * Convert them to logical Vulkan planes by pairing each primary plane with
    * the immediately preceding metadata range.  Do not rely on ordering or a
    * private native-handle layout: the exact offset/size adjacency is the
    * proof that a metadata plane belongs to a data plane.
    */
   if (metadata_count != data_count)
      return -EINVAL;

   for (size_t i = 0; i < layouts.size(); i++) {
      if (!layout_range_fits(layouts[i], allocation_size))
         return -EINVAL;

      const uint64_t start =
         static_cast<uint64_t>(layouts[i].offsetInBytes);
      const uint64_t end =
         start + static_cast<uint64_t>(layouts[i].totalSizeInBytes);
      for (size_t j = 0; j < i; j++) {
         const uint64_t other_start =
            static_cast<uint64_t>(layouts[j].offsetInBytes);
         const uint64_t other_end =
            other_start +
            static_cast<uint64_t>(layouts[j].totalSizeInBytes);
         if (start < other_end && other_start < end)
            return -EINVAL;
      }
   }

   bool metadata_used[MAX_MAPPER_PLANE_COUNT] = {};
   for (size_t i = 0; i < data_count; i++) {
      const PlaneLayout &data = layouts[data_indices[i]];
      size_t match = metadata_count;

      for (size_t j = 0; j < metadata_count; j++) {
         if (metadata_used[j])
            continue;

         const PlaneLayout &metadata = layouts[metadata_indices[j]];
         if (metadata.offsetInBytes > data.offsetInBytes)
            continue;

         const uint64_t metadata_end =
            static_cast<uint64_t>(metadata.offsetInBytes) +
            static_cast<uint64_t>(metadata.totalSizeInBytes);
         if (metadata_end == static_cast<uint64_t>(data.offsetInBytes)) {
            match = j;
            break;
         }
      }

      if (match == metadata_count)
         return -EINVAL;

      metadata_used[match] = true;
      const PlaneLayout &metadata = layouts[metadata_indices[match]];
      if (metadata.strideInBytes <= 0)
         return -EINVAL;

      out->fds[i] = handle->data[0];
      out->offsets[i] = static_cast<int>(metadata.offsetInBytes);
      out->strides[i] = static_cast<int>(data.strideInBytes);
      out->modifier_plane_layouts[i] = {
         .data_offset = static_cast<uint64_t>(data.offsetInBytes),
         .data_size = static_cast<uint64_t>(data.totalSizeInBytes),
         .metadata_size = static_cast<uint64_t>(metadata.totalSizeInBytes),
         .metadata_row_pitch =
            static_cast<uint32_t>(metadata.strideInBytes),
      };
   }

   out->num_planes = static_cast<int>(data_count);
   out->has_explicit_modifier_layout = true;
   return 0;
}

static int
copy_qti_standard_rgb_layout(
   const struct u_gralloc_buffer_handle *hnd,
   const std::vector<PlaneLayout> &layouts, uint64_t allocation_size,
   struct u_gralloc_buffer_basic_info *out)
{
   size_t data_count = 0;
   size_t metadata_count = 0;
   const PlaneLayout *data = nullptr;

   for (const PlaneLayout &layout : layouts) {
      if (is_data_plane(layout)) {
         data_count++;
         data = &layout;
      } else if (layout.sampleIncrementInBits == 0) {
         metadata_count++;
      } else {
         return -EINVAL;
      }
   }

   if (data_count != 1 || !data || metadata_count > 1)
      return -EINVAL;

   const int drm_fourcc = get_fourcc_from_hal_format(hnd->hal_format);
   if (drm_fourcc == -1)
      return -ENOTSUP;

   out->drm_fourcc = drm_fourcc;
   if (metadata_count == 0 && data->offsetInBytes == 0) {
      out->modifier = DRM_FORMAT_MOD_LINEAR;
      return copy_linear_layout(hnd->handle, layouts, allocation_size, out);
   }

   /* QTI may either expose a separate RGB metadata plane or fold the leading
    * metadata size into a single data PlaneLayout.  Both representations are
    * handled by copy_qcom_ubwc_layout() and validated against the dma-buf.
    */
   out->modifier = DRM_FORMAT_MOD_QCOM_COMPRESSED;
   return copy_qcom_ubwc_layout(hnd->handle, layouts, allocation_size, out);
}

static bool
can_use_fallback_for_known_non_yuv(
   const struct u_gralloc_buffer_handle *hnd)
{
   return hnd && !is_hal_format_yuv(hnd->hal_format) &&
          get_hal_format_bpp(hnd->hal_format) > 0;
}

static int
qti_fallback_get_buffer_basic_info(
   qti_metadata_gralloc *gr, struct u_gralloc_buffer_handle *hnd,
   struct u_gralloc_buffer_basic_info *out)
{
   if (!gr->fallback)
      return -ENOTSUP;

   return gr->fallback->ops.get_buffer_basic_info(gr->fallback, hnd, out);
}

static int
qti_get_buffer_basic_info(struct u_gralloc *gralloc,
                          struct u_gralloc_buffer_handle *hnd,
                          struct u_gralloc_buffer_basic_info *out)
{
   qti_metadata_gralloc *gr =
      reinterpret_cast<qti_metadata_gralloc *>(gralloc);

   if (!hnd || !hnd->handle)
      return -EINVAL;

   if (!qti_handle_is_compatible(hnd->handle)) {
      /* The explicit-YUV capability must never make an unknown private handle
       * enter a heuristic path.  Preserve the old fallback only for an
       * unambiguously standard, non-YUV HAL format.
       */
      if (can_use_fallback_for_known_non_yuv(hnd))
         return qti_fallback_get_buffer_basic_info(gr, hnd, out);
      return -EINVAL;
   }

   uint64_t allocation_size = 0;
   std::vector<PlaneLayout> layouts;
   if (!get_dma_buf_allocation_size(hnd->handle, &allocation_size) ||
       gr->get_plane_layouts(const_cast<native_handle_t *>(hnd->handle),
                             &layouts) != 0 ||
       layouts.empty() || layouts.size() > MAX_MAPPER_PLANE_COUNT) {
      if (can_use_fallback_for_known_non_yuv(hnd))
         return qti_fallback_get_buffer_basic_info(gr, hnd, out);

      mesa_logw_once("QTI gralloc failed to return complete plane metadata");
      return -EINVAL;
   }

   for (const PlaneLayout &layout : layouts) {
      if (!valid_layout(layout, allocation_size)) {
         if (can_use_fallback_for_known_non_yuv(hnd))
            return qti_fallback_get_buffer_basic_info(gr, hnd, out);

         mesa_logw_once("QTI gralloc returned an invalid plane layout");
         return -EINVAL;
      }
   }

   /* layer_count is not exposed by this QTI metadata entry point.  Keep it
    * zero rather than assuming a single-layer allocation, but preserve the
    * authoritative dma-buf allocation size for callers which can use it.
    */
   out->alloc_size = allocation_size;

   std::vector<PlaneLayout> normalized;
   bool compressed = false;
   if (normalize_qti_nv12_layouts(layouts, hnd->hal_format, &normalized,
                                  &compressed)) {
      out->drm_fourcc = DRM_FORMAT_NV12;
      out->modifier = compressed ? DRM_FORMAT_MOD_QCOM_COMPRESSED
                                 : DRM_FORMAT_MOD_LINEAR;

      const int ret = compressed
                         ? copy_qcom_ubwc_layout(hnd->handle, normalized,
                                                allocation_size, out)
                         : copy_linear_layout(hnd->handle, normalized,
                                              allocation_size, out);
      if (ret) {
         mesa_logw_once("Unsupported or inconsistent QTI NV12 plane layout");
         return ret;
      }

      return 0;
   }

   /* QTI's private IMPLEMENTATION_DEFINED value can describe an ordinary RGB
    * allocation.  Component metadata proves that case is non-YUV; retain the
    * established Mesa RGB import path rather than guessing a private YUV
    * FourCC.  Standard non-YUV HAL formats are equally safe to delegate.
    */
   if (layouts_describe_standard_rgb(layouts)) {
      const int ret = copy_qti_standard_rgb_layout(
         hnd, layouts, allocation_size, out);
      if (ret)
         mesa_logw_once("Unsupported or inconsistent QTI RGB plane layout");
      return ret;
   }

   if (can_use_fallback_for_known_non_yuv(hnd))
      return qti_fallback_get_buffer_basic_info(gr, hnd, out);

   mesa_logw_once("Unsupported QTI private YUV plane component layout "
                  "(HAL format 0x%x)", hnd->hal_format);
   return -ENOTSUP;
}

static int
qti_metadata_gralloc_destroy(struct u_gralloc *gralloc)
{
   qti_metadata_gralloc *gr =
      reinterpret_cast<qti_metadata_gralloc *>(gralloc);

   if (gr->fallback)
      gr->fallback->ops.destroy(gr->fallback);
   if (gr->grallocutils)
      dlclose(gr->grallocutils);
   delete gr;
   return 0;
}

static int
libui_get_buffer_basic_info(struct u_gralloc *gralloc,
                            struct u_gralloc_buffer_handle *hnd,
                            struct u_gralloc_buffer_basic_info *out)
{
   libui_gralloc *gr = reinterpret_cast<libui_gralloc *>(gralloc);

   if (!hnd || !hnd->handle || hnd->handle->numFds <= 0)
      return -EINVAL;

   /* Singleton::hasInstance() takes the platform lock.  Reading sInstance
    * only after it succeeds avoids racing Android's lazy mapper creation.  An
    * AHardwareBuffer that has been imported through libui normally guarantees
    * this; if it does not, waiting for a future query is safer than constructing
    * a private platform C++ object from the Vulkan HAL.
    */
   if (!gr->has_mapper_instance()) {
      mesa_logw_once("libui GraphicBufferMapper is not initialized; rejecting buffer metadata query");
      return -EAGAIN;
   }

   void *mapper = __atomic_load_n(gr->mapper_instance, __ATOMIC_ACQUIRE);
   if (!mapper)
      return -EAGAIN;

   uint32_t drm_fourcc = 0;
   uint64_t modifier = DRM_FORMAT_MOD_INVALID;
   uint64_t allocation_size = 0;
   std::vector<PlaneLayout> layouts;

   if (gr->get_fourcc(mapper, hnd->handle, &drm_fourcc) != 0 ||
       gr->get_modifier(mapper, hnd->handle, &modifier) != 0 ||
       gr->get_allocation_size(mapper, hnd->handle, &allocation_size) != 0 ||
       allocation_size == 0 ||
       gr->get_plane_layouts(mapper, hnd->handle, &layouts) != 0 ||
       layouts.empty() || layouts.size() > MAX_MAPPER_PLANE_COUNT) {
      mesa_logw_once("libui failed to return complete gralloc metadata");
      return -EINVAL;
   }

   for (const PlaneLayout &layout : layouts) {
      if (!valid_layout(layout, allocation_size)) {
         mesa_logw_once("libui returned an invalid gralloc plane layout");
         return -EINVAL;
      }
   }

   /* This libui bridge predates the public getLayerCount() ABI used by the
    * native Mapper5 backend.  Do not infer it from plane metadata.
    */
   out->alloc_size = allocation_size;

   if (drm_fourcc == 0)
      return -EINVAL;

   out->drm_fourcc = drm_fourcc;
   out->modifier = modifier;

   int ret;
   switch (modifier) {
   case DRM_FORMAT_MOD_LINEAR:
      ret = copy_linear_layout(hnd->handle, layouts, allocation_size, out);
      break;
   case DRM_FORMAT_MOD_QCOM_COMPRESSED:
      ret = copy_qcom_ubwc_layout(hnd->handle, layouts, allocation_size, out);
      break;
   default:
      /* Turnip cannot safely infer the layout of an unknown modifier. */
      ret = -ENOTSUP;
      break;
   }

   if (ret)
      mesa_logw_once("Unsupported or inconsistent gralloc plane layout (fourcc=0x%x, modifier=0x%" PRIx64 ")",
                     drm_fourcc, modifier);

   return ret;
}

static int
libui_gralloc_destroy(struct u_gralloc *gralloc)
{
   libui_gralloc *gr = reinterpret_cast<libui_gralloc *>(gralloc);
   if (gr->libui)
      dlclose(gr->libui);
   delete gr;
   return 0;
}

} /* namespace */

extern "C" struct u_gralloc *
u_gralloc_qti_metadata_api_create(void)
{
   /* Only the supplied 64-bit QTI ABI has been verified.  A 32-bit build can
    * still use the generic libui/fallback backends below.
    */
   if (sizeof(void *) != 8)
      return nullptr;

   qti_metadata_gralloc *gr =
      new (std::nothrow) qti_metadata_gralloc{};
   if (!gr)
      return nullptr;

   /* Mapper4 is a same-process HAL, so its gralloc helper may already be in
    * the process-wide symbol group.  Reuse it first; this avoids a filesystem
    * access that an untrusted app's SELinux domain may reject.  Otherwise take
    * our own dlopen reference through the Vulkan/SP-HAL linker namespace.
    */
   if (!load_function(RTLD_DEFAULT, QTI_GET_PLANE_LAYOUT_SYMBOL,
                      &gr->get_plane_layouts)) {
      gr->grallocutils =
         dlopen("libgrallocutils.so", RTLD_NOW | RTLD_LOCAL);
      if (!gr->grallocutils ||
          !load_function(gr->grallocutils, QTI_GET_PLANE_LAYOUT_SYMBOL,
                         &gr->get_plane_layouts))
         goto fail;
   }

   gr->fallback = u_gralloc_fallback_create();
   if (!gr->fallback)
      goto fail;

   gr->base.ops.get_buffer_basic_info = qti_get_buffer_basic_info;
   gr->base.ops.destroy = qti_metadata_gralloc_destroy;
   gr->base.capabilities = U_GRALLOC_CAP_EXPLICIT_YUV_LAYOUT;

   mesa_logi("Using runtime QTI gralloc plane metadata bridge");
   return &gr->base;

fail:
   qti_metadata_gralloc_destroy(&gr->base);
   return nullptr;
}

extern "C" struct u_gralloc *
u_gralloc_imapper_api_create(void)
{
   libui_gralloc *gr = new (std::nothrow) libui_gralloc{};
   if (!gr)
      return nullptr;

   gr->libui = dlopen("libui.so", RTLD_NOW | RTLD_LOCAL);
   if (!gr->libui)
      goto fail;

   gr->mapper_instance = static_cast<void **>(dlsym(
      gr->libui,
      "_ZN7android9SingletonINS_19GraphicBufferMapperEE9sInstanceE"));

   if (!gr->mapper_instance ||
       !load_function(
          gr->libui,
          "_ZN7android9SingletonINS_19GraphicBufferMapperEE11hasInstanceEv",
          &gr->has_mapper_instance) ||
       !load_function(
          gr->libui,
          "_ZN7android19GraphicBufferMapper20getPixelFormatFourCCEPK13native_handlePj",
          &gr->get_fourcc) ||
       !load_function(
          gr->libui,
          "_ZN7android19GraphicBufferMapper22getPixelFormatModifierEPK13native_handlePm",
          &gr->get_modifier) ||
       !load_function(
          gr->libui,
          "_ZN7android19GraphicBufferMapper17getAllocationSizeEPK13native_handlePm",
          &gr->get_allocation_size) ||
       !load_function(
          gr->libui,
          "_ZN7android19GraphicBufferMapper15getPlaneLayoutsEPK13native_handlePNSt3__16vectorIN4aidl7android8hardware8graphics6common11PlaneLayoutENS4_9allocatorISB_EEEE",
          &gr->get_plane_layouts))
      goto fail;

   gr->base.ops.get_buffer_basic_info = libui_get_buffer_basic_info;
   gr->base.ops.destroy = libui_gralloc_destroy;
   gr->base.capabilities = U_GRALLOC_CAP_EXPLICIT_YUV_LAYOUT;

   mesa_logi("Using runtime libui gralloc metadata bridge");
   return &gr->base;

fail:
   libui_gralloc_destroy(&gr->base);
   return nullptr;
}
