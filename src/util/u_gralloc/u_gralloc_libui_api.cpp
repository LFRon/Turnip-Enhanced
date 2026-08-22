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
 * Newer QTI devices are tried through gralloc::GetPlaneLayout() in
 * libgrallocutils first.  Older stacks expose a public BufferInfo/
 * PlaneLayoutInfo interface instead.  Runtime-selected backend operations
 * isolate those ABI families; native-handle profiles isolate compatible
 * revisions within a family.  A still older Venus-only interface remains a
 * narrow final fallback.  No vendor symbol is linked directly.  Unlike libui,
 * libgrallocutils is part of the same vendor gralloc stack as the in-process
 * mapper HAL and does not require the Vulkan client to access Android's
 * private GraphicBufferMapper singleton.  The libui bridge remains as a
 * generic fallback for platforms where it is accessible.
 */

#include <dlfcn.h>
#include <errno.h>
#include <hardware/gralloc.h>
#include <inttypes.h>
#include <limits.h>
#include <new>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <system/graphics.h>

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
constexpr int QTI_MODERN_HANDLE_NUM_INTS = 24;
constexpr int QTI_LEGACY_BASE_HANDLE_NUM_INTS = 22;
constexpr int QTI_LEGACY_RESERVED_HANDLE_NUM_INTS = 23;
constexpr int QTI_LEGACY_TWO_OPTIONAL_WORDS_HANDLE_NUM_INTS = 24;
constexpr int QTI_LEGACY_THREE_OPTIONAL_WORDS_HANDLE_NUM_INTS = 25;
constexpr int QTI_LEGACY_ALL_OPTIONAL_WORDS_HANDLE_NUM_INTS = 26;
constexpr int32_t QTI_HANDLE_MAGIC =
   ('g' << 24) | ('m' << 16) | ('s' << 8) | 'm';

/* Offsets in the legacy arm64 private_handle_t::data[] ABI.  Keep these
 * separate from the modern qtigralloc handle: matching the magic alone is
 * not sufficient to call a private C++ entry point safely.
 */
constexpr int QTI_LEGACY_HANDLE_FLAGS_INDEX = 3;
constexpr int QTI_LEGACY_HANDLE_WIDTH_INDEX = 4;
constexpr int QTI_LEGACY_HANDLE_HEIGHT_INDEX = 5;
constexpr int QTI_LEGACY_HANDLE_UNALIGNED_WIDTH_INDEX = 6;
constexpr int QTI_LEGACY_HANDLE_UNALIGNED_HEIGHT_INDEX = 7;
constexpr int QTI_LEGACY_HANDLE_FORMAT_INDEX = 8;
constexpr int QTI_LEGACY_HANDLE_LAYER_COUNT_INDEX = 10;
constexpr int QTI_LEGACY_HANDLE_USAGE_INDEX = 13;
constexpr int QTI_LEGACY_HANDLE_SIZE_INDEX = 15;
constexpr int QTI_LEGACY_HANDLE_OFFSET_INDEX = 16;
constexpr int QTI_LEGACY_HANDLE_BASE_INDEX = 18;

static_assert(QTI_LEGACY_HANDLE_BASE_INDEX + 2 <=
              QTI_HANDLE_NUM_FDS + QTI_LEGACY_BASE_HANDLE_NUM_INTS);

constexpr uint32_t QTI_HANDLE_FLAG_UBWC_ALIGNED = UINT32_C(0x08000000);
constexpr uint32_t QTI_HANDLE_FLAG_UBWC_ALIGNED_PI =
   UINT32_C(0x40000000);
constexpr uint32_t QTI_HANDLE_FLAG_SECURE_BUFFER = UINT32_C(0x00000400);

/* The public QTI display stack uses this legacy modifier bit to describe the
 * 16-bit container used by P010.  Its numeric value was later assigned to
 * DRM_FORMAT_MOD_QCOM_TILED2, so it must only be interpreted as DX after an
 * exact QTI P010 format and plane-layout match.
 */
constexpr uint64_t QTI_DRM_FORMAT_MODIFIER_DX =
   fourcc_mod_code(QCOM, 0x2);

/* QTI's private 8-bit NV12 UBWC format.  This is a gralloc software-format
 * ABI gate, not a GPU-model gate: any QTI stack using this format still has
 * to pass all of the plane geometry and allocation-layout checks below.
 */
constexpr int32_t QTI_HAL_PIXEL_FORMAT_YCBCR_420_SP_VENUS_UBWC =
   INT32_C(0x7fa30c06);
constexpr int32_t QTI_HAL_PIXEL_FORMAT_NV12_ENCODEABLE = INT32_C(0x102);
constexpr int32_t QTI_HAL_PIXEL_FORMAT_NV21_ENCODEABLE =
   INT32_C(0x7fa30c00);
constexpr int32_t QTI_HAL_PIXEL_FORMAT_YCBCR_420_SP_VENUS =
   INT32_C(0x7fa30c04);
constexpr int32_t QTI_HAL_PIXEL_FORMAT_YCBCR_420_SP = INT32_C(0x109);
constexpr int32_t QTI_HAL_PIXEL_FORMAT_YCRCB_420_SP_ADRENO =
   INT32_C(0x7fa30c01);
constexpr int32_t QTI_HAL_PIXEL_FORMAT_NV21_ZSL = INT32_C(0x113);
constexpr int32_t QTI_HAL_PIXEL_FORMAT_YCRCB_420_SP_VENUS = INT32_C(0x114);
constexpr int32_t QTI_HAL_PIXEL_FORMAT_NV12_HEIF = INT32_C(0x116);
constexpr int32_t QTI_HAL_PIXEL_FORMAT_NV12_LINEAR_FLEX = INT32_C(0x125);
constexpr int32_t QTI_HAL_PIXEL_FORMAT_YCBCR_420_P010 =
   HAL_PIXEL_FORMAT_YCBCR_P010;
constexpr int32_t QTI_HAL_PIXEL_FORMAT_YCBCR_420_P010_VENUS =
   INT32_C(0x7fa30c0a);
constexpr int32_t QTI_HAL_PIXEL_FORMAT_RGBA_5551 = INT32_C(6);
constexpr int32_t QTI_HAL_PIXEL_FORMAT_RGBA_4444 = INT32_C(7);
constexpr int32_t QTI_HAL_PIXEL_FORMAT_BGRX_8888 = INT32_C(0x112);
constexpr int32_t QTI_HAL_PIXEL_FORMAT_BGR_565 = INT32_C(0x115);
constexpr int32_t QTI_HAL_PIXEL_FORMAT_ARGB_2101010 = INT32_C(0x117);
constexpr int32_t QTI_HAL_PIXEL_FORMAT_RGBX_1010102 = INT32_C(0x118);
constexpr int32_t QTI_HAL_PIXEL_FORMAT_XRGB_2101010 = INT32_C(0x119);
constexpr int32_t QTI_HAL_PIXEL_FORMAT_BGRA_1010102 = INT32_C(0x11a);
constexpr int32_t QTI_HAL_PIXEL_FORMAT_ABGR_2101010 = INT32_C(0x11b);
constexpr int32_t QTI_HAL_PIXEL_FORMAT_BGRX_1010102 = INT32_C(0x11c);
constexpr int32_t QTI_HAL_PIXEL_FORMAT_XBGR_2101010 = INT32_C(0x11d);

/* android.hardware.graphics.mapper StandardMetadataType values.  These are
 * part of the stable Mapper4 metadata contract and match the public QTI
 * MetadataType definition.  GetMetaDataValue() copies the decoded value into
 * caller-owned storage, avoiding any dependency on a private_handle_t field
 * offset.
 */
constexpr int64_t STANDARD_METADATA_WIDTH = 3;
constexpr int64_t STANDARD_METADATA_HEIGHT = 4;
constexpr int64_t STANDARD_METADATA_LAYER_COUNT = 5;
constexpr int64_t STANDARD_METADATA_PIXEL_FORMAT_REQUESTED = 6;
constexpr int64_t STANDARD_METADATA_PIXEL_FORMAT_FOURCC = 7;
constexpr int64_t STANDARD_METADATA_PIXEL_FORMAT_MODIFIER = 8;
constexpr int64_t STANDARD_METADATA_USAGE = 9;
constexpr int64_t STANDARD_METADATA_ALLOCATION_SIZE = 10;
constexpr int64_t STANDARD_METADATA_PROTECTED_CONTENT = 11;
constexpr int64_t QTI_METADATA_ALIGNED_WIDTH_IN_PIXELS = 10014;
constexpr int64_t QTI_METADATA_ALIGNED_HEIGHT_IN_PIXELS = 10015;

/* Public QTI Mapper metadata.  HWC uses this immutable field to distinguish
 * ordinary UBWC from UBWC-PI; PlaneLayout and the generic QCOM DRM modifier do
 * not encode that distinction.
 */
constexpr int64_t QTI_METADATA_PRIVATE_FLAGS = 10013;

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
constexpr char QTI_GET_METADATA_VALUE_SYMBOL[] =
   "_ZN7gralloc16GetMetaDataValueEPvlS0_";
constexpr char QTI_GET_MODERN_COLOR_SPACE_SYMBOL[] =
   "_ZN7gralloc25GetColorSpaceFromMetadataEPN10qtigralloc16private_handle_tEPi";
constexpr char QTI_GET_LEGACY_YUV_UBWC_PLANE_INFO_SYMBOL[] =
   "_ZN7gralloc21GetYuvUbwcSPPlaneInfoEjjiPNS_15PlaneLayoutInfoE";
/* The SM8150/SDM845 public gralloc exports the same helper name with a
 * different, narrower signature that fills an android_ycbcr in-place instead
 * of the newer PlaneLayoutInfo array.  Both ABIs coexist in the wild, so each
 * one is loaded by its exact mangled symbol.
 */
constexpr char QTI_GET_LEGACY_YCB_CR_UBWC_PLANE_INFO_SYMBOL[] =
   "_ZN7gralloc25GetYuvUbwcSPPlaneInfoEjjjiP13android_ycbcr";
constexpr char QTI_GET_LEGACY_YUV_PLANE_INFO_SYMBOL[] =
   "_ZN7gralloc15GetYUVPlaneInfoEPK16private_handle_tP13android_ycbcr";
constexpr char QTI_GET_LEGACY_YUV_PLANE_LAYOUTS_SYMBOL[] =
   "_ZN7gralloc15GetYUVPlaneInfoERKNS_10BufferInfoEiiiiPiPNS_15PlaneLayoutInfoE";
constexpr char QTI_GET_LEGACY_RGB_PLANE_LAYOUTS_SYMBOL[] =
   "_ZN7gralloc15GetRGBPlaneInfoERKNS_10BufferInfoEiiiiPiPNS_15PlaneLayoutInfoE";
constexpr char QTI_GET_LEGACY_DRM_FORMAT_SYMBOL[] =
   "_ZN7gralloc12GetDRMFormatEjjPjPm";
constexpr char QTI_GET_LEGACY_COLOR_SPACE_SYMBOL[] =
   "_ZN7gralloc25GetColorSpaceFromMetadataEP16private_handle_tPi";
constexpr char QTI_IS_UBWC_ENABLED_SYMBOL[] =
   "_ZN7gralloc13IsUBwcEnabledEim";

/* This is the legacy msm_media_info.h enum value consumed by
 * GetYuvUbwcSPPlaneInfo().  The handle-aware helper below must independently
 * reproduce the returned data offsets before the result is accepted.
 */
constexpr int QTI_LEGACY_COLOR_FMT_NV12_UBWC = 3;

constexpr uint32_t QTI_LEGACY_PLANE_COMPONENT_Y = UINT32_C(1) << 0;
constexpr uint32_t QTI_LEGACY_PLANE_COMPONENT_CB = UINT32_C(1) << 1;
constexpr uint32_t QTI_LEGACY_PLANE_COMPONENT_CR = UINT32_C(1) << 2;
constexpr uint32_t QTI_LEGACY_PLANE_COMPONENT_R = UINT32_C(1) << 10;
constexpr uint32_t QTI_LEGACY_PLANE_COMPONENT_G = UINT32_C(1) << 11;
constexpr uint32_t QTI_LEGACY_PLANE_COMPONENT_B = UINT32_C(1) << 12;
constexpr uint32_t QTI_LEGACY_PLANE_COMPONENT_A = UINT32_C(1) << 20;
constexpr uint32_t QTI_LEGACY_PLANE_COMPONENT_META = UINT32_C(1) << 31;

/* Exact legacy gralloc::BufferInfo ABI. */
struct LegacyQtiBufferInfo {
   int32_t width;
   int32_t height;
   int32_t format;
   int32_t layer_count;
   uint64_t usage;
};

static_assert(sizeof(LegacyQtiBufferInfo) == 24);
static_assert(alignof(LegacyQtiBufferInfo) == 8);

/* Exact legacy gralloc::PlaneLayoutInfo ABI. */
struct LegacyQtiPlaneLayoutInfo {
   uint32_t component;
   uint32_t horizontal_subsampling;
   uint32_t vertical_subsampling;
   uint32_t offset;
   int32_t step;
   int32_t stride;
   int32_t stride_bytes;
   int32_t scanlines;
   uint32_t size;
};

static_assert(sizeof(LegacyQtiPlaneLayoutInfo) == 36);
static_assert(alignof(LegacyQtiPlaneLayoutInfo) == 4);

using HasMapperInstance = bool (*)();
using GetFourcc = int32_t (*)(void *, const native_handle_t *, uint32_t *);
using GetModifier = int32_t (*)(void *, const native_handle_t *, uint64_t *);
using GetAllocationSize =
   int32_t (*)(void *, const native_handle_t *, uint64_t *);
using GetPlaneLayouts = int32_t (*)(void *, const native_handle_t *,
                                   std::vector<PlaneLayout> *);
using GetQtiPlaneLayouts =
   int32_t (*)(native_handle_t *, std::vector<PlaneLayout> *);
using GetQtiMetadataValue = int32_t (*)(void *, int64_t, void *);
using GetLegacyQtiYuvUbwcPlaneInfo =
   void (*)(uint32_t, uint32_t, int, LegacyQtiPlaneLayoutInfo *);
/* SM8150/SDM845 variant: GetYuvUbwcSPPlaneInfo(base, width, height,
 * color_format, struct android_ycbcr *).  The first argument is the GPU
 * address base; on the legacy handle ABI that base equals the handle base,
 * which the caller already validates, so the result is used as an
 * authoritative geometry hint for the data planes of an NV12 UBWC buffer.
 */
using GetLegacyQtiYcbcrUbwcPlaneInfo =
   void (*)(uint64_t, uint32_t, uint32_t, int, struct android_ycbcr *);
using GetLegacyQtiYuvPlaneInfo =
   int (*)(const native_handle_t *, struct android_ycbcr *);
using GetLegacyQtiYuvPlaneLayouts =
   int (*)(const LegacyQtiBufferInfo &, int32_t, int32_t, int32_t, int32_t,
           int *, LegacyQtiPlaneLayoutInfo *);
using GetLegacyQtiRgbPlaneLayouts =
   void (*)(const LegacyQtiBufferInfo &, int32_t, int32_t, int32_t, int32_t,
            int *, LegacyQtiPlaneLayoutInfo *);
using GetLegacyQtiDrmFormat =
   void (*)(uint32_t, uint32_t, uint32_t *, uint64_t *);
using GetQtiColorSpace = void (*)(native_handle_t *, int *);
using IsQtiUbwcEnabled = bool (*)(int, unsigned long);

struct qti_metadata_gralloc;

struct qti_metadata_backend {
   const char *name;
   const char *color_space_symbol;
   bool supports_swapchain_ubwc;
   bool (*load)(void *library, qti_metadata_gralloc *gr);
   bool (*handle_is_compatible)(const native_handle_t *handle);
   int (*get_buffer_basic_info)(
      qti_metadata_gralloc *gr, struct u_gralloc_buffer_handle *hnd,
      struct u_gralloc_buffer_basic_info *out);
};

struct qti_legacy_handle_profile {
   int num_ints;
   const char *name;
};

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
   const qti_metadata_backend *backend;
   GetQtiPlaneLayouts get_plane_layouts;
   GetQtiMetadataValue get_metadata_value;
   GetLegacyQtiYuvUbwcPlaneInfo get_legacy_yuv_ubwc_plane_info;
   GetLegacyQtiYcbcrUbwcPlaneInfo get_legacy_ycbcr_ubwc_plane_info;
   GetLegacyQtiYuvPlaneInfo get_legacy_yuv_plane_info;
   GetLegacyQtiYuvPlaneLayouts get_legacy_yuv_plane_layouts;
   GetLegacyQtiRgbPlaneLayouts get_legacy_rgb_plane_layouts;
   GetLegacyQtiDrmFormat get_legacy_drm_format;
   GetQtiColorSpace get_color_space;
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
load_qti_modern_metadata_api(void *library, qti_metadata_gralloc *gr)
{
   GetQtiPlaneLayouts get_plane_layouts = nullptr;
   if (!load_function(library, QTI_GET_PLANE_LAYOUT_SYMBOL,
                      &get_plane_layouts))
      return false;

   gr->get_plane_layouts = get_plane_layouts;

   /* The public SM8550 helper exposes immutable Mapper metadata through this
    * entry point.  Keep it optional so another QTI revision which only
    * exposes PlaneLayout retains the existing narrow import path; when it is
    * present, every queried field becomes mandatory for that buffer.
    */
   GetQtiMetadataValue get_metadata_value = nullptr;
   if (load_function(library, QTI_GET_METADATA_VALUE_SYMBOL,
                     &get_metadata_value))
      gr->get_metadata_value = get_metadata_value;

   return true;
}

static bool
load_qti_legacy_plane_layout_api(void *library, qti_metadata_gralloc *gr)
{
   GetLegacyQtiYuvPlaneLayouts get_yuv_plane_layouts = nullptr;
   GetLegacyQtiYuvPlaneInfo get_yuv_plane_info = nullptr;
   if (!load_function(library, QTI_GET_LEGACY_YUV_PLANE_LAYOUTS_SYMBOL,
                      &get_yuv_plane_layouts) ||
       !load_function(library, QTI_GET_LEGACY_YUV_PLANE_INFO_SYMBOL,
                      &get_yuv_plane_info))
      return false;

   gr->get_legacy_yuv_plane_layouts = get_yuv_plane_layouts;
   gr->get_legacy_yuv_plane_info = get_yuv_plane_info;

   /* RGB import is optional for this backend.  YUV remains useful when an
    * older libgrallocutils exports the public YUV ABI but not both RGB helpers.
    */
   GetLegacyQtiRgbPlaneLayouts get_rgb_plane_layouts = nullptr;
   GetLegacyQtiDrmFormat get_drm_format = nullptr;
   if (load_function(library, QTI_GET_LEGACY_RGB_PLANE_LAYOUTS_SYMBOL,
                     &get_rgb_plane_layouts) &&
       load_function(library, QTI_GET_LEGACY_DRM_FORMAT_SYMBOL,
                     &get_drm_format)) {
      gr->get_legacy_rgb_plane_layouts = get_rgb_plane_layouts;
      gr->get_legacy_drm_format = get_drm_format;
   }

   return true;
}

static bool
load_qti_legacy_nv12_ubwc_api(void *library, qti_metadata_gralloc *gr)
{
   GetLegacyQtiYuvUbwcPlaneInfo get_yuv_ubwc_plane_info = nullptr;
   GetLegacyQtiYuvPlaneInfo get_yuv_plane_info = nullptr;
   if (!load_function(library, QTI_GET_LEGACY_YUV_UBWC_PLANE_INFO_SYMBOL,
                      &get_yuv_ubwc_plane_info) ||
       !load_function(library, QTI_GET_LEGACY_YUV_PLANE_INFO_SYMBOL,
                      &get_yuv_plane_info))
      return false;

   gr->get_legacy_yuv_ubwc_plane_info = get_yuv_ubwc_plane_info;
   gr->get_legacy_yuv_plane_info = get_yuv_plane_info;
   return true;
}

static bool
load_qti_legacy_ycbcr_ubwc_api(void *library, qti_metadata_gralloc *gr)
{
   /* SM8150/SDM845 public gralloc ABI.  GetYuvUbwcSPPlaneInfo fills an
    * android_ycbcr (not a PlaneLayoutInfo array) and takes the base as its
    * first argument, while GetYUVPlaneInfo is the same handle-aware helper as
    * the other legacy backends.  Both exact symbols must exist before this
    * backend is selected; absence only means this revision is not SM8150/
    * SDM845 and the caller falls through to a newer backend.
    */
   GetLegacyQtiYcbcrUbwcPlaneInfo get_ycbcr_ubwc_plane_info = nullptr;
   GetLegacyQtiYuvPlaneInfo get_yuv_plane_info = nullptr;
   if (!load_function(library,
                      QTI_GET_LEGACY_YCB_CR_UBWC_PLANE_INFO_SYMBOL,
                      &get_ycbcr_ubwc_plane_info) ||
       !load_function(library, QTI_GET_LEGACY_YUV_PLANE_INFO_SYMBOL,
                      &get_yuv_plane_info))
      return false;

   gr->get_legacy_ycbcr_ubwc_plane_info = get_ycbcr_ubwc_plane_info;
   gr->get_legacy_yuv_plane_info = get_yuv_plane_info;
   return true;
}

static bool
qti_modern_handle_is_compatible(const native_handle_t *handle)
{
   /* This is the private_handle_t ABI validated by the supplied SM8550 QTI
    * gralloc stack.  It is a runtime software-ABI gate, not a GPU-model gate:
    * no QTI entry point is called unless the incoming handle matches it.
    */
   return sizeof(void *) == 8 && handle &&
          handle->version == sizeof(native_handle_t) &&
          handle->numFds == QTI_HANDLE_NUM_FDS &&
          handle->numInts == QTI_MODERN_HANDLE_NUM_INTS &&
          handle->data[handle->numFds] == QTI_HANDLE_MAGIC;
}

static bool
qti_legacy_handle_has_common_header(const native_handle_t *handle)
{
   return sizeof(void *) == 8 && handle &&
          handle->version == sizeof(native_handle_t) &&
          handle->numFds == QTI_HANDLE_NUM_FDS &&
          handle->numInts >= QTI_LEGACY_BASE_HANDLE_NUM_INTS &&
          handle->data[handle->numFds] == QTI_HANDLE_MAGIC;
}

static const qti_legacy_handle_profile *
qti_get_legacy_handle_profile(const native_handle_t *handle)
{
   static constexpr qti_legacy_handle_profile profiles[] = {
      /*
       * The public SM8150/SM8250 private_handle_t keeps this entire common
       * prefix stable and conditionally appends reserved_size (one word),
       * custom_content_md_reserved_size (one word), and the UBWCP
       * linear_size/format pair.  We never inspect those optional fields;
       * enumerate every layout produced by that header instead of treating a
       * valid build-time tail as a different core ABI.
       */
      {QTI_LEGACY_ALL_OPTIONAL_WORDS_HANDLE_NUM_INTS,
       "four-optional-words"},
      {QTI_LEGACY_THREE_OPTIONAL_WORDS_HANDLE_NUM_INTS,
       "three-optional-words"},
      {QTI_LEGACY_TWO_OPTIONAL_WORDS_HANDLE_NUM_INTS,
       "two-optional-words"},
      {QTI_LEGACY_RESERVED_HANDLE_NUM_INTS, "reserved-size"},
      {QTI_LEGACY_BASE_HANDLE_NUM_INTS, "base"},
   };

   if (!qti_legacy_handle_has_common_header(handle))
      return nullptr;

   for (const qti_legacy_handle_profile &profile : profiles) {
      if (handle->numInts == profile.num_ints)
         return &profile;
   }

   return nullptr;
}

static bool
qti_legacy_handle_is_compatible(const native_handle_t *handle)
{
   return qti_get_legacy_handle_profile(handle) != nullptr;
}

static bool
qti_handle_is_compatible(const qti_metadata_gralloc *gr,
                         const native_handle_t *handle)
{
   return gr->backend && gr->backend->handle_is_compatible(handle);
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

struct qti_standard_buffer_metadata {
   uint32_t width;
   uint32_t height;
   uint32_t layer_count;
   int32_t requested_format;
   uint32_t drm_fourcc;
   uint64_t modifier;
   uint64_t allocation_size;
   uint32_t aligned_width;
   uint32_t aligned_height;
   bool protected_content;
   uint32_t private_flags;
};

static bool
buffer_description_matches_allocation_protection(
   const struct u_gralloc_buffer_handle *hnd, uint64_t allocation_usage)
{
   if (!hnd->has_usage)
      return true;

   /* AHardwareBuffer usage belongs to the GraphicBuffer wrapper, not to the
    * native handle itself.  In particular, AOSP Codec2 migrates a buffer by
    * wrapping the same native handle with its original usage ORed with the
    * destination consumer usage.  QTI's standard USAGE metadata instead
    * returns the immutable allocation-time usage from private_handle_t, so
    * comparing the complete masks would reject a valid migrated buffer.
    *
    * Protected content is different: Turnip has no protected-memory import
    * path, and a wrapper must not change whether the underlying allocation is
    * secure.  Keep that security boundary fail-closed.  Other usage bits may
    * have influenced the original allocation policy, but cannot be compared
    * as native-handle identity after AOSP has re-wrapped the allocation.
    */
   return !((hnd->usage ^ allocation_usage) & GRALLOC_USAGE_PROTECTED);
}

static bool
qti_get_standard_buffer_metadata(
   qti_metadata_gralloc *gr, const struct u_gralloc_buffer_handle *hnd,
   uint64_t dma_buf_size, qti_standard_buffer_metadata *metadata,
   struct u_gralloc_buffer_handle *validated_hnd)
{
   if (!gr->get_metadata_value || !hnd || !hnd->handle)
      return false;

   uint64_t width = 0;
   uint64_t height = 0;
   uint64_t layer_count = 0;
   int32_t requested_format = 0;
   uint32_t drm_fourcc = 0;
   uint64_t modifier = DRM_FORMAT_MOD_INVALID;
   uint64_t usage = 0;

   /* The public SM8550 implementation copies the uint32_t private-handle size
    * here even though standard Mapper metadata defines ALLOCATION_SIZE as a
    * uint64_t.  A zero-initialized uint64_t destination safely accepts that
    * 32-bit write and also remains correct if a newer helper fixes the width.
    */
   uint64_t allocation_size = 0;
   uint64_t aligned_width = 0;
   uint64_t aligned_height = 0;
   uint64_t protected_content = 0;
   uint32_t private_flags = 0;
   void *handle = const_cast<native_handle_t *>(hnd->handle);

   if (gr->get_metadata_value(handle, STANDARD_METADATA_WIDTH, &width) != 0 ||
       gr->get_metadata_value(handle, STANDARD_METADATA_HEIGHT, &height) != 0 ||
       gr->get_metadata_value(handle, STANDARD_METADATA_LAYER_COUNT,
                              &layer_count) != 0 ||
       gr->get_metadata_value(handle,
                              STANDARD_METADATA_PIXEL_FORMAT_REQUESTED,
                              &requested_format) != 0 ||
       gr->get_metadata_value(handle, STANDARD_METADATA_PIXEL_FORMAT_FOURCC,
                              &drm_fourcc) != 0 ||
        gr->get_metadata_value(handle, STANDARD_METADATA_PIXEL_FORMAT_MODIFIER,
                               &modifier) != 0 ||
        gr->get_metadata_value(handle, STANDARD_METADATA_USAGE, &usage) != 0 ||
        gr->get_metadata_value(handle, STANDARD_METADATA_ALLOCATION_SIZE,
                               &allocation_size) != 0 ||
        gr->get_metadata_value(handle, QTI_METADATA_ALIGNED_WIDTH_IN_PIXELS,
                               &aligned_width) != 0 ||
        gr->get_metadata_value(handle, QTI_METADATA_ALIGNED_HEIGHT_IN_PIXELS,
                               &aligned_height) != 0 ||
        gr->get_metadata_value(handle, STANDARD_METADATA_PROTECTED_CONTENT,
                               &protected_content) != 0 ||
        gr->get_metadata_value(handle, QTI_METADATA_PRIVATE_FLAGS,
                               &private_flags) != 0)
      return false;

    if (width == 0 || width > UINT32_MAX || height == 0 ||
        height > UINT32_MAX || layer_count != 1 ||
        modifier == DRM_FORMAT_MOD_INVALID || allocation_size == 0 ||
        allocation_size > dma_buf_size || aligned_width == 0 ||
        aligned_width > UINT32_MAX || aligned_height == 0 ||
        aligned_height > UINT32_MAX || aligned_width < width ||
        aligned_height < height || protected_content > 1)
       return false;

   const bool usage_is_protected = usage & GRALLOC_USAGE_PROTECTED;
   const bool flags_are_protected =
      private_flags & QTI_HANDLE_FLAG_SECURE_BUFFER;
   if (static_cast<bool>(protected_content) != usage_is_protected ||
       static_cast<bool>(protected_content) != flags_are_protected)
      return false;

   if (!buffer_description_matches_allocation_protection(hnd, usage))
      return false;

   if ((hnd->width && hnd->width != width) ||
       (hnd->height && hnd->height != height) ||
       (hnd->layer_count && hnd->layer_count != layer_count))
      return false;

    *metadata = {
       .width = static_cast<uint32_t>(width),
       .height = static_cast<uint32_t>(height),
       .layer_count = static_cast<uint32_t>(layer_count),
       .requested_format = requested_format,
       .drm_fourcc = drm_fourcc,
       .modifier = modifier,
       .allocation_size = allocation_size,
       .aligned_width = static_cast<uint32_t>(aligned_width),
       .aligned_height = static_cast<uint32_t>(aligned_height),
       .protected_content = protected_content != 0,
       .private_flags = private_flags,
    };

   *validated_hnd = *hnd;
   validated_hnd->width = metadata->width;
   validated_hnd->height = metadata->height;
   validated_hnd->layer_count = metadata->layer_count;
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
is_420sp_chroma_plane(const PlaneLayout &layout, bool cr_first,
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
          component.offsetInBits == (cr_first ? 8 : 0) &&
          component.sizeInBits == 8) {
         found_cb = true;
      } else if (is_standard_component(component, PLANE_COMPONENT_CR) &&
                 component.offsetInBits == (cr_first ? 0 : 8) &&
                 component.sizeInBits == 8) {
         found_cr = true;
      } else {
         return false;
      }
   }

   return found_cb && found_cr;
}

static bool
is_p010_luma_plane(const PlaneLayout &layout)
{
   if (layout.sampleIncrementInBits != 16 ||
       layout.horizontalSubsampling != 1 ||
       layout.verticalSubsampling != 1 || layout.components.size() != 1)
      return false;

   const PlaneLayoutComponent &component = layout.components[0];
   return is_standard_component(component, PLANE_COMPONENT_Y) &&
          component.offsetInBits == 6 && component.sizeInBits == 10;
}

static bool
is_p010_chroma_plane(const PlaneLayout &layout)
{
   if (layout.sampleIncrementInBits != 32 ||
       layout.horizontalSubsampling != 2 ||
       layout.verticalSubsampling != 2 || layout.components.size() != 2)
      return false;

   bool found_cb = false;
   bool found_cr = false;
   for (const PlaneLayoutComponent &component : layout.components) {
      if (is_standard_component(component, PLANE_COMPONENT_CB) &&
          component.offsetInBits == 6 && component.sizeInBits == 10) {
         found_cb = true;
      } else if (is_standard_component(component, PLANE_COMPONENT_CR) &&
                 component.offsetInBits == 22 &&
                 component.sizeInBits == 10) {
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
buffer_description_supported(const struct u_gralloc_buffer_handle *hnd)
{
   if (!hnd || (hnd->width == 0) != (hnd->height == 0))
      return false;

   /* u_gralloc's result has no array-layer stride.  Zero means that an older
    * caller could not provide the public description; a known multi-layer
    * allocation must not be flattened into a single-layer plane layout.
    */
   return hnd->layer_count <= 1;
}

static bool
buffer_description_matches_allocated_format(int32_t described_format,
                                            int32_t allocated_format)
{
   if (described_format == allocated_format)
      return true;

   /* QTI's public GetImplDefinedFormat() only resolves these two Android
    * descriptions to another physical format.  A mismatch for an explicit
    * RGB or YUV description is contradictory metadata, not permission to
    * reinterpret the allocation as the private handle format.
    */
   return described_format == HAL_PIXEL_FORMAT_IMPLEMENTATION_DEFINED ||
          described_format == HAL_PIXEL_FORMAT_YCBCR_420_888;
}

static bool
layout_covers_buffer_description(
   const PlaneLayout &layout, const struct u_gralloc_buffer_handle *hnd)
{
   if (!hnd->width || layout.sampleIncrementInBits == 0)
      return true;

   if (layout.horizontalSubsampling <= 0 ||
       layout.verticalSubsampling <= 0)
      return false;

   const uint64_t hsub =
      static_cast<uint64_t>(layout.horizontalSubsampling);
   const uint64_t vsub =
      static_cast<uint64_t>(layout.verticalSubsampling);
   const uint64_t min_width =
      (static_cast<uint64_t>(hnd->width) + hsub - 1) / hsub;
   const uint64_t min_height =
      (static_cast<uint64_t>(hnd->height) + vsub - 1) / vsub;

   /* Some mappers report aligned sample extents while others report logical
    * extents.  Both are valid, but a data plane smaller than the public
    * AHB/ANB description cannot contain the requested image.
    */
   return static_cast<uint64_t>(layout.widthInSamples) >= min_width &&
          static_cast<uint64_t>(layout.heightInSamples) >= min_height;
}

static bool
valid_layout(const PlaneLayout &layout, uint64_t allocation_size,
             const struct u_gralloc_buffer_handle *hnd)
{
   if (!layout_value_fits_int(layout.offsetInBytes) ||
       !layout_value_fits_int(layout.strideInBytes) ||
       layout.totalSizeInBytes <= 0 ||
       !layout_value_fits_int(layout.widthInSamples) ||
       !layout_value_fits_int(layout.heightInSamples) ||
       layout.widthInSamples == 0 || layout.heightInSamples == 0 ||
       !layout_value_fits_int(layout.sampleIncrementInBits) ||
       !layout_value_fits_int(layout.horizontalSubsampling) ||
       !layout_value_fits_int(layout.verticalSubsampling) ||
       layout.horizontalSubsampling == 0 ||
       layout.verticalSubsampling == 0 ||
       layout.components.size() > MAX_COMPONENTS_PER_PLANE ||
       !layout_covers_buffer_description(layout, hnd))
      return false;

   if (layout.sampleIncrementInBits > 0) {
      const uint64_t width =
         static_cast<uint64_t>(layout.widthInSamples);
      const uint64_t height =
         static_cast<uint64_t>(layout.heightInSamples);
      const uint64_t sample_bits =
         static_cast<uint64_t>(layout.sampleIncrementInBits);
      const uint64_t stride =
         static_cast<uint64_t>(layout.strideInBytes);
      const uint64_t size =
         static_cast<uint64_t>(layout.totalSizeInBytes);

      /* A range being inside the dma-buf is not enough: it also has to cover
       * every sample described by this plane.  Keep metadata-only planes out
       * of this calculation because their sample increment is intentionally
       * zero and their geometry is modifier-specific.
       */
      if (layout.strideInBytes <= 0 ||
          width > (UINT64_MAX - 7) / sample_bits)
         return false;

      const uint64_t min_row_bytes = (width * sample_bits + 7) / 8;
      if (stride < min_row_bytes || height > size / stride)
         return false;
   }

   const uint64_t offset = static_cast<uint64_t>(layout.offsetInBytes);
   const uint64_t size = static_cast<uint64_t>(layout.totalSizeInBytes);
   if (offset < allocation_size && size <= allocation_size - offset)
      return true;

   /* QTI's public GetRGBPlaneInfo() reports an RGB UBWC data offset after
    * the leading metadata, but reports the size of the entire allocation
    * rather than the size of the data range beginning at that offset.  Only
    * accept that exact, self-bounded RGB representation here.  The later
    * copy_qcom_ubwc_layout() path still requires one data plane, a non-zero
    * metadata prefix, and totalSizeInBytes == allocation_size before exposing
    * the modifier layout to Turnip.
    */
   return offset > 0 && offset < allocation_size &&
          size == allocation_size &&
          is_standard_rgb_plane(layout);
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
is_yv12_plane(const PlaneLayout &layout, int64_t component_type,
              int64_t subsampling)
{
   if (layout.components.size() != 1 ||
       layout.sampleIncrementInBits != 8 ||
       layout.horizontalSubsampling != subsampling ||
       layout.verticalSubsampling != subsampling ||
       layout.offsetInBytes < 0 || layout.strideInBytes <= 0 ||
       layout.widthInSamples <= 0 || layout.heightInSamples <= 0 ||
       layout.totalSizeInBytes <= 0)
      return false;

   const PlaneLayoutComponent &component = layout.components[0];
   return is_standard_component(component, component_type) &&
          component.offsetInBits == 0 && component.sizeInBits == 8;
}

static bool
normalize_qti_p010_layouts(
   const std::vector<PlaneLayout> &layouts,
   const struct u_gralloc_buffer_handle *hnd,
   std::vector<PlaneLayout> *normalized)
{
   if (layouts.size() != 2)
      return false;

   const PlaneLayout *y = nullptr;
   const PlaneLayout *uv = nullptr;
   for (const PlaneLayout &layout : layouts) {
      const PlaneLayout **plane = nullptr;
      if (is_p010_luma_plane(layout))
         plane = &y;
      else if (is_p010_chroma_plane(layout))
         plane = &uv;
      else
         return false;

      if (*plane)
         return false;
      *plane = &layout;
   }

   if (!y || !uv || (y->widthInSamples & 1) ||
       (y->heightInSamples & 1) ||
       uv->widthInSamples != y->widthInSamples / 2 ||
       uv->heightInSamples != y->heightInSamples / 2 ||
       y->strideInBytes != uv->strideInBytes ||
       (y->strideInBytes & 1) || y->offsetInBytes != 0 ||
       static_cast<uint64_t>(uv->offsetInBytes) !=
          static_cast<uint64_t>(y->totalSizeInBytes))
      return false;

   /* Standard metadata and AHardwareBuffer describe the logical extent;
    * QTI PlaneLayout may additionally include aligned rows in totalSize.
    * Require an exact logical match when the caller supplied that extent,
    * while retaining the already-validated aligned allocation footprint.
    */
   if ((hnd->width &&
        static_cast<uint64_t>(y->widthInSamples) != hnd->width) ||
       (hnd->height &&
        static_cast<uint64_t>(y->heightInSamples) != hnd->height))
      return false;

   normalized->clear();
   normalized->reserve(2);
   normalized->push_back(*y);
   normalized->push_back(*uv);
   return true;
}

static bool
normalize_qti_yv12_layouts(const std::vector<PlaneLayout> &layouts,
                             int32_t hal_format, uint32_t width,
                             uint32_t height, uint32_t aligned_width,
                             uint32_t aligned_height,
                             std::vector<PlaneLayout> *normalized)
{
   if (hal_format != HAL_PIXEL_FORMAT_YV12 || layouts.size() != 3)
      return false;

   const PlaneLayout *y = nullptr;
   const PlaneLayout *cb = nullptr;
   const PlaneLayout *cr = nullptr;

   for (const PlaneLayout &layout : layouts) {
      const PlaneLayout **plane = nullptr;
      if (is_yv12_plane(layout, PLANE_COMPONENT_Y, 1))
         plane = &y;
      else if (is_yv12_plane(layout, PLANE_COMPONENT_CB, 2))
         plane = &cb;
      else if (is_yv12_plane(layout, PLANE_COMPONENT_CR, 2))
         plane = &cr;
      else
         return false;

      if (*plane)
         return false;
      *plane = &layout;
   }

   if (!y || !cb || !cr || width == 0 || height == 0 ||
       aligned_width == 0 || aligned_height == 0 || (width & 1) ||
       (height & 1) || (aligned_width & 1) || (aligned_height & 1) ||
       aligned_width < width || aligned_height < height ||
       y->widthInSamples != width || y->heightInSamples != height ||
       y->strideInBytes < aligned_width ||
       cb->widthInSamples != width / 2 ||
       cr->widthInSamples != cb->widthInSamples ||
       cb->heightInSamples != height / 2 ||
       cr->heightInSamples != cb->heightInSamples ||
       cb->strideInBytes < cb->widthInSamples ||
       cr->strideInBytes != cb->strideInBytes)
      return false;

   const uint64_t y_stride = static_cast<uint64_t>(y->strideInBytes);
   const uint64_t cb_stride = static_cast<uint64_t>(cb->strideInBytes);
   const uint64_t cr_stride = static_cast<uint64_t>(cr->strideInBytes);
   const uint64_t y_height = static_cast<uint64_t>(y->heightInSamples);
   const uint64_t chroma_height = y_height / 2;

   if ((y_stride & 15) != 0 || (cb_stride & 15) != 0 ||
       cb_stride != cr_stride || cb_stride < y_stride / 2 ||
       cb_stride < cb->widthInSamples ||
       y_height > UINT64_MAX / y_stride ||
       chroma_height > UINT64_MAX / cb_stride)
      return false;

   const uint64_t min_y_size = y_stride * y_height;
   const uint64_t min_chroma_size = cb_stride * chroma_height;
   const uint64_t y_size = static_cast<uint64_t>(y->totalSizeInBytes);
   const uint64_t chroma_size =
      static_cast<uint64_t>(cb->totalSizeInBytes);
   if (y_size < min_y_size || chroma_size < min_chroma_size ||
       static_cast<uint64_t>(cr->totalSizeInBytes) != chroma_size ||
       y->offsetInBytes != 0 || cr->offsetInBytes < y->offsetInBytes ||
       cb->offsetInBytes < cr->offsetInBytes ||
       static_cast<uint64_t>(cr->offsetInBytes) != y_size ||
       y_size > UINT64_MAX - chroma_size ||
       static_cast<uint64_t>(cb->offsetInBytes) != y_size + chroma_size)
      return false;

   /* QTI reports components in logical Y-Cb-Cr order even though YV12 is
    * stored as Y-Cr-Cb.  u_gralloc uses DRM plane order; vk_android will
    * swap the chroma layouts back to Vulkan's Y-Cb-Cr order later.
    */
   normalized->clear();
   normalized->reserve(3);
   normalized->push_back(*y);
   normalized->push_back(*cr);
   normalized->push_back(*cb);
   return true;
}

static bool
normalize_qti_420sp_layouts(const std::vector<PlaneLayout> &layouts,
                            bool cr_first, bool allow_omitted_components,
                            std::vector<PlaneLayout> *normalized,
                            bool *compressed)
{
   /* Component omission is enabled only for exact QTI formats whose public
    * conversion helper is known to omit otherwise authoritative labels.  The
    * caller's format profile still has to agree with the resulting linear or
    * compressed plane structure, so an omitted label never guesses between
    * those two physical layouts.
    */
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
         else if (is_420sp_chroma_plane(layouts[i], cr_first,
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
       (metadata_count != 0 && metadata_count != data_count))
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

static bool
qti_drm_fourcc_is_standard_rgb(uint32_t fourcc)
{
   /* RGB formats emitted by the public QTI GetDRMFormat() implementation.
    * Keep this explicit: an unknown or YUV FourCC must not be imported merely
    * because another metadata field happened to describe RGB components.
    */
   switch (fourcc) {
   case DRM_FORMAT_ABGR8888:
   case DRM_FORMAT_ABGR1555:
   case DRM_FORMAT_ABGR4444:
   case DRM_FORMAT_ARGB8888:
   case DRM_FORMAT_XBGR8888:
   case DRM_FORMAT_XRGB8888:
   case DRM_FORMAT_BGR888:
   case DRM_FORMAT_BGR565:
   case DRM_FORMAT_ABGR2101010:
   case DRM_FORMAT_BGRA1010102:
   case DRM_FORMAT_XBGR2101010:
   case DRM_FORMAT_BGRX1010102:
   case DRM_FORMAT_ARGB2101010:
   case DRM_FORMAT_RGBA1010102:
   case DRM_FORMAT_XRGB2101010:
   case DRM_FORMAT_RGBX1010102:
   case DRM_FORMAT_ABGR16161616F:
      return true;
   default:
      return false;
   }
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
   int fd_indices[ARRAY_SIZE(out->fds)] = {};
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

      const uint64_t start =
         static_cast<uint64_t>(layout.offsetInBytes);
      const uint64_t end =
         start + static_cast<uint64_t>(layout.totalSizeInBytes);
      for (size_t j = 0; j < i; j++) {
         if (fd_indices[j] != fd_index)
            continue;

         const uint64_t other_start =
            static_cast<uint64_t>(layouts[j].offsetInBytes);
         const uint64_t other_end =
            other_start +
            static_cast<uint64_t>(layouts[j].totalSizeInBytes);
         if (start < other_end && other_start < end)
            return -EINVAL;
      }

      fd_indices[i] = fd_index;
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
          static_cast<uint64_t>(data.offsetInBytes) >= allocation_size ||
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
   int32_t allocated_format, uint32_t drm_fourcc, uint64_t modifier,
   bool validate_private_flags, uint32_t private_flags,
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

   if (drm_fourcc == 0) {
      const int inferred_fourcc =
         get_fourcc_from_hal_format(allocated_format);
      if (inferred_fourcc == -1)
         return -ENOTSUP;
      drm_fourcc = inferred_fourcc;
   }

   if (!qti_drm_fourcc_is_standard_rgb(drm_fourcc))
      return -EINVAL;

   if (hnd->pixel_stride > 0) {
      if ((data->sampleIncrementInBits & 7) != 0)
         return -EINVAL;

      const uint64_t bytes_per_sample =
         static_cast<uint64_t>(data->sampleIncrementInBits) / 8;
      const uint64_t pixel_stride = static_cast<uint64_t>(hnd->pixel_stride);
      if (bytes_per_sample == 0 ||
          pixel_stride > UINT64_MAX / bytes_per_sample ||
          static_cast<uint64_t>(data->strideInBytes) !=
             pixel_stride * bytes_per_sample)
         return -EINVAL;
   }

   if (modifier == DRM_FORMAT_MOD_INVALID) {
      modifier = metadata_count == 0 && data->offsetInBytes == 0
                    ? DRM_FORMAT_MOD_LINEAR
                    : DRM_FORMAT_MOD_QCOM_COMPRESSED;
   }

   const bool layout_is_compressed =
      metadata_count != 0 || data->offsetInBytes > 0;

   /* PlaneLayout and DRM_FORMAT_MOD_QCOM_COMPRESSED cannot distinguish
    * ordinary UBWC from UBWC-PI.  Without the immutable QTI private flags,
    * accepting a compressed allocation would silently guess that semantic.
    */
   if (layout_is_compressed && !validate_private_flags)
      return -ENOTSUP;

   if (validate_private_flags &&
       layout_is_compressed !=
          !!(private_flags & QTI_HANDLE_FLAG_UBWC_ALIGNED))
      return -EINVAL;

   /* The public SM8550 GetDRMFormat() implementation does not set the QCOM
    * modifier for every UBWC-capable RGB FourCC (notably RGBA_8888), leaving
    * its zero-initialized output equal to LINEAR.  Its public GetPlaneLayout()
    * still describes the actual allocation unambiguously: linear RGB starts
    * at byte zero, while UBWC RGB has a leading metadata range before the
    * sole data plane.  Accept that one documented inconsistency and derive
    * COMPRESSED from the complete PlaneLayout.  No unknown non-linear
    * modifier is reinterpreted this way.
    */
   if (modifier == DRM_FORMAT_MOD_LINEAR && layout_is_compressed)
      modifier = DRM_FORMAT_MOD_QCOM_COMPRESSED;

   out->drm_fourcc = drm_fourcc;
   out->modifier = modifier;
   if (modifier == DRM_FORMAT_MOD_LINEAR) {
      if (layout_is_compressed)
         return -EINVAL;
      return copy_linear_layout(hnd->handle, layouts, allocation_size, out);
   }

   /* QTI may either expose a separate RGB metadata plane or fold the leading
    * metadata size into a single data PlaneLayout.  Both representations are
    * handled by copy_qcom_ubwc_layout() and validated against the dma-buf.
    */
   if (modifier == DRM_FORMAT_MOD_QCOM_COMPRESSED && layout_is_compressed)
      return copy_qcom_ubwc_layout(hnd->handle, layouts, allocation_size, out);

   return -ENOTSUP;
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

static bool
qti_legacy_description_can_be_nv12_ubwc(int32_t hal_format)
{
   /* IMPLEMENTATION_DEFINED and YCbCr_420_888 are the two standard Android
    * descriptions under which gralloc may select this private physical
    * format.  The private handle format and both vendor layout helpers still
    * have to prove that the allocation is NV12 UBWC below.
    */
   return hal_format ==
             QTI_HAL_PIXEL_FORMAT_YCBCR_420_SP_VENUS_UBWC ||
          hal_format == HAL_PIXEL_FORMAT_IMPLEMENTATION_DEFINED ||
          hal_format == HAL_PIXEL_FORMAT_YCBCR_420_888;
}

static bool
qti_legacy_format_can_use_fallback(
   const struct u_gralloc_buffer_handle *hnd, int32_t private_format)
{
   /* Once the legacy handle ABI is known, its internal format can safely
    * distinguish standard allocations from other private producer formats.
    * Keep the established fallback for standard RGB/YUV (including the
    * historically ambiguous IMPLEMENTATION_DEFINED UI case), but never let
    * an unknown private YUV format be guessed as RGB.
    */
   return can_use_fallback_for_known_non_yuv(hnd) ||
          private_format == HAL_PIXEL_FORMAT_IMPLEMENTATION_DEFINED ||
          is_hal_format_yuv(private_format) ||
          get_hal_format_bpp(private_format) > 0;
}

enum class qti_yuv_layout_kind {
   NV12_LINEAR,
   NV21_LINEAR,
   YV12_LINEAR,
   P010_LINEAR,
   NV12_UBWC,
};

struct qti_yuv_format_profile {
   int32_t hal_format;
   uint32_t drm_fourcc;
   qti_yuv_layout_kind layout;
};

static const qti_yuv_format_profile *
qti_get_yuv_format_profile(int32_t format)
{
   static constexpr qti_yuv_format_profile profiles[] = {
      {QTI_HAL_PIXEL_FORMAT_YCBCR_420_SP, DRM_FORMAT_NV12,
       qti_yuv_layout_kind::NV12_LINEAR},
      {QTI_HAL_PIXEL_FORMAT_NV12_ENCODEABLE, DRM_FORMAT_NV12,
       qti_yuv_layout_kind::NV12_LINEAR},
      {QTI_HAL_PIXEL_FORMAT_YCBCR_420_SP_VENUS, DRM_FORMAT_NV12,
       qti_yuv_layout_kind::NV12_LINEAR},
      {QTI_HAL_PIXEL_FORMAT_NV12_HEIF, DRM_FORMAT_NV12,
       qti_yuv_layout_kind::NV12_LINEAR},
      {QTI_HAL_PIXEL_FORMAT_NV12_LINEAR_FLEX, DRM_FORMAT_NV12,
       qti_yuv_layout_kind::NV12_LINEAR},
      {HAL_PIXEL_FORMAT_YCrCb_420_SP, DRM_FORMAT_NV21,
       qti_yuv_layout_kind::NV21_LINEAR},
      {QTI_HAL_PIXEL_FORMAT_NV21_ENCODEABLE, DRM_FORMAT_NV21,
       qti_yuv_layout_kind::NV21_LINEAR},
      {QTI_HAL_PIXEL_FORMAT_YCRCB_420_SP_ADRENO, DRM_FORMAT_NV21,
       qti_yuv_layout_kind::NV21_LINEAR},
      {QTI_HAL_PIXEL_FORMAT_NV21_ZSL, DRM_FORMAT_NV21,
       qti_yuv_layout_kind::NV21_LINEAR},
      {QTI_HAL_PIXEL_FORMAT_YCRCB_420_SP_VENUS, DRM_FORMAT_NV21,
       qti_yuv_layout_kind::NV21_LINEAR},
      {HAL_PIXEL_FORMAT_YV12, DRM_FORMAT_YVU420,
       qti_yuv_layout_kind::YV12_LINEAR},
      {QTI_HAL_PIXEL_FORMAT_YCBCR_420_P010, DRM_FORMAT_P010,
       qti_yuv_layout_kind::P010_LINEAR},
      {QTI_HAL_PIXEL_FORMAT_YCBCR_420_P010_VENUS, DRM_FORMAT_P010,
       qti_yuv_layout_kind::P010_LINEAR},
      {QTI_HAL_PIXEL_FORMAT_YCBCR_420_SP_VENUS_UBWC, DRM_FORMAT_NV12,
       qti_yuv_layout_kind::NV12_UBWC},
   };

   for (const qti_yuv_format_profile &profile : profiles) {
      if (profile.hal_format == format)
         return &profile;
   }

   return nullptr;
}

static bool
qti_legacy_rgb_format_is_supported(int32_t format)
{
   static constexpr int32_t formats[] = {
      HAL_PIXEL_FORMAT_RGBA_8888,
      HAL_PIXEL_FORMAT_RGBX_8888,
      HAL_PIXEL_FORMAT_RGB_888,
      HAL_PIXEL_FORMAT_RGB_565,
      HAL_PIXEL_FORMAT_BGRA_8888,
      HAL_PIXEL_FORMAT_RGBA_FP16,
      HAL_PIXEL_FORMAT_RGBA_1010102,
      QTI_HAL_PIXEL_FORMAT_RGBA_5551,
      QTI_HAL_PIXEL_FORMAT_RGBA_4444,
      QTI_HAL_PIXEL_FORMAT_BGRX_8888,
      QTI_HAL_PIXEL_FORMAT_BGR_565,
      QTI_HAL_PIXEL_FORMAT_ARGB_2101010,
      QTI_HAL_PIXEL_FORMAT_RGBX_1010102,
      QTI_HAL_PIXEL_FORMAT_XRGB_2101010,
      QTI_HAL_PIXEL_FORMAT_BGRA_1010102,
      QTI_HAL_PIXEL_FORMAT_ABGR_2101010,
      QTI_HAL_PIXEL_FORMAT_BGRX_1010102,
      QTI_HAL_PIXEL_FORMAT_XBGR_2101010,
   };

   for (int32_t candidate : formats) {
      if (candidate == format)
         return true;
   }

   return false;
}

struct qti_legacy_handle_info {
   const qti_legacy_handle_profile *profile;
   int32_t flags;
   int32_t width;
   int32_t height;
   int32_t unaligned_width;
   int32_t unaligned_height;
   int32_t format;
   int32_t layer_count;
   uint64_t usage;
   uint32_t declared_size;
   uintptr_t base;
   uint64_t allocation_size;
};

static uint64_t
qti_legacy_read_u64(const native_handle_t *handle, int index)
{
   uint64_t value;
   memcpy(&value, &handle->data[index], sizeof(value));
   return value;
}

static bool
qti_get_legacy_handle_info(const native_handle_t *handle,
                           qti_legacy_handle_info *info)
{
   const qti_legacy_handle_profile *profile =
      qti_get_legacy_handle_profile(handle);
   if (!profile)
      return false;

   *info = {
      .profile = profile,
      .flags = handle->data[QTI_LEGACY_HANDLE_FLAGS_INDEX],
      .width = handle->data[QTI_LEGACY_HANDLE_WIDTH_INDEX],
      .height = handle->data[QTI_LEGACY_HANDLE_HEIGHT_INDEX],
      .unaligned_width =
         handle->data[QTI_LEGACY_HANDLE_UNALIGNED_WIDTH_INDEX],
      .unaligned_height =
         handle->data[QTI_LEGACY_HANDLE_UNALIGNED_HEIGHT_INDEX],
      .format = handle->data[QTI_LEGACY_HANDLE_FORMAT_INDEX],
      .layer_count = handle->data[QTI_LEGACY_HANDLE_LAYER_COUNT_INDEX],
      .usage = qti_legacy_read_u64(handle, QTI_LEGACY_HANDLE_USAGE_INDEX),
      .declared_size =
         static_cast<uint32_t>(handle->data[QTI_LEGACY_HANDLE_SIZE_INDEX]),
      .base = static_cast<uintptr_t>(
         qti_legacy_read_u64(handle, QTI_LEGACY_HANDLE_BASE_INDEX)),
   };

   if (info->width <= 0 || info->height <= 0 ||
       info->unaligned_width <= 0 || info->unaligned_height <= 0 ||
       info->width < info->unaligned_width ||
       info->height < info->unaligned_height || info->layer_count != 1 ||
       info->declared_size == 0 ||
       handle->data[QTI_LEGACY_HANDLE_OFFSET_INDEX] != 0 ||
       !get_dma_buf_allocation_size(handle, &info->allocation_size) ||
       info->declared_size > info->allocation_size)
      return false;

   return true;
}

static int
qti_validate_legacy_buffer_contract(
   const qti_legacy_handle_info &info,
   const struct u_gralloc_buffer_handle *hnd)
{
   /* Validate the public Android description before any format-specific
    * helper or fallback is considered.  Otherwise an unmodelled linear YUV
    * allocation could be routed through fallback using a contradictory
    * explicit RGB description (or vice versa).  Only the two abstract Android
    * formats resolved by QTI's public GetImplDefinedFormat() may differ from
    * the private handle's allocated format.
    */
   if (!buffer_description_matches_allocated_format(hnd->hal_format,
                                                    info.format))
      return -EINVAL;

   const bool secure_flag =
      (static_cast<uint32_t>(info.flags) &
       QTI_HANDLE_FLAG_SECURE_BUFFER) != 0;
   const bool protected_usage =
      (info.usage & GRALLOC_USAGE_PROTECTED) != 0;

   /* Public QTI GetCustomFormatFlags() derives the private secure bit from
    * GRALLOC_USAGE_PROTECTED.  A disagreement is corrupt metadata; agreement
    * on protected means a real secure allocation, which Turnip cannot import
    * while protectedMemory is false.
    */
   if (secure_flag != protected_usage)
      return -EINVAL;

   if (!buffer_description_matches_allocation_protection(hnd, info.usage))
      return -EINVAL;

   if (secure_flag)
      return -ENOTSUP;

   /* The native handle stores the allocator's logical geometry.  Cross-check
    * those immutable fields before calling a legacy helper with private-handle
    * dimensions.  Usage protection was checked separately above because the
    * remaining wrapper usage bits are intentionally allowed to evolve.  Zero
    * remains the explicit "unknown to this caller" value used by older Mesa
    * entry points.
    */
   if ((hnd->width &&
        hnd->width != static_cast<uint32_t>(info.unaligned_width)) ||
       (hnd->height &&
        hnd->height != static_cast<uint32_t>(info.unaligned_height)) ||
       (hnd->layer_count &&
        hnd->layer_count != static_cast<uint32_t>(info.layer_count)))
      return -EINVAL;

   return 0;
}

static bool
qti_legacy_description_matches(
   const struct u_gralloc_buffer_handle *hnd,
   const qti_yuv_format_profile &profile)
{
   return buffer_description_matches_allocated_format(hnd->hal_format,
                                                      profile.hal_format);
}

static bool
qti_legacy_ycbcr_is_empty(const struct android_ycbcr &layout)
{
   return !layout.y && !layout.cb && !layout.cr && layout.ystride == 0 &&
          layout.cstride == 0 && layout.chroma_step == 0;
}

static bool
qti_legacy_pointer_matches(uintptr_t base, uint32_t offset,
                           const void *pointer)
{
   return offset <= UINTPTR_MAX - base &&
          reinterpret_cast<uintptr_t>(pointer) == base + offset;
}

static bool
qti_legacy_plane_range_fits(const LegacyQtiPlaneLayoutInfo &layout,
                            uint64_t allocation_size)
{
   return layout.offset <= INT_MAX && layout.size > 0 &&
          layout.offset <= allocation_size &&
          layout.size <= allocation_size - layout.offset;
}

static bool
qti_legacy_plane_is_valid(const LegacyQtiPlaneLayoutInfo &layout,
                          uint32_t component, uint32_t h_subsampling,
                          uint32_t v_subsampling, int32_t step,
                          const qti_legacy_handle_info &handle_info)
{
   if (layout.component != component ||
       layout.horizontal_subsampling != h_subsampling ||
       layout.vertical_subsampling != v_subsampling ||
       layout.step != step || layout.stride <= 0 ||
       layout.stride_bytes <= 0 || layout.scanlines <= 0 ||
       !qti_legacy_plane_range_fits(layout,
                                    handle_info.allocation_size) ||
       layout.offset > handle_info.declared_size ||
       layout.size > handle_info.declared_size - layout.offset)
      return false;

   const uint64_t stride = static_cast<uint64_t>(layout.stride_bytes);
   const uint64_t scanlines = static_cast<uint64_t>(layout.scanlines);
   return scanlines <= UINT64_MAX / stride &&
          stride * scanlines <= layout.size;
}

static bool
qti_legacy_ranges_do_not_overlap(const LegacyQtiPlaneLayoutInfo *layouts,
                                 size_t count)
{
   for (size_t i = 0; i < count; i++) {
      const uint64_t start = layouts[i].offset;
      const uint64_t end = start + layouts[i].size;
      for (size_t j = 0; j < i; j++) {
         const uint64_t other_start = layouts[j].offset;
         const uint64_t other_end = other_start + layouts[j].size;
         if (start < other_end && other_start < end)
            return false;
      }
   }

   return true;
}

static bool
qti_legacy_validate_semiplanar_layout(
   const qti_legacy_handle_info &handle_info,
   qti_yuv_layout_kind layout_kind,
   const LegacyQtiPlaneLayoutInfo *layouts,
   const struct android_ycbcr (&ycbcr)[2])
{
   const LegacyQtiPlaneLayoutInfo &y = layouts[0];
   const LegacyQtiPlaneLayoutInfo &uv = layouts[1];
   if (!qti_legacy_plane_is_valid(
          y, QTI_LEGACY_PLANE_COMPONENT_Y, 0, 0, 1, handle_info) ||
       !qti_legacy_plane_is_valid(
          uv, QTI_LEGACY_PLANE_COMPONENT_CB |
                 QTI_LEGACY_PLANE_COMPONENT_CR,
          1, 1, 2, handle_info) ||
       y.stride != handle_info.width || uv.stride != handle_info.width ||
       y.stride_bytes < handle_info.unaligned_width ||
       uv.stride_bytes < handle_info.unaligned_width ||
       y.scanlines < handle_info.unaligned_height ||
       uv.scanlines < (handle_info.unaligned_height + 1) / 2 ||
       y.offset != 0 ||
       static_cast<uint64_t>(uv.offset) <
          static_cast<uint64_t>(y.offset) + y.size ||
       !qti_legacy_ranges_do_not_overlap(layouts, 2) ||
       !qti_legacy_ycbcr_is_empty(ycbcr[1]) ||
       ycbcr[0].ystride != static_cast<size_t>(y.stride_bytes) ||
       ycbcr[0].cstride != static_cast<size_t>(uv.stride_bytes) ||
       ycbcr[0].chroma_step != 2 ||
       !qti_legacy_pointer_matches(handle_info.base, y.offset, ycbcr[0].y))
      return false;

   if (uv.offset == UINT32_MAX)
      return false;

   const bool nv21 = layout_kind == qti_yuv_layout_kind::NV21_LINEAR;
   const uint32_t cb_offset = uv.offset + (nv21 ? 1 : 0);
   const uint32_t cr_offset = uv.offset + (nv21 ? 0 : 1);
   return qti_legacy_pointer_matches(handle_info.base, cb_offset,
                                     ycbcr[0].cb) &&
          qti_legacy_pointer_matches(handle_info.base, cr_offset,
                                     ycbcr[0].cr);
}

static bool
qti_legacy_validate_p010_layout(
   const qti_legacy_handle_info &handle_info,
   const LegacyQtiPlaneLayoutInfo *layouts,
   const struct android_ycbcr (&ycbcr)[2])
{
   const LegacyQtiPlaneLayoutInfo &y = layouts[0];
   const LegacyQtiPlaneLayoutInfo &uv = layouts[1];

   if ((handle_info.unaligned_width & 1) ||
       (handle_info.unaligned_height & 1) ||
       handle_info.width > INT_MAX / 2)
      return false;

   const uint64_t expected_stride_bytes =
      static_cast<uint64_t>(handle_info.width) * 2;
   const uint64_t y_rows =
      static_cast<uint64_t>(handle_info.unaligned_height);
   const uint64_t uv_rows = y_rows / 2;

   /* The public SM8150 helper reports the standard P010 chroma scanline
    * count as the full aligned height while its byte range correctly stores
    * half-height 4:2:0 chroma.  Validate the logical AOSP P010 footprint
    * against the bounded range instead of trusting that one inconsistent
    * diagnostic field.  The Venus variant reports the half height directly.
    */
   if (y.component != QTI_LEGACY_PLANE_COMPONENT_Y ||
       uv.component != (QTI_LEGACY_PLANE_COMPONENT_CB |
                        QTI_LEGACY_PLANE_COMPONENT_CR) ||
       y.horizontal_subsampling != 0 || y.vertical_subsampling != 0 ||
       uv.horizontal_subsampling != 1 || uv.vertical_subsampling != 1 ||
       y.step != 2 || uv.step != 4 || y.stride != handle_info.width ||
       uv.stride != handle_info.width || y.stride_bytes <= 0 ||
       uv.stride_bytes <= 0 ||
       static_cast<uint64_t>(y.stride_bytes) != expected_stride_bytes ||
       static_cast<uint64_t>(uv.stride_bytes) != expected_stride_bytes ||
       y.scanlines < handle_info.unaligned_height ||
       uv.scanlines < handle_info.unaligned_height / 2 ||
       !qti_legacy_plane_range_fits(y, handle_info.allocation_size) ||
       !qti_legacy_plane_range_fits(uv, handle_info.allocation_size) ||
       y.offset > handle_info.declared_size ||
       y.size > handle_info.declared_size - y.offset ||
       uv.offset > handle_info.declared_size ||
       uv.size > handle_info.declared_size - uv.offset ||
       y_rows > y.size / expected_stride_bytes ||
       uv_rows > uv.size / expected_stride_bytes || y.offset != 0 ||
       static_cast<uint64_t>(uv.offset) !=
          static_cast<uint64_t>(y.offset) + y.size ||
       !qti_legacy_ranges_do_not_overlap(layouts, 2) ||
       !qti_legacy_ycbcr_is_empty(ycbcr[1]) ||
       ycbcr[0].ystride != static_cast<size_t>(y.stride_bytes) ||
       ycbcr[0].cstride != static_cast<size_t>(uv.stride_bytes) ||
       ycbcr[0].chroma_step != 4 ||
       !qti_legacy_pointer_matches(handle_info.base, y.offset, ycbcr[0].y) ||
       !qti_legacy_pointer_matches(handle_info.base, uv.offset,
                                   ycbcr[0].cb))
      return false;

   if (uv.offset > UINT32_MAX - 2)
      return false;

   /* CopyPlaneLayoutInfotoAndroidYcbcr() advances Cr by one byte for every
    * semiplanar format, including P010.  Accept that public-helper result as
    * well as the correct next 16-bit sample address; neither pointer is used
    * to derive the Vulkan plane layout.
    */
   return qti_legacy_pointer_matches(handle_info.base, uv.offset + 1,
                                     ycbcr[0].cr) ||
          qti_legacy_pointer_matches(handle_info.base, uv.offset + 2,
                                     ycbcr[0].cr);
}

static int
qti_copy_legacy_semiplanar_layout(
   const native_handle_t *handle, const qti_yuv_format_profile &format,
   const LegacyQtiPlaneLayoutInfo *layouts,
   struct u_gralloc_buffer_basic_info *out)
{
   out->drm_fourcc = format.drm_fourcc;
   out->modifier = DRM_FORMAT_MOD_LINEAR;
   out->num_planes = 2;
   out->fds[0] = out->fds[1] = handle->data[0];
   out->offsets[0] = static_cast<int>(layouts[0].offset);
   out->offsets[1] = static_cast<int>(layouts[1].offset);
   out->strides[0] = layouts[0].stride_bytes;
   out->strides[1] = layouts[1].stride_bytes;
   return 0;
}

static bool
qti_legacy_validate_yv12_layout(
   const qti_legacy_handle_info &handle_info,
   const LegacyQtiPlaneLayoutInfo *layouts,
   const struct android_ycbcr (&ycbcr)[2])
{
   const LegacyQtiPlaneLayoutInfo &y = layouts[0];
   const LegacyQtiPlaneLayoutInfo &cb = layouts[1];
   const LegacyQtiPlaneLayoutInfo &cr = layouts[2];
   if (!qti_legacy_plane_is_valid(
          y, QTI_LEGACY_PLANE_COMPONENT_Y, 0, 0, 1, handle_info) ||
       !qti_legacy_plane_is_valid(
          cb, QTI_LEGACY_PLANE_COMPONENT_CB, 1, 1, 1, handle_info) ||
       !qti_legacy_plane_is_valid(
          cr, QTI_LEGACY_PLANE_COMPONENT_CR, 1, 1, 1, handle_info) ||
       y.stride_bytes < handle_info.unaligned_width ||
       cb.stride_bytes < (handle_info.unaligned_width + 1) / 2 ||
       cr.stride_bytes != cb.stride_bytes ||
       y.scanlines < handle_info.unaligned_height ||
       cb.scanlines < (handle_info.unaligned_height + 1) / 2 ||
       cr.scanlines != cb.scanlines || y.offset != 0 ||
       cr.offset != static_cast<uint64_t>(y.offset) + y.size ||
       cb.offset != static_cast<uint64_t>(cr.offset) + cr.size ||
       !qti_legacy_ranges_do_not_overlap(layouts, 3) ||
       !qti_legacy_ycbcr_is_empty(ycbcr[1]) ||
       ycbcr[0].ystride != static_cast<size_t>(y.stride_bytes) ||
       ycbcr[0].cstride != static_cast<size_t>(cb.stride_bytes) ||
       ycbcr[0].chroma_step != 1)
      return false;

   return qti_legacy_pointer_matches(handle_info.base, y.offset,
                                     ycbcr[0].y) &&
          qti_legacy_pointer_matches(handle_info.base, cb.offset,
                                     ycbcr[0].cb) &&
          qti_legacy_pointer_matches(handle_info.base, cr.offset,
                                     ycbcr[0].cr);
}

static int
qti_copy_legacy_yv12_layout(
   const native_handle_t *handle,
   const LegacyQtiPlaneLayoutInfo *layouts,
   struct u_gralloc_buffer_basic_info *out)
{
   /* PlaneLayoutInfo is logical Y-Cb-Cr while DRM_FORMAT_YVU420 is physical
    * Y-Cr-Cb.  vk_android swaps these DRM planes back to Vulkan's logical
    * Y-Cb-Cr order when constructing explicit image layouts.
    */
   constexpr size_t drm_plane_order[] = {0, 2, 1};

   out->drm_fourcc = DRM_FORMAT_YVU420;
   out->modifier = DRM_FORMAT_MOD_LINEAR;
   out->num_planes = 3;
   for (size_t i = 0; i < ARRAY_SIZE(drm_plane_order); i++) {
      const LegacyQtiPlaneLayoutInfo &layout = layouts[drm_plane_order[i]];
      out->fds[i] = handle->data[0];
      out->offsets[i] = static_cast<int>(layout.offset);
      out->strides[i] = layout.stride_bytes;
   }

   return 0;
}

static bool
qti_legacy_validate_nv12_ubwc_layout(
   const qti_legacy_handle_info &handle_info,
   const LegacyQtiPlaneLayoutInfo *layouts,
   const struct android_ycbcr (&ycbcr)[2])
{
   const LegacyQtiPlaneLayoutInfo &y = layouts[0];
   const LegacyQtiPlaneLayoutInfo &uv = layouts[1];
   const LegacyQtiPlaneLayoutInfo &y_metadata = layouts[2];
   const LegacyQtiPlaneLayoutInfo &uv_metadata = layouts[3];
   const uint32_t flags = static_cast<uint32_t>(handle_info.flags);
   if (!(flags & QTI_HANDLE_FLAG_UBWC_ALIGNED) ||
       (flags & QTI_HANDLE_FLAG_UBWC_ALIGNED_PI) ||
       !qti_legacy_plane_is_valid(
          y, QTI_LEGACY_PLANE_COMPONENT_Y, 0, 0, 1, handle_info) ||
       !qti_legacy_plane_is_valid(
          uv, QTI_LEGACY_PLANE_COMPONENT_CB |
                 QTI_LEGACY_PLANE_COMPONENT_CR,
          1, 1, 2, handle_info) ||
       !qti_legacy_plane_is_valid(
          y_metadata,
          QTI_LEGACY_PLANE_COMPONENT_META |
             QTI_LEGACY_PLANE_COMPONENT_Y,
          0, 0, 0, handle_info) ||
       !qti_legacy_plane_is_valid(
          uv_metadata,
          QTI_LEGACY_PLANE_COMPONENT_META |
             QTI_LEGACY_PLANE_COMPONENT_CB |
             QTI_LEGACY_PLANE_COMPONENT_CR,
          0, 0, 0, handle_info) ||
       y.stride != handle_info.width || uv.stride != handle_info.width ||
       y_metadata.stride != handle_info.width ||
       uv_metadata.stride != handle_info.width || y_metadata.offset != 0 ||
       y.offset != static_cast<uint64_t>(y_metadata.offset) +
                      y_metadata.size ||
       uv_metadata.offset != static_cast<uint64_t>(y.offset) + y.size ||
       uv.offset != static_cast<uint64_t>(uv_metadata.offset) +
                       uv_metadata.size ||
       !qti_legacy_ranges_do_not_overlap(layouts, 4) ||
       !qti_legacy_ycbcr_is_empty(ycbcr[1]) ||
       ycbcr[0].ystride != static_cast<size_t>(y.stride_bytes) ||
       ycbcr[0].cstride != static_cast<size_t>(uv.stride_bytes) ||
       ycbcr[0].chroma_step != 2 ||
       !qti_legacy_pointer_matches(handle_info.base, y.offset, ycbcr[0].y) ||
       !qti_legacy_pointer_matches(handle_info.base, uv.offset,
                                   ycbcr[0].cb) ||
       uv.offset == UINT32_MAX ||
       !qti_legacy_pointer_matches(handle_info.base, uv.offset + 1,
                                   ycbcr[0].cr))
      return false;

   return true;
}

static int
qti_copy_legacy_nv12_ubwc_layout(
   const native_handle_t *handle,
   const LegacyQtiPlaneLayoutInfo *layouts,
   struct u_gralloc_buffer_basic_info *out)
{
   const LegacyQtiPlaneLayoutInfo &y = layouts[0];
   const LegacyQtiPlaneLayoutInfo &uv = layouts[1];
   const LegacyQtiPlaneLayoutInfo &y_metadata = layouts[2];
   const LegacyQtiPlaneLayoutInfo &uv_metadata = layouts[3];

   out->drm_fourcc = DRM_FORMAT_NV12;
   out->modifier = DRM_FORMAT_MOD_QCOM_COMPRESSED;
   out->num_planes = 2;
   out->fds[0] = out->fds[1] = handle->data[0];
   out->offsets[0] = static_cast<int>(y_metadata.offset);
   out->offsets[1] = static_cast<int>(uv_metadata.offset);
   out->strides[0] = y.stride_bytes;
   out->strides[1] = uv.stride_bytes;
   out->modifier_plane_layouts[0] = {
      .data_offset = y.offset,
      .data_size = y.size,
      .metadata_size = y_metadata.size,
      .metadata_row_pitch =
         static_cast<uint32_t>(y_metadata.stride_bytes),
   };
   out->modifier_plane_layouts[1] = {
      .data_offset = uv.offset,
      .data_size = uv.size,
      .metadata_size = uv_metadata.size,
      .metadata_row_pitch =
         static_cast<uint32_t>(uv_metadata.stride_bytes),
   };
   out->has_explicit_modifier_layout = true;
   return 0;
}

static int
qti_get_legacy_rgb_buffer_basic_info(
   qti_metadata_gralloc *gr, struct u_gralloc_buffer_handle *hnd,
   const qti_legacy_handle_info &handle_info,
   struct u_gralloc_buffer_basic_info *out)
{
   if (!gr->get_legacy_rgb_plane_layouts || !gr->get_legacy_drm_format)
      return -ENOTSUP;

   if (!buffer_description_matches_allocated_format(hnd->hal_format,
                                                    handle_info.format))
      return -EINVAL;

   const LegacyQtiBufferInfo buffer_info = {
      .width = handle_info.unaligned_width,
      .height = handle_info.unaligned_height,
      .format = handle_info.format,
      .layer_count = 1,
      .usage = handle_info.usage,
   };
   int plane_count = 0;
   LegacyQtiPlaneLayoutInfo layout = {};
   gr->get_legacy_rgb_plane_layouts(
      buffer_info, handle_info.format, handle_info.width, handle_info.height,
      handle_info.flags, &plane_count, &layout);

   constexpr uint32_t rgb_components =
      QTI_LEGACY_PLANE_COMPONENT_R | QTI_LEGACY_PLANE_COMPONENT_G |
      QTI_LEGACY_PLANE_COMPONENT_B;
   constexpr uint32_t allowed_components =
      rgb_components | QTI_LEGACY_PLANE_COMPONENT_A;
   const uint64_t stride = layout.stride_bytes > 0
                              ? static_cast<uint64_t>(layout.stride_bytes)
                              : 0;
   const uint64_t scanlines = layout.scanlines > 0
                                 ? static_cast<uint64_t>(layout.scanlines)
                                 : 0;
   const uint64_t data_size =
      stride && scanlines <= UINT64_MAX / stride ? stride * scanlines : 0;

   if (plane_count != 1 || (layout.component & rgb_components) !=
                              rgb_components ||
       (layout.component & ~allowed_components) != 0 ||
       layout.horizontal_subsampling != 0 ||
       layout.vertical_subsampling != 0 || layout.step <= 0 ||
       layout.stride <= 0 || layout.stride_bytes <= 0 ||
       layout.scanlines <= 0 ||
       static_cast<uint64_t>(layout.stride) * layout.step != stride ||
       layout.size != handle_info.declared_size ||
       layout.offset > INT_MAX || layout.offset > layout.size ||
       data_size == 0 ||
       data_size > layout.size - layout.offset ||
       (hnd->pixel_stride > 0 && hnd->pixel_stride != layout.stride))
      return -EINVAL;

   uint32_t drm_fourcc = 0;
   uint64_t reported_modifier = DRM_FORMAT_MOD_LINEAR;
   gr->get_legacy_drm_format(
      static_cast<uint32_t>(handle_info.format),
      static_cast<uint32_t>(handle_info.flags), &drm_fourcc,
      &reported_modifier);
   if (drm_fourcc == 0) {
      /* This legacy helper deliberately omits the standard Android FP16
       * format.  Its memory representation is nevertheless public and
       * unambiguous; keep vendor metadata authoritative for the layout and
       * use Mesa's standard HAL-to-DRM mapping only for the FourCC.
       */
      const int standard_fourcc =
         get_fourcc_from_hal_format(handle_info.format);
      if (standard_fourcc < 0)
         return -ENOTSUP;
      drm_fourcc = static_cast<uint32_t>(standard_fourcc);
   }

   if (reported_modifier != DRM_FORMAT_MOD_LINEAR &&
       reported_modifier != DRM_FORMAT_MOD_QCOM_COMPRESSED)
      return -ENOTSUP;

   const bool compressed = layout.offset != 0;
   const bool ubwc_flag =
      (static_cast<uint32_t>(handle_info.flags) &
       QTI_HANDLE_FLAG_UBWC_ALIGNED) != 0;
   const bool ubwc_pi =
      (static_cast<uint32_t>(handle_info.flags) &
       QTI_HANDLE_FLAG_UBWC_ALIGNED_PI) != 0;
   if (ubwc_pi || compressed != ubwc_flag ||
       (reported_modifier == DRM_FORMAT_MOD_QCOM_COMPRESSED && !compressed))
      return -EINVAL;

   out->drm_fourcc = drm_fourcc;
   out->modifier = compressed ? DRM_FORMAT_MOD_QCOM_COMPRESSED
                              : DRM_FORMAT_MOD_LINEAR;
   out->num_planes = 1;
   out->fds[0] = hnd->handle->data[0];
   out->offsets[0] = compressed ? 0 : static_cast<int>(layout.offset);
   out->strides[0] = layout.stride_bytes;
   if (compressed) {
      out->modifier_plane_layouts[0] = {
         .data_offset = layout.offset,
         .data_size = layout.size - layout.offset,
         .metadata_size = layout.offset,
         .metadata_row_pitch = 0,
      };
      out->has_explicit_modifier_layout = true;
   }

   return 0;
}

static int
qti_get_legacy_plane_layout_buffer_basic_info(
   qti_metadata_gralloc *gr, struct u_gralloc_buffer_handle *hnd,
   struct u_gralloc_buffer_basic_info *out)
{
   if (!hnd || !hnd->handle)
      return -EINVAL;

   if (!qti_legacy_handle_is_compatible(hnd->handle)) {
      if (can_use_fallback_for_known_non_yuv(hnd))
         return qti_fallback_get_buffer_basic_info(gr, hnd, out);
      return -EINVAL;
   }

   qti_legacy_handle_info handle_info = {};
   if (!qti_get_legacy_handle_info(hnd->handle, &handle_info))
      return -EINVAL;

   const int contract_result =
      qti_validate_legacy_buffer_contract(handle_info, hnd);
   if (contract_result)
      return contract_result;

   /* The validated legacy handle is explicitly single-layer and its declared
    * allocation size has been bounded by the dma-buf size above.
    */
   out->alloc_size = handle_info.declared_size;
   out->layer_count = 1;

   const qti_yuv_format_profile *format =
      qti_get_yuv_format_profile(handle_info.format);
   if (!format) {
      int ret = -ENOTSUP;
      if (qti_legacy_rgb_format_is_supported(handle_info.format)) {
         ret = qti_get_legacy_rgb_buffer_basic_info(
            gr, hnd, handle_info, out);
         if (ret == 0)
            return 0;
      }

      /* Never route an unmodelled compressed allocation through the linear
       * fallback.  This is especially important for camera flex formats and
       * legacy RGB formats omitted by GetDRMFormat(): a plausible pitch is
       * not evidence that UBWC metadata was accounted for.
       */
      const uint32_t private_flags =
         static_cast<uint32_t>(handle_info.flags);
      if (private_flags & (QTI_HANDLE_FLAG_UBWC_ALIGNED |
                           QTI_HANDLE_FLAG_UBWC_ALIGNED_PI))
         return ret;

      if (qti_legacy_format_can_use_fallback(hnd, handle_info.format))
         return qti_fallback_get_buffer_basic_info(gr, hnd, out);
      return ret;
   }

   if (!qti_legacy_description_matches(hnd, *format))
      return -EINVAL;

   const uint32_t private_flags =
      static_cast<uint32_t>(handle_info.flags);
   const bool ubwc =
      (private_flags & QTI_HANDLE_FLAG_UBWC_ALIGNED) != 0;
   const bool ubwc_pi =
      (private_flags & QTI_HANDLE_FLAG_UBWC_ALIGNED_PI) != 0;
   const bool format_is_ubwc =
      format->layout == qti_yuv_layout_kind::NV12_UBWC;
   if (ubwc_pi || ubwc != format_is_ubwc)
      return -ENOTSUP;

   const LegacyQtiBufferInfo buffer_info = {
      .width = handle_info.unaligned_width,
      .height = handle_info.unaligned_height,
      .format = handle_info.format,
      .layer_count = 1,
      .usage = handle_info.usage,
   };
   int plane_count = 0;
   LegacyQtiPlaneLayoutInfo layouts[MAX_MAPPER_PLANE_COUNT] = {};
   struct android_ycbcr ycbcr[2] = {};

   /* Passing zero here requests the progressive description.  The
    * handle-aware query independently consumes buffer metadata; a linear
    * override, geometry override, or interlaced second field therefore makes
    * the two results disagree and is rejected rather than guessed.
    */
   if (gr->get_legacy_yuv_plane_layouts(
          buffer_info, handle_info.format, handle_info.width,
          handle_info.height, 0, &plane_count, layouts) != 0 ||
       plane_count <= 0 || plane_count > MAX_MAPPER_PLANE_COUNT ||
       gr->get_legacy_yuv_plane_info(hnd->handle, ycbcr) != 0) {
      mesa_logw_once("Legacy QTI gralloc failed to return complete plane "
                     "metadata");
      return -EINVAL;
   }

   bool valid = false;
   switch (format->layout) {
   case qti_yuv_layout_kind::NV12_LINEAR:
   case qti_yuv_layout_kind::NV21_LINEAR:
      valid = plane_count == 2 &&
              qti_legacy_validate_semiplanar_layout(
                 handle_info, format->layout, layouts, ycbcr);
      if (valid)
         return qti_copy_legacy_semiplanar_layout(
            hnd->handle, *format, layouts, out);
      break;
   case qti_yuv_layout_kind::YV12_LINEAR:
      valid = plane_count == 3 &&
              qti_legacy_validate_yv12_layout(handle_info, layouts, ycbcr);
      if (valid)
         return qti_copy_legacy_yv12_layout(hnd->handle, layouts, out);
      break;
   case qti_yuv_layout_kind::P010_LINEAR:
      valid = plane_count == 2 &&
              qti_legacy_validate_p010_layout(handle_info, layouts, ycbcr);
      if (valid)
         return qti_copy_legacy_semiplanar_layout(
            hnd->handle, *format, layouts, out);
      break;
   case qti_yuv_layout_kind::NV12_UBWC:
      valid = plane_count == 4 &&
              qti_legacy_validate_nv12_ubwc_layout(handle_info, layouts,
                                                   ycbcr);
      if (valid)
         return qti_copy_legacy_nv12_ubwc_layout(hnd->handle, layouts, out);
      break;
   }

   mesa_logw_once("Legacy QTI gralloc returned an inconsistent layout for "
                  "private format 0x%x", handle_info.format);
   return -EINVAL;
}

static bool
qti_validate_legacy_nv12_ubwc_layout(
   const native_handle_t *handle, uint32_t aligned_width,
   uint64_t allocation_size, const LegacyQtiPlaneLayoutInfo (&layouts)[4],
   const struct android_ycbcr (&ycbcr)[2])
{
   constexpr uint32_t expected_components[4] = {
      QTI_LEGACY_PLANE_COMPONENT_Y,
      QTI_LEGACY_PLANE_COMPONENT_CB | QTI_LEGACY_PLANE_COMPONENT_CR,
      QTI_LEGACY_PLANE_COMPONENT_META | QTI_LEGACY_PLANE_COMPONENT_Y,
      QTI_LEGACY_PLANE_COMPONENT_META | QTI_LEGACY_PLANE_COMPONENT_CB |
         QTI_LEGACY_PLANE_COMPONENT_CR,
   };

   const uint32_t flags = static_cast<uint32_t>(
      handle->data[QTI_LEGACY_HANDLE_FLAGS_INDEX]);
   if (!(flags & QTI_HANDLE_FLAG_UBWC_ALIGNED) ||
       (flags & (QTI_HANDLE_FLAG_UBWC_ALIGNED_PI |
                 QTI_HANDLE_FLAG_SECURE_BUFFER)))
      return false;

   for (size_t i = 0; i < ARRAY_SIZE(layouts); i++) {
      const LegacyQtiPlaneLayoutInfo &layout = layouts[i];
      if (layout.component != expected_components[i] ||
          layout.horizontal_subsampling != 0 ||
          layout.vertical_subsampling != 0 || layout.step != 0 ||
          layout.stride != static_cast<int32_t>(aligned_width) ||
          layout.stride_bytes <= 0 || layout.scanlines <= 0 ||
          !qti_legacy_plane_range_fits(layout, allocation_size))
         return false;
   }

   const LegacyQtiPlaneLayoutInfo &y = layouts[0];
   const LegacyQtiPlaneLayoutInfo &uv = layouts[1];
   const LegacyQtiPlaneLayoutInfo &y_metadata = layouts[2];
   const LegacyQtiPlaneLayoutInfo &uv_metadata = layouts[3];

   /* The legacy helper reports logical planes as Y, UV, Y-meta, UV-meta,
    * while the dma-buf is physically Y-meta, Y, UV-meta, UV.  Require exact
    * adjacency so no inferred metadata range can overlap or belong to a
    * different data plane.
    */
   const uint64_t y_metadata_end =
      static_cast<uint64_t>(y_metadata.offset) + y_metadata.size;
   const uint64_t y_end = static_cast<uint64_t>(y.offset) + y.size;
   const uint64_t uv_metadata_end =
      static_cast<uint64_t>(uv_metadata.offset) + uv_metadata.size;
   const uint64_t uv_end = static_cast<uint64_t>(uv.offset) + uv.size;

   if (y_metadata.offset != 0 || y.offset != y_metadata_end ||
       uv_metadata.offset != y_end || uv.offset != uv_metadata_end ||
       uv_end > allocation_size)
      return false;

   const uint32_t declared_size = static_cast<uint32_t>(
      handle->data[QTI_LEGACY_HANDLE_SIZE_INDEX]);
   if (declared_size == 0 || declared_size > allocation_size ||
       uv_end > declared_size)
      return false;

   /* GetYUVPlaneInfo() derives the data addresses from the actual handle and
    * its metadata.  Cross-check it against the width/height-only helper so a
    * wrong enum value, private-handle revision, interlaced allocation, or
    * metadata override cannot silently produce a plausible layout.
    */
   if (!qti_legacy_ycbcr_is_empty(ycbcr[1]) ||
       ycbcr[0].ystride != static_cast<size_t>(y.stride_bytes) ||
       ycbcr[0].cstride != static_cast<size_t>(uv.stride_bytes) ||
       ycbcr[0].chroma_step != 2)
      return false;

   uintptr_t base = 0;
   memcpy(&base, &handle->data[QTI_LEGACY_HANDLE_BASE_INDEX], sizeof(base));
   if (!qti_legacy_pointer_matches(base, y.offset, ycbcr[0].y) ||
       !qti_legacy_pointer_matches(base, uv.offset, ycbcr[0].cb) ||
       uv.offset == UINT32_MAX ||
       !qti_legacy_pointer_matches(base, uv.offset + 1, ycbcr[0].cr))
      return false;

   return true;
}

static int
qti_get_legacy_nv12_ubwc_buffer_basic_info(
   qti_metadata_gralloc *gr, struct u_gralloc_buffer_handle *hnd,
   struct u_gralloc_buffer_basic_info *out)
{
   if (!hnd || !hnd->handle)
      return -EINVAL;

   if (!qti_legacy_handle_is_compatible(hnd->handle)) {
      /* Loading the legacy helper does not prove that every handle in the
       * process uses its private ABI.  Unknown/ambiguous handles must not
       * re-enter fallback's IMPLEMENTATION_DEFINED RGB heuristic.
       */
      if (can_use_fallback_for_known_non_yuv(hnd))
         return qti_fallback_get_buffer_basic_info(gr, hnd, out);
      return -EINVAL;
   }

   const native_handle_t *handle = hnd->handle;
   qti_legacy_handle_info handle_info = {};
   if (!qti_get_legacy_handle_info(handle, &handle_info))
      return -EINVAL;

   const int contract_result =
      qti_validate_legacy_buffer_contract(handle_info, hnd);
   if (contract_result)
      return contract_result;

   const int32_t private_format =
      handle->data[QTI_LEGACY_HANDLE_FORMAT_INDEX];

   /* The legacy bridge is deliberately limited to the private video format
    * for which it can recover both primary and metadata planes.  Preserve
    * the pre-existing fallback backend for known standard allocations,
    * including ordinary RGB UI buffers, so enabling this bridge cannot
    * change boot or compositor behavior.  Other private formats remain
    * fail-closed until an authoritative layout ABI is implemented for them.
    */
   if (private_format !=
       QTI_HAL_PIXEL_FORMAT_YCBCR_420_SP_VENUS_UBWC) {
      if (qti_legacy_format_can_use_fallback(hnd, private_format))
         return qti_fallback_get_buffer_basic_info(gr, hnd, out);
      return -ENOTSUP;
   }

   if (!qti_legacy_description_can_be_nv12_ubwc(hnd->hal_format))
      return -EINVAL;

   const int32_t width = handle->data[QTI_LEGACY_HANDLE_WIDTH_INDEX];
   const int32_t height = handle->data[QTI_LEGACY_HANDLE_HEIGHT_INDEX];
   const int32_t unaligned_width =
      handle->data[QTI_LEGACY_HANDLE_UNALIGNED_WIDTH_INDEX];
   const int32_t unaligned_height =
      handle->data[QTI_LEGACY_HANDLE_UNALIGNED_HEIGHT_INDEX];
   const int32_t layer_count =
      handle->data[QTI_LEGACY_HANDLE_LAYER_COUNT_INDEX];

   if (width <= 0 || height <= 0 || unaligned_width <= 0 ||
       unaligned_height <= 0 || width < unaligned_width ||
       height < unaligned_height || layer_count != 1 ||
       handle->data[QTI_LEGACY_HANDLE_OFFSET_INDEX] != 0)
      return -EINVAL;

   uint64_t allocation_size = 0;
   if (!get_dma_buf_allocation_size(handle, &allocation_size))
      return -EINVAL;

   LegacyQtiPlaneLayoutInfo layouts[4] = {};
   struct android_ycbcr ycbcr[2] = {};
   gr->get_legacy_yuv_ubwc_plane_info(
      static_cast<uint32_t>(width), static_cast<uint32_t>(height),
      QTI_LEGACY_COLOR_FMT_NV12_UBWC, layouts);

   if (gr->get_legacy_yuv_plane_info(handle, ycbcr) != 0 ||
       !qti_validate_legacy_nv12_ubwc_layout(
          handle, static_cast<uint32_t>(width), allocation_size, layouts,
          ycbcr)) {
      mesa_logw_once("Legacy QTI gralloc returned inconsistent NV12 UBWC "
                     "plane metadata");
      return -EINVAL;
   }

   const LegacyQtiPlaneLayoutInfo &y = layouts[0];
   const LegacyQtiPlaneLayoutInfo &uv = layouts[1];
   const LegacyQtiPlaneLayoutInfo &y_metadata = layouts[2];
   const LegacyQtiPlaneLayoutInfo &uv_metadata = layouts[3];

   out->alloc_size = handle_info.declared_size;
   out->layer_count = 1;
   out->drm_fourcc = DRM_FORMAT_NV12;
   out->modifier = DRM_FORMAT_MOD_QCOM_COMPRESSED;
   out->num_planes = 2;
   out->fds[0] = out->fds[1] = handle->data[0];
   out->offsets[0] = static_cast<int>(y_metadata.offset);
   out->offsets[1] = static_cast<int>(uv_metadata.offset);
   out->strides[0] = y.stride_bytes;
   out->strides[1] = uv.stride_bytes;
   out->modifier_plane_layouts[0] = {
      .data_offset = y.offset,
      .data_size = y.size,
      .metadata_size = y_metadata.size,
      .metadata_row_pitch =
         static_cast<uint32_t>(y_metadata.stride_bytes),
   };
   out->modifier_plane_layouts[1] = {
      .data_offset = uv.offset,
      .data_size = uv.size,
      .metadata_size = uv_metadata.size,
      .metadata_row_pitch =
         static_cast<uint32_t>(uv_metadata.stride_bytes),
   };
   out->has_explicit_modifier_layout = true;
   return 0;
}

static int
qti_get_legacy_ycbcr_ubwc_buffer_basic_info(
   qti_metadata_gralloc *gr, struct u_gralloc_buffer_handle *hnd,
   struct u_gralloc_buffer_basic_info *out)
{
   if (!hnd || !hnd->handle)
      return -EINVAL;

   if (!qti_legacy_handle_is_compatible(hnd->handle)) {
      /* Loading the legacy helper does not prove that every handle in the
       * process uses its private ABI.  Unknown/ambiguous handles must not
       * re-enter fallback's IMPLEMENTATION_DEFINED RGB heuristic.
       */
      if (can_use_fallback_for_known_non_yuv(hnd))
         return qti_fallback_get_buffer_basic_info(gr, hnd, out);
      return -EINVAL;
   }

   const native_handle_t *handle = hnd->handle;
   qti_legacy_handle_info handle_info = {};
   if (!qti_get_legacy_handle_info(handle, &handle_info))
      return -EINVAL;

   const int contract_result =
      qti_validate_legacy_buffer_contract(handle_info, hnd);
   if (contract_result)
      return contract_result;

   /* This is the SM8150/SDM845 private gralloc ABI, selected purely by the
    * exact mangled symbols of the handle-aware helper and the android_ycbcr
    * variant of GetYuvUbwcSPPlaneInfo().  It is a software-ABI gate, not a
    * GPU or device-model gate: any stack exporting the same symbols uses this
    * backend.
    */
   const qti_yuv_format_profile *format =
      qti_get_yuv_format_profile(handle_info.format);
   if (!format || !qti_legacy_description_matches(hnd, *format))
      return -EINVAL;

   struct android_ycbcr ycbcr[2] = {};
   if (gr->get_legacy_yuv_plane_info(handle, ycbcr) != 0 ||
       !qti_legacy_ycbcr_is_empty(ycbcr[1]))
      return -EINVAL;

   const uint32_t flags = static_cast<uint32_t>(handle_info.flags);
   if (flags & QTI_HANDLE_FLAG_UBWC_ALIGNED_PI)
      return -ENOTSUP;

   /* The SM8150/SDM845 android_ycbcr helper reports the data-plane geometry
    * but not the leading UBWC metadata sizes.  Turnip requires those
    * authoritative metadata/data ranges for a QCOM_COMPRESSED import (see the
    * gralloc/Turnip modifier layout cross-check in tu_image.c), and deriving
    * them would mean re-implementing the vendor VENUS layout formula.  Refuse
    * a compressed allocation rather than guess its metadata or silently treat
    * it as linear.
    */
   if (flags & QTI_HANDLE_FLAG_UBWC_ALIGNED) {
      mesa_logw_once("Unsupported QTI NV12 UBWC allocation without "
                     "authoritative metadata on the legacy ycbcr ABI");
      return -ENOTSUP;
   }

   if (flags & QTI_HANDLE_FLAG_SECURE_BUFFER)
      return -ENOTSUP;

   const uint64_t y_off = reinterpret_cast<uintptr_t>(ycbcr[0].y);
   const uint64_t cb_off = reinterpret_cast<uintptr_t>(ycbcr[0].cb);
   const uint64_t cr_off = reinterpret_cast<uintptr_t>(ycbcr[0].cr);

   const uint64_t base = static_cast<uint64_t>(handle_info.base);
   const uint64_t alloc = handle_info.allocation_size;
   const uint64_t declared = handle_info.declared_size;
   const auto in_range = [base, alloc, declared](uint64_t off, uint64_t bytes) {
      if (off < base || bytes == 0 || bytes > alloc)
         return false;
      const uint64_t rel = off - base;
      return rel <= declared && bytes <= alloc - rel &&
             bytes <= declared - rel;
   };

   const uint64_t width = static_cast<uint64_t>(handle_info.unaligned_width);
   const uint64_t height = static_cast<uint64_t>(handle_info.unaligned_height);
   const uint64_t aligned_width =
      static_cast<uint64_t>(handle_info.width);
   const uint64_t aligned_height =
      static_cast<uint64_t>(handle_info.height);
   if (width == 0 || height == 0 || aligned_width < width ||
       aligned_height < height)
      return -EINVAL;

   const uint64_t ystride = static_cast<uint64_t>(ycbcr[0].ystride);
   const uint64_t cstride = static_cast<uint64_t>(ycbcr[0].cstride);
   const uint64_t chroma_step = static_cast<uint64_t>(ycbcr[0].chroma_step);
   if (ystride == 0 || cstride == 0 || ystride < width)
      return -EINVAL;

   /* GetYuvSPPlaneInfo() yields stride == aligned width for both planes and a
    * chroma_step of 2 for NV12/NV21.  YV12 uses chroma_step 1 and a separate
    * Cb/Cr.  Any other chroma_step (3 = TP10, 4 = P010, ...) is not handled by
    * this backend and stays fail-closed.
    */
   const bool yv12 = chroma_step == 1;
   const bool semiplanar = chroma_step == 2;
   if (!semiplanar && !yv12)
      return -ENOTSUP;

   /* Reject NV21/other private formats whose profile does not match the
    * chroma geometry reported by the helper before trusting any pointer.
    */
   const bool nv21 = format->layout == qti_yuv_layout_kind::NV21_LINEAR;
   if (semiplanar && format->layout != qti_yuv_layout_kind::NV12_LINEAR &&
       format->layout != qti_yuv_layout_kind::NV21_LINEAR)
      return -EINVAL;
   if (yv12 && format->layout != qti_yuv_layout_kind::YV12_LINEAR)
      return -EINVAL;

   if (semiplanar) {
      /* The handle-aware helper derives the Y and interleaved UV offsets from
       * the handle's aligned dimensions, exactly like GetYuvSPPlaneInfo().
       * The Y plane starts at base; UV immediately follows Y.
       */
      if (y_off != base)
         return -EINVAL;

      /* ystride is the aligned row length; the helper used the aligned height
       * when computing the UV base, so the Y plane extends over aligned rows.
       */
      const uint64_t y_size = ystride * aligned_height;
      if (y_size > UINT64_MAX - base)
         return -EINVAL;

      const uint64_t expected_uv_off = y_off + y_size;
      const uint64_t uv_start = cr_off < cb_off ? cr_off : cb_off;
      if (uv_start != expected_uv_off)
         return -EINVAL;

      /* Cb and Cr are interleaved, exactly one byte apart. */
      const uint64_t delta =
         cb_off > cr_off ? cb_off - cr_off : cr_off - cb_off;
      if (delta != 1)
         return -EINVAL;

      const uint64_t uv_rel = uv_start - base;
      const uint64_t uv_size =
         cstride * ((aligned_height + 1) / 2);
      if (uv_rel > INT_MAX || !in_range(uv_start, uv_size))
         return -EINVAL;

      out->drm_fourcc = nv21 ? DRM_FORMAT_NV21 : DRM_FORMAT_NV12;
      out->modifier = DRM_FORMAT_MOD_LINEAR;
      out->num_planes = 2;
      out->fds[0] = out->fds[1] = handle->data[0];
      out->offsets[0] = 0;
      out->offsets[1] = static_cast<int>(uv_rel);
      out->strides[0] = static_cast<int>(ystride);
      out->strides[1] = static_cast<int>(cstride);
      return 0;
   }

   /* YV12: Y at base, Cr immediately after Y, Cb after Cr.  The chroma stride
    * is 16-byte aligned by the Android YV12 contract.  The helper derives all
    * plane bases from the handle's aligned dimensions.
    */
   if (y_off != base || (ystride & 15) != 0)
      return -EINVAL;

   const uint64_t y_size = ystride * aligned_height;
   const uint64_t c_size = cstride * ((aligned_height + 1) / 2);
   if (y_size > UINT64_MAX - base || c_size > UINT64_MAX - (base + y_size) ||
       cr_off != y_off + y_size || cb_off != cr_off + c_size ||
       cstride < (width + 1) / 2 || !in_range(y_off, y_size) ||
       !in_range(cr_off, c_size) || !in_range(cb_off, c_size))
      return -EINVAL;

   const uint64_t cr_rel = cr_off - base;
   const uint64_t cb_rel = cb_off - base;
   if (cr_rel > INT_MAX || cb_rel > INT_MAX)
      return -EINVAL;

   out->drm_fourcc = DRM_FORMAT_YVU420;
   out->modifier = DRM_FORMAT_MOD_LINEAR;
   out->num_planes = 3;
   out->fds[0] = out->fds[1] = out->fds[2] = handle->data[0];
   /* DRM_FORMAT_YVU420 is physical Y-Cr-Cb; vk_android swaps back to Vulkan's
    * logical Y-Cb-Cr order.
    */
   out->offsets[0] = 0;
   out->offsets[1] = static_cast<int>(cr_rel);
   out->offsets[2] = static_cast<int>(cb_rel);
   out->strides[0] = static_cast<int>(ystride);
   out->strides[1] = static_cast<int>(cstride);
   out->strides[2] = static_cast<int>(cstride);
   return 0;
}

static int
qti_get_modern_buffer_basic_info(
   qti_metadata_gralloc *gr, struct u_gralloc_buffer_handle *hnd,
   struct u_gralloc_buffer_basic_info *out)
{
   if (!hnd || !hnd->handle || !buffer_description_supported(hnd))
      return -EINVAL;

   if (!qti_modern_handle_is_compatible(hnd->handle)) {
      /* The explicit-YUV capability must never make an unknown private handle
       * enter a heuristic path.  Preserve the old fallback only for an
       * unambiguously standard, non-YUV HAL format.
       */
      if (can_use_fallback_for_known_non_yuv(hnd))
         return qti_fallback_get_buffer_basic_info(gr, hnd, out);
      return -EINVAL;
   }

   /* Without standard metadata there is no authoritative QTI secure flag to
    * cross-check.  Turnip has no protected-memory import path, so reject an
    * explicitly protected public description instead of importing it as an
    * ordinary dma-buf.  Non-protected linear imports can still use the public
    * plane-layout API on QTI revisions without GetMetaDataValue().
    */
   if (!gr->get_metadata_value && hnd->has_usage &&
       (hnd->usage & GRALLOC_USAGE_PROTECTED))
      return -ENOTSUP;

   uint64_t dma_buf_size = 0;
   if (!get_dma_buf_allocation_size(hnd->handle, &dma_buf_size))
      return -EINVAL;

   uint64_t allocation_size = dma_buf_size;
   int32_t actual_format = hnd->hal_format;
   uint32_t authoritative_fourcc = 0;
   uint64_t authoritative_modifier = DRM_FORMAT_MOD_INVALID;
   uint32_t authoritative_private_flags = 0;
   bool has_authoritative_private_flags = false;
   qti_standard_buffer_metadata metadata = {};
   struct u_gralloc_buffer_handle validated_hnd = *hnd;

   if (gr->get_metadata_value) {
      if (!qti_get_standard_buffer_metadata(gr, hnd, dma_buf_size, &metadata,
                                            &validated_hnd)) {
         mesa_logw_once("QTI gralloc returned inconsistent standard metadata");
         return -EINVAL;
      }

      allocation_size = metadata.allocation_size;
      actual_format = metadata.requested_format;
      authoritative_fourcc = metadata.drm_fourcc;
      authoritative_modifier = metadata.modifier;
      authoritative_private_flags = metadata.private_flags;
      has_authoritative_private_flags = true;

      if (!buffer_description_matches_allocated_format(hnd->hal_format,
                                                       actual_format)) {
         mesa_logw_once("QTI gralloc allocated format disagrees with the "
                        "Android buffer description");
         return -EINVAL;
      }

      /* Turnip currently advertises protectedMemory=false and has no KGSL
       * secure context/mapping path.  Never import a secure allocation as an
       * ordinary dma-buf merely because its public plane layout is readable.
       */
      if (metadata.protected_content) {
         mesa_logw_once("Unsupported protected QTI external buffer");
         return -ENOTSUP;
      }

      /* UBWC-PI has distinct lossy/prediction semantics which are not encoded
       * by DRM_FORMAT_MOD_QCOM_COMPRESSED or Turnip's current FDL layout.  QTI
       * HWC consumes this same public immutable flag explicitly.  Treating PI
       * as ordinary UBWC would silently reinterpret the allocation.
       */
      if (authoritative_private_flags & QTI_HANDLE_FLAG_UBWC_ALIGNED_PI) {
         mesa_logw_once("Unsupported QTI UBWC-PI external buffer");
         return -ENOTSUP;
      }
   }

   std::vector<PlaneLayout> layouts;
   if (gr->get_plane_layouts(const_cast<native_handle_t *>(hnd->handle),
                             &layouts) != 0 ||
       layouts.empty() || layouts.size() > MAX_MAPPER_PLANE_COUNT) {
      if (!gr->get_metadata_value && can_use_fallback_for_known_non_yuv(hnd))
         return qti_fallback_get_buffer_basic_info(gr, hnd, out);

      mesa_logw_once("QTI gralloc failed to return complete plane metadata");
      return -EINVAL;
   }

   for (const PlaneLayout &layout : layouts) {
      if (!valid_layout(layout, allocation_size, &validated_hnd)) {
         if (!gr->get_metadata_value && can_use_fallback_for_known_non_yuv(hnd))
            return qti_fallback_get_buffer_basic_info(gr, hnd, out);

         mesa_logw_once("QTI gralloc returned an invalid plane layout");
         return -EINVAL;
      }
   }

   /* Standard metadata fills validated_hnd.layer_count.  Without that API,
    * retain the public description's zero-for-unknown convention.  Known
    * multi-layer descriptions were rejected before querying plane layouts.
    */
   out->alloc_size = allocation_size;
   out->layer_count = validated_hnd.layer_count;

   std::vector<PlaneLayout> normalized;
   const qti_yuv_format_profile *format_profile =
      qti_get_yuv_format_profile(actual_format);
   const uint32_t effective_fourcc =
      authoritative_fourcc
         ? authoritative_fourcc
         : (format_profile ? format_profile->drm_fourcc : 0);

   if (format_profile &&
       format_profile->layout == qti_yuv_layout_kind::P010_LINEAR &&
       normalize_qti_p010_layouts(layouts, &validated_hnd, &normalized)) {
      /* QTI's public GetDRMFormat() predates DRM_FORMAT_P010 and describes
       * the same linear allocation as NV12 plus its private DX bit.  That bit
       * now aliases QCOM_TILED2, so normalize it only after the HAL format
       * and every AOSP P010 component field have independently matched.
       */
      const bool standard_metadata =
         authoritative_fourcc == DRM_FORMAT_P010 &&
         authoritative_modifier == DRM_FORMAT_MOD_LINEAR;
      const bool legacy_qti_metadata =
         authoritative_fourcc == DRM_FORMAT_NV12 &&
         authoritative_modifier == QTI_DRM_FORMAT_MODIFIER_DX;
      if ((has_authoritative_private_flags &&
           (!standard_metadata && !legacy_qti_metadata)) ||
          (has_authoritative_private_flags &&
           (authoritative_private_flags & QTI_HANDLE_FLAG_UBWC_ALIGNED))) {
         mesa_logw_once("QTI gralloc P010 FourCC/modifier/flags disagree "
                        "with its plane layout");
         return -EINVAL;
      }

      out->drm_fourcc = DRM_FORMAT_P010;
      out->modifier = DRM_FORMAT_MOD_LINEAR;
      const int ret = copy_linear_layout(hnd->handle, normalized,
                                         allocation_size, out);
      if (ret)
         mesa_logw_once("Unsupported or inconsistent QTI P010 plane layout");
      return ret;
   }

   if (has_authoritative_private_flags &&
       authoritative_modifier == DRM_FORMAT_MOD_LINEAR &&
       (!effective_fourcc || effective_fourcc == DRM_FORMAT_YVU420)) {
      if ((authoritative_private_flags &
           QTI_HANDLE_FLAG_UBWC_ALIGNED)) {
         mesa_logw_once("QTI gralloc UBWC flag disagrees with its YV12 "
                        "plane layout");
         return -EINVAL;
      }

      if (normalize_qti_yv12_layouts(
              layouts, actual_format, metadata.width,
              metadata.height, metadata.aligned_width,
              metadata.aligned_height, &normalized)) {
         out->drm_fourcc = DRM_FORMAT_YVU420;
         out->modifier = DRM_FORMAT_MOD_LINEAR;

         const int ret = copy_linear_layout(
            hnd->handle, normalized, allocation_size, out);
         if (ret)
            mesa_logw_once(
               "Unsupported or inconsistent QTI YV12 plane layout");
         return ret;
      }
   }

   if ((!effective_fourcc || effective_fourcc == DRM_FORMAT_YVU420) &&
       effective_fourcc != DRM_FORMAT_NV12 &&
       effective_fourcc != DRM_FORMAT_NV21 &&
       !has_authoritative_private_flags &&
       normalize_qti_yv12_layouts(
          layouts, actual_format, validated_hnd.width,
          validated_hnd.height, validated_hnd.width,
          validated_hnd.height, &normalized)) {
      out->drm_fourcc = DRM_FORMAT_YVU420;
      out->modifier = DRM_FORMAT_MOD_LINEAR;

      const int ret = copy_linear_layout(
         hnd->handle, normalized, allocation_size, out);
      if (ret)
         mesa_logw_once(
            "Unsupported or inconsistent QTI YV12 plane layout");
      return ret;
   }

   const bool authoritative_nv12 = effective_fourcc == DRM_FORMAT_NV12;
   const bool authoritative_nv21 = effective_fourcc == DRM_FORMAT_NV21;
   const bool try_420sp = authoritative_nv12 || authoritative_nv21;
   const bool cr_first = authoritative_nv21;
   const bool allow_omitted_components =
      !cr_first &&
      (actual_format == QTI_HAL_PIXEL_FORMAT_YCBCR_420_SP_VENUS_UBWC ||
       actual_format == QTI_HAL_PIXEL_FORMAT_NV12_HEIF);
   bool compressed = false;
   if (try_420sp &&
       normalize_qti_420sp_layouts(layouts, cr_first,
                                   allow_omitted_components, &normalized,
                                   &compressed)) {
      const uint64_t layout_modifier =
         compressed ? DRM_FORMAT_MOD_QCOM_COMPRESSED : DRM_FORMAT_MOD_LINEAR;
      const bool profile_matches_layout =
         !format_profile ||
         (format_profile->layout == qti_yuv_layout_kind::NV12_LINEAR &&
          !cr_first && !compressed) ||
         (format_profile->layout == qti_yuv_layout_kind::NV21_LINEAR &&
          cr_first && !compressed) ||
         (format_profile->layout == qti_yuv_layout_kind::NV12_UBWC &&
          !cr_first && compressed);

      if ((has_authoritative_private_flags &&
           authoritative_modifier != layout_modifier) ||
          (compressed && !has_authoritative_private_flags) ||
          !profile_matches_layout ||
          (has_authoritative_private_flags &&
           compressed !=
              !!(authoritative_private_flags &
                 QTI_HANDLE_FLAG_UBWC_ALIGNED)) ||
          (cr_first && compressed)) {
         mesa_logw_once("QTI gralloc FourCC/modifier/flags disagree with "
                        "its 4:2:0 plane layout");
         return -EINVAL;
      }

      out->drm_fourcc = cr_first ? DRM_FORMAT_NV21 : DRM_FORMAT_NV12;
      out->modifier = layout_modifier;

      const int ret = compressed
                         ? copy_qcom_ubwc_layout(hnd->handle, normalized,
                                                allocation_size, out)
                         : copy_linear_layout(hnd->handle, normalized,
                                              allocation_size, out);
      if (ret) {
         mesa_logw_once("Unsupported or inconsistent QTI 4:2:0 semiplanar "
                        "layout");
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
         hnd, layouts, allocation_size, actual_format, authoritative_fourcc,
         authoritative_modifier, has_authoritative_private_flags,
         authoritative_private_flags, out);
      if (ret)
         mesa_logw_once("Unsupported or inconsistent QTI RGB plane layout");
      return ret;
   }

   if (!gr->get_metadata_value && can_use_fallback_for_known_non_yuv(hnd))
      return qti_fallback_get_buffer_basic_info(gr, hnd, out);

   mesa_logw_once("Unsupported QTI buffer layout (HAL format 0x%x, "
                  "FourCC 0x%x, modifier 0x%" PRIx64 ")",
                  hnd->hal_format, authoritative_fourcc,
                  authoritative_modifier);
   return -ENOTSUP;
}

static constexpr qti_metadata_backend qti_metadata_backends[] = {
   {
      .name = "modern AIDL PlaneLayout",
      .color_space_symbol = QTI_GET_MODERN_COLOR_SPACE_SYMBOL,
      .supports_swapchain_ubwc = true,
      .load = load_qti_modern_metadata_api,
      .handle_is_compatible = qti_modern_handle_is_compatible,
      .get_buffer_basic_info = qti_get_modern_buffer_basic_info,
   },
   {
      .name = "legacy BufferInfo/PlaneLayoutInfo",
      .color_space_symbol = QTI_GET_LEGACY_COLOR_SPACE_SYMBOL,
      .supports_swapchain_ubwc = false,
      .load = load_qti_legacy_plane_layout_api,
      .handle_is_compatible = qti_legacy_handle_is_compatible,
      .get_buffer_basic_info =
         qti_get_legacy_plane_layout_buffer_basic_info,
   },
   {
      .name = "legacy NV12 UBWC",
      .color_space_symbol = QTI_GET_LEGACY_COLOR_SPACE_SYMBOL,
      .supports_swapchain_ubwc = false,
      .load = load_qti_legacy_nv12_ubwc_api,
      .handle_is_compatible = qti_legacy_handle_is_compatible,
      .get_buffer_basic_info =
         qti_get_legacy_nv12_ubwc_buffer_basic_info,
   },
   {
      /* SM8150/SDM845 private gralloc ABI.  Selected by the exact mangled
       * symbols of the android_ycbcr variant of GetYuvUbwcSPPlaneInfo() and
       * the handle-aware GetYUVPlaneInfo(); both are required.  It restores
       * authoritative NV12/NV21/YV12 linear imports for gralloc stacks that
       * expose neither the modern AIDL PlaneLayout API nor the newer
       * BufferInfo/PlaneLayoutInfo helpers.  This is a software-ABI gate, not
       * a GPU-model gate.
       */
      .name = "legacy ycbcr handle",
      .color_space_symbol = QTI_GET_LEGACY_COLOR_SPACE_SYMBOL,
      .supports_swapchain_ubwc = false,
      .load = load_qti_legacy_ycbcr_ubwc_api,
      .handle_is_compatible = qti_legacy_handle_is_compatible,
      .get_buffer_basic_info =
         qti_get_legacy_ycbcr_ubwc_buffer_basic_info,
   },
};

static bool
qti_select_metadata_backend(void *library, qti_metadata_gralloc *gr)
{
   for (const qti_metadata_backend &backend : qti_metadata_backends) {
      if (backend.load(library, gr)) {
         gr->backend = &backend;
         return true;
      }
   }

   return false;
}

static int
qti_get_buffer_basic_info(struct u_gralloc *gralloc,
                          struct u_gralloc_buffer_handle *hnd,
                          struct u_gralloc_buffer_basic_info *out)
{
   qti_metadata_gralloc *gr =
      reinterpret_cast<qti_metadata_gralloc *>(gralloc);
   if (!gr->backend)
      return -ENOTSUP;

   return gr->backend->get_buffer_basic_info(gr, hnd, out);
}

static int
qti_get_buffer_color_info(struct u_gralloc *gralloc,
                          struct u_gralloc_buffer_handle *hnd,
                          struct u_gralloc_buffer_color_info *out)
{
   qti_metadata_gralloc *gr =
      reinterpret_cast<qti_metadata_gralloc *>(gralloc);

   if (!hnd || !qti_handle_is_compatible(gr, hnd->handle) ||
       !gr->get_color_space)
      return -EINVAL;

   /* Values are the stable QTI HAL_CSC_* ABI used by
    * GetColorSpaceFromMetadata().  Newer stacks add 709 full-range as 5.
    */
   constexpr int QTI_CSC_ITU_R_601 = 0;
   constexpr int QTI_CSC_ITU_R_601_FR = 1;
   constexpr int QTI_CSC_ITU_R_709 = 2;
   constexpr int QTI_CSC_ITU_R_2020 = 3;
   constexpr int QTI_CSC_ITU_R_2020_FR = 4;
   constexpr int QTI_CSC_ITU_R_709_FR = 5;

   int color_space = QTI_CSC_ITU_R_601;
   gr->get_color_space(const_cast<native_handle_t *>(hnd->handle),
                       &color_space);

   *out = {
      .yuv_color_space = __DRI_YUV_COLOR_SPACE_ITU_REC601,
      .sample_range = __DRI_YUV_NARROW_RANGE,
      .horizontal_siting = __DRI_YUV_CHROMA_SITING_0_5,
      .vertical_siting = __DRI_YUV_CHROMA_SITING_0_5,
   };

   switch (color_space) {
   case QTI_CSC_ITU_R_601_FR:
      out->sample_range = __DRI_YUV_FULL_RANGE;
      break;
   case QTI_CSC_ITU_R_709:
      out->yuv_color_space = __DRI_YUV_COLOR_SPACE_ITU_REC709;
      break;
   case QTI_CSC_ITU_R_709_FR:
      out->yuv_color_space = __DRI_YUV_COLOR_SPACE_ITU_REC709;
      out->sample_range = __DRI_YUV_FULL_RANGE;
      break;
   case QTI_CSC_ITU_R_2020:
      out->yuv_color_space = __DRI_YUV_COLOR_SPACE_ITU_REC2020;
      break;
   case QTI_CSC_ITU_R_2020_FR:
      out->yuv_color_space = __DRI_YUV_COLOR_SPACE_ITU_REC2020;
      out->sample_range = __DRI_YUV_FULL_RANGE;
      break;
   case QTI_CSC_ITU_R_601:
   default:
      break;
   }

   return 0;
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

   if (!hnd || !hnd->handle || hnd->handle->numFds <= 0 ||
       !buffer_description_supported(hnd))
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
      if (!valid_layout(layout, allocation_size, hnd)) {
         mesa_logw_once("libui returned an invalid gralloc plane layout");
         return -EINVAL;
      }
   }

   /* This libui bridge predates the public getLayerCount() ABI used by the
    * native Mapper5 backend.  Preserve a known single-layer public
    * description, but do not infer an unknown count from plane metadata.
    */
   out->alloc_size = allocation_size;
   out->layer_count = hnd->layer_count;

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
   void *symbol_scope = RTLD_DEFAULT;
   if (!qti_select_metadata_backend(symbol_scope, gr)) {
      gr->grallocutils =
         dlopen("libgrallocutils.so", RTLD_NOW | RTLD_LOCAL);
      if (!gr->grallocutils ||
          !qti_select_metadata_backend(gr->grallocutils, gr))
         goto fail;

      symbol_scope = gr->grallocutils;
   }

   gr->fallback = u_gralloc_fallback_create();
   if (!gr->fallback)
      goto fail;

   gr->base.ops.get_buffer_basic_info = qti_get_buffer_basic_info;
   if (load_function(symbol_scope, gr->backend->color_space_symbol,
                     &gr->get_color_space))
      gr->base.ops.get_buffer_color_info = qti_get_buffer_color_info;
   gr->base.ops.destroy = qti_metadata_gralloc_destroy;
   gr->base.capabilities = U_GRALLOC_CAP_EXPLICIT_YUV_LAYOUT;

   /* Use the exact exported QTI allocation-policy symbol as a second runtime
    * ABI gate for private producer bit 0.  Do not call it here: gralloc must
    * make the final per-buffer decision after Android has combined producer,
    * consumer, and compositor usages.  Absence of the symbol only disables
    * the allocation request; the metadata bridge remains usable.
    * The legacy ABI can validate already allocated RGB/YUV buffers, but it
    * does not establish that Vulkan private producer bit 0 requests the same
    * RGB UBWC layout on every revision.  Never turn import support into an
    * allocation-policy claim.  The modern ABI additionally needs standard
    * allocation metadata so every UBWC buffer requested here can be checked
    * against its immutable FourCC, modifier, size, usage, and private flags
    * before import.
    */
   if (gr->backend->supports_swapchain_ubwc && gr->get_metadata_value) {
      IsQtiUbwcEnabled is_ubwc_enabled;
      if (load_function(symbol_scope, QTI_IS_UBWC_ENABLED_SYMBOL,
                        &is_ubwc_enabled)) {
         gr->base.capabilities |= U_GRALLOC_CAP_QCOM_SWAPCHAIN_UBWC;
      }
   }

   mesa_logi("Using runtime QTI gralloc %s metadata backend%s%s",
             gr->backend->name,
             gr->get_metadata_value
                ? " with standard Mapper metadata validation"
                : "",
             gr->base.capabilities & U_GRALLOC_CAP_QCOM_SWAPCHAIN_UBWC
                ? " with swapchain UBWC allocation support"
                : "");
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
