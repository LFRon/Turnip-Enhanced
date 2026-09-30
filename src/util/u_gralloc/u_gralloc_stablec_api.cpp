/*
 * Copyright (C) 2026
 * SPDX-License-Identifier: MIT
 *
 * Android IMapper stable-C (AIMapper) u_gralloc backend.
 *
 * The stable-C IMapper is the VINTF-stable, documented gralloc metadata
 * interface introduced with gralloc 5 (mandatory for Android 15+ vendor
 * images).  It is reached the same way libui's own Gralloc5 client reaches
 * it: the vendor-provided `mapper.<instance>.so` is opened through the
 * vendor linker namespace and the plain-C `AIMapper_loadIMapper` entry
 * point yields a versioned function-pointer table.  All buffer information
 * consumed here comes from `getStandardMetadata()` keys defined by
 * android.hardware.graphics.common.StandardMetadataType and from the
 * documented metadata wire format; no platform-private or vendor-private
 * C++ symbol is resolved, and no native-handle layout is parsed.
 */

#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <dirent.h>

#include <string>
#include <vector>

#include "drm-uapi/drm_fourcc.h"
#include "util/log.h"

#include "u_gralloc_internal.h"

#include <android/hardware/graphics/mapper/IMapper.h>

namespace {

/* android.hardware.graphics.common.StandardMetadataType */
enum standard_metadata_type : int64_t {
   MD_BUFFER_ID = 1,
   MD_NAME = 2,
   MD_WIDTH = 3,
   MD_HEIGHT = 4,
   MD_LAYER_COUNT = 5,
   MD_PIXEL_FORMAT_REQUESTED = 6,
   MD_PIXEL_FORMAT_FOURCC = 7,
   MD_DRM_PIXEL_FORMAT_MODIFIER = 8,
   MD_USAGE = 9,
   MD_ALLOCATION_SIZE = 10,
   MD_PROTECTED_CONTENT = 11,
   MD_COMPRESSION = 12,
   MD_CHROMA_SITING = 14,
   MD_PLANE_LAYOUTS = 15,
   MD_DATASPACE = 17,
   MD_STRIDE = 23,
};

/* Standard metadata payloads are prefixed with a header consisting of the
 * type-namespace string and the StandardMetadataType value, serialized as
 * an int64 length followed by raw bytes and little-endian integers.  This
 * mirrors the AIMapper stable-C metadata codec shipped with the interface
 * (implutils IMapperMetadataTypes.h).
 */
constexpr char STANDARD_METADATA_NAME[] =
   "android.hardware.graphics.common.StandardMetadataType";

/* PlaneLayoutComponent / ChromaSiting are ExtendableType values whose name
 * carries the aidl enum type.
 */
constexpr char CHROMA_SITING_EXT_NAME[] =
   "android.hardware.graphics.common.ChromaSiting";

/* aidl android.hardware.graphics.common.ChromaSiting */
enum chroma_siting : int64_t {
   CHROMA_SITING_NONE = 0,
   CHROMA_SITING_UNKNOWN = 1,
   CHROMA_SITING_INTERSTITIAL = 2,
   CHROMA_SITING_COSITED_HORIZONTAL = 3,
   CHROMA_SITING_COSITED_VERTICAL = 4,
   CHROMA_SITING_COSITED_BOTH = 5,
};

/* aidl android.hardware.graphics.common.Dataspace (used subset) */
constexpr int64_t DATASPACE_STANDARD_MASK = 63 << 16;
constexpr int64_t DATASPACE_STANDARD_BT709 = 1 << 16;
constexpr int64_t DATASPACE_STANDARD_BT601_625 = 2 << 16;
constexpr int64_t DATASPACE_STANDARD_BT601_525 = 4 << 16;
constexpr int64_t DATASPACE_STANDARD_BT2020 = 6 << 16;
constexpr int64_t DATASPACE_STANDARD_BT2020_CL = 7 << 16;
constexpr int64_t DATASPACE_RANGE_MASK = 7 << 27;
constexpr int64_t DATASPACE_RANGE_FULL = 1 << 27;

/* aidl android.hardware.graphics.common.BufferUsage::FRONT_BUFFER
 * (BufferUsage has @Backing(type="long"), so the wire width is int64.)
 */
constexpr uint64_t BUFFER_USAGE_FRONT = 1ULL << 32;

struct stablec_gralloc {
   struct u_gralloc base;
   void *so;
   const AIMapper *mapper;
};

typedef AIMapper_Error (*AIMapper_loadIMapperFn)(
   AIMapper *_Nullable *_Nonnull outImplementation);

class Reader {
 public:
   Reader(const void *data, size_t size)
      : m_src((const uint8_t *)data), m_rem(size)
   {
   }

   bool ok() const { return m_ok; }

   bool read_i64(int64_t *out) { return read_raw(out, sizeof(*out)); }
   bool read_u32(uint32_t *out) { return read_raw(out, sizeof(*out)); }

   bool skip_i64() { return read_raw(nullptr, sizeof(int64_t)); }

   bool read_string(std::string *out)
   {
      int64_t len;
      if (!read_i64(&len) || len < 0 || (uint64_t)len > m_rem) {
         m_ok = false;
         return false;
      }
      out->assign((const char *)m_src, (size_t)len);
      m_src += len;
      m_rem -= (size_t)len;
      return true;
   }

   /* Validates the standard-metadata header and consumes it. */
   bool check_standard_header(int64_t key)
   {
      std::string name;
      int64_t value;
      return read_string(&name) && read_i64(&value) && m_ok &&
             name == STANDARD_METADATA_NAME && value == key;
   }

   /* Consumes an ExtendableType (name + int64 value). */
   bool skip_extendable_type()
   {
      std::string name;
      int64_t value;
      return read_string(&name) && read_i64(&value);
   }

 private:
   bool read_raw(void *out, size_t size)
   {
      if (!m_ok || m_rem < size) {
         m_ok = false;
         return false;
      }
      if (out)
         memcpy(out, m_src, size);
      m_src += size;
      m_rem -= size;
      return true;
   }

   const uint8_t *m_src;
   size_t m_rem;
   bool m_ok = true;
};

bool
query_standard_metadata(const struct stablec_gralloc *gr,
                        buffer_handle_t buffer, int64_t key,
                        std::vector<uint8_t> *out, size_t *payload_offset)
{
   int32_t required =
      gr->mapper->v5.getStandardMetadata(buffer, key, nullptr, 0);
   if (required <= 0)
      return false;

   out->resize((size_t)required);
   int32_t written = gr->mapper->v5.getStandardMetadata(
      buffer, key, out->data(), out->size());
   if (written < 0 || (uint32_t)written > out->size()) {
      out->clear();
      return false;
   }
   out->resize((uint32_t)written);

   Reader reader(out->data(), out->size());
   if (!reader.check_standard_header(key) || !reader.ok()) {
      out->clear();
      return false;
   }

   /* The payload begins right after the validated name + key header. */
   *payload_offset = sizeof(int64_t) + sizeof(key) + strlen(STANDARD_METADATA_NAME);
   return true;
}

bool
get_standard_int64(const struct stablec_gralloc *gr, buffer_handle_t buffer,
                   int64_t key, int64_t *out)
{
   std::vector<uint8_t> blob;
   size_t offset = 0;
   if (!query_standard_metadata(gr, buffer, key, &blob, &offset))
      return false;

   Reader reader(blob.data() + offset, blob.size() - offset);
   int64_t value;
   if (!reader.read_i64(&value) || !reader.ok())
      return false;
   *out = value;
   return true;
}

/* Dataspace is an aidl enum with @Backing(type="int"): the payload is the
 * 4-byte underlying value, not int64.
 */
bool
get_standard_int32(const struct stablec_gralloc *gr, buffer_handle_t buffer,
                   int64_t key, int64_t *out)
{
   std::vector<uint8_t> blob;
   size_t offset = 0;
   if (!query_standard_metadata(gr, buffer, key, &blob, &offset))
      return false;

   Reader reader(blob.data() + offset, blob.size() - offset);
   uint32_t value;
   if (!reader.read_u32(&value) || !reader.ok())
      return false;
   *out = (int64_t)(int32_t)value;
   return true;
}

struct PlaneLayout {
   int64_t offsetInBytes;
   int64_t sampleIncrementInBits;
   int64_t strideInBytes;
   int64_t totalSizeInBytes;
   int64_t horizontalSubsampling;
   int64_t verticalSubsampling;
};

bool
get_standard_plane_layouts(const struct stablec_gralloc *gr,
                           buffer_handle_t buffer,
                           std::vector<PlaneLayout> *out)
{
   std::vector<uint8_t> blob;
   size_t offset = 0;
   if (!query_standard_metadata(gr, buffer, MD_PLANE_LAYOUTS, &blob, &offset))
      return false;

   Reader reader(blob.data() + offset, blob.size() - offset);
   int64_t num_planes = 0;
   if (!reader.read_i64(&num_planes) || num_planes <= 0 ||
       num_planes > 4)
      return false;

   for (int64_t i = 0; i < num_planes && reader.ok(); i++) {
      int64_t num_components = 0;
      if (!reader.read_i64(&num_components) || num_components < 0 ||
          num_components > 8)
         return false;
      for (int64_t j = 0; j < num_components; j++) {
         /* ExtendableType + offsetInBits + sizeInBits */
         if (!reader.skip_extendable_type() || !reader.skip_i64() ||
             !reader.skip_i64())
            return false;
      }

      PlaneLayout plane;
      if (!reader.read_i64(&plane.offsetInBytes) ||
          !reader.read_i64(&plane.sampleIncrementInBits) ||
          !reader.read_i64(&plane.strideInBytes) ||
          !reader.skip_i64() /* widthInSamples */ ||
          !reader.skip_i64() /* heightInSamples */ ||
          !reader.read_i64(&plane.totalSizeInBytes) ||
          !reader.read_i64(&plane.horizontalSubsampling) ||
          !reader.read_i64(&plane.verticalSubsampling))
         return false;

      out->push_back(plane);
   }

   return reader.ok() && (int64_t)out->size() == num_planes;
}

int
stablec_get_buffer_basic_info(struct u_gralloc *gralloc,
                              struct u_gralloc_buffer_handle *hnd,
                              struct u_gralloc_buffer_basic_info *out)
{
   struct stablec_gralloc *gr = (struct stablec_gralloc *)gralloc;

   if (!hnd || !hnd->handle || hnd->handle->numFds < 1)
      return -EINVAL;

   buffer_handle_t buffer = hnd->handle;
   buffer_handle_t imported = nullptr;
   const bool owns_import =
      gr->mapper->v5.importBuffer(hnd->handle, &imported) ==
         AIMAPPER_ERROR_NONE && imported != nullptr;
   if (owns_import)
      buffer = imported;

   int ret = -EINVAL;
   do {
      std::vector<uint8_t> blob;
      size_t offset = 0;
      uint32_t fourcc;
      if (!query_standard_metadata(gr, buffer, MD_PIXEL_FORMAT_FOURCC, &blob,
                                   &offset))
         break;
      {
         Reader r(blob.data() + offset, blob.size() - offset);
         if (!r.read_u32(&fourcc) || !r.ok())
            break;
      }

      int64_t modifier = -1;
      if (!get_standard_int64(gr, buffer, MD_DRM_PIXEL_FORMAT_MODIFIER,
                              &modifier))
         break;

      int64_t allocation_size = 0;
      if (!get_standard_int64(gr, buffer, MD_ALLOCATION_SIZE,
                              &allocation_size) ||
          allocation_size <= 0)
         break;

      int64_t layer_count = 1;
      get_standard_int64(gr, buffer, MD_LAYER_COUNT, &layer_count);
      if (layer_count <= 0)
         layer_count = 1;

      std::vector<PlaneLayout> planes;
      if (!get_standard_plane_layouts(gr, buffer, &planes))
         break;

      memset(out, 0, sizeof(*out));
      out->drm_fourcc = fourcc;
      out->modifier =
         modifier == (int64_t)DRM_FORMAT_MOD_INVALID
            ? DRM_FORMAT_MOD_INVALID
            : (uint64_t)modifier;
      out->num_planes = (int)planes.size();
      out->alloc_size = (uint64_t)allocation_size;
      out->layer_count = (uint64_t)layer_count;

      const int numFds = hnd->handle->numFds;
      for (int i = 0; i < out->num_planes; i++) {
         const PlaneLayout &plane = planes[i];
         if (plane.offsetInBytes < 0 || plane.offsetInBytes > INT32_MAX ||
             plane.strideInBytes <= 0 || plane.strideInBytes > INT32_MAX) {
            ret = -EINVAL;
            break;
         }
         out->offsets[i] = (int)plane.offsetInBytes;
         out->strides[i] = (int)plane.strideInBytes;
         /* Multi-planar gralloc buffers share the first dma-buf unless the
          * handle provides one fd per plane.
          */
         out->fds[i] =
            (numFds == out->num_planes) ? hnd->handle->data[i]
                                         : hnd->handle->data[0];
      }
      if (ret != -EINVAL)
         ret = 0;
   } while (0);

   if (owns_import)
      gr->mapper->v5.freeBuffer(imported);

   return ret;
}

int
stablec_get_buffer_color_info(struct u_gralloc *gralloc,
                              struct u_gralloc_buffer_handle *hnd,
                              struct u_gralloc_buffer_color_info *out)
{
   struct stablec_gralloc *gr = (struct stablec_gralloc *)gralloc;

   buffer_handle_t buffer = hnd->handle;
   buffer_handle_t imported = nullptr;
   const bool owns_import =
      gr->mapper->v5.importBuffer(hnd->handle, &imported) ==
         AIMAPPER_ERROR_NONE && imported != nullptr;
   if (owns_import)
      buffer = imported;

   /* Defaults match the legacy behavior: Rec.601, narrow range, siting at
    * the midpoint.
    */
   out->yuv_color_space = __DRI_YUV_COLOR_SPACE_ITU_REC601;
   out->sample_range = __DRI_YUV_NARROW_RANGE;
   out->horizontal_siting = __DRI_YUV_CHROMA_SITING_0_5;
   out->vertical_siting = __DRI_YUV_CHROMA_SITING_0_5;

   int64_t dataspace;
   if (get_standard_int32(gr, buffer, MD_DATASPACE, &dataspace)) {
      switch (dataspace & DATASPACE_STANDARD_MASK) {
      case DATASPACE_STANDARD_BT709:
         out->yuv_color_space = __DRI_YUV_COLOR_SPACE_ITU_REC709;
         break;
      case DATASPACE_STANDARD_BT2020:
      case DATASPACE_STANDARD_BT2020_CL:
         out->yuv_color_space = __DRI_YUV_COLOR_SPACE_ITU_REC2020;
         break;
      case DATASPACE_STANDARD_BT601_625:
      case DATASPACE_STANDARD_BT601_525:
      default:
         out->yuv_color_space = __DRI_YUV_COLOR_SPACE_ITU_REC601;
         break;
      }

      if ((dataspace & DATASPACE_RANGE_MASK) == DATASPACE_RANGE_FULL)
         out->sample_range = __DRI_YUV_FULL_RANGE;
   }

   int64_t chroma_siting;
   std::vector<uint8_t> blob;
   size_t offset = 0;
   if (query_standard_metadata(gr, buffer, MD_CHROMA_SITING, &blob, &offset)) {
      Reader r(blob.data() + offset, blob.size() - offset);
      std::string name;
      int64_t value;
      if (r.read_string(&name) && r.read_i64(&value) && r.ok() &&
          name == CHROMA_SITING_EXT_NAME) {
         chroma_siting = value;
         switch (chroma_siting) {
         case CHROMA_SITING_COSITED_HORIZONTAL:
            out->horizontal_siting = __DRI_YUV_CHROMA_SITING_0;
            out->vertical_siting = __DRI_YUV_CHROMA_SITING_0_5;
            break;
         case CHROMA_SITING_COSITED_VERTICAL:
            out->horizontal_siting = __DRI_YUV_CHROMA_SITING_0_5;
            out->vertical_siting = __DRI_YUV_CHROMA_SITING_0;
            break;
         case CHROMA_SITING_COSITED_BOTH:
            out->horizontal_siting = __DRI_YUV_CHROMA_SITING_0;
            out->vertical_siting = __DRI_YUV_CHROMA_SITING_0;
            break;
         case CHROMA_SITING_INTERSTITIAL:
         case CHROMA_SITING_NONE:
         case CHROMA_SITING_UNKNOWN:
         default:
            out->horizontal_siting = __DRI_YUV_CHROMA_SITING_0_5;
            out->vertical_siting = __DRI_YUV_CHROMA_SITING_0_5;
            break;
         }
      }
   }

   if (owns_import)
      gr->mapper->v5.freeBuffer(imported);

   return 0;
}

int
stablec_get_front_rendering_usage(struct u_gralloc *gralloc,
                                  uint64_t *out_usage)
{
   (void)gralloc;
   *out_usage = BUFFER_USAGE_FRONT;
   return 0;
}

int
stablec_destroy(struct u_gralloc *gralloc)
{
   struct stablec_gralloc *gr = (struct stablec_gralloc *)gralloc;

   /* Keep the mapper library loaded for the process lifetime: the
    * AIMapper table references code and data inside it, mirroring libui,
    * which never unloads the vendor mapper.
    */
   free(gr);
   return 0;
}

const struct u_gralloc_ops stablec_gralloc_ops = {
   stablec_get_buffer_basic_info,
   stablec_get_buffer_color_info,
   stablec_get_front_rendering_usage,
   stablec_destroy,
};

/*
 * Vendor IMapper stable-C libraries are declared as native HALs in the
 * device VINTF manifest (see hardware/interfaces/graphics/mapper/stable-c
 * README): /vendor/lib[64]/hw/mapper.<instance>.so where <instance> is the
 * declared passthrough instance name.  Discover them the same way libui
 * does, and additionally scan the vendor hw module directories so devices
 * whose manifest layout differs are still covered.  A short built-in list
 * is the last resort, and every failure is logged so the selection is
 * observable in a bugreport.
 */
constexpr const char *VINTF_MANIFEST_PATHS[] = {
   "/vendor/etc/vintf/manifest.xml",
   "/vendor/etc/vintf/manifest_odm.xml",
   "/odm/etc/vintf/manifest.xml",
};

constexpr const char *MAPPER_HW_DIRS[] = {
   "/vendor/lib64/hw",
   "/odm/lib64/hw",
   "/vendor/lib/hw",
};

constexpr const char *MAPPER_SUFFIX_FALLBACKS[] = {
   "qti",
   "snapalloc",
   "minigbm",
   "google",
   "unisoc",
   "rockchip",
   "exynos",
   "mediatek",
};

void
push_unique(std::vector<std::string> *list, const std::string &value)
{
   for (const auto &existing : *list) {
      if (existing == value)
         return;
   }
   list->push_back(value);
}

std::string
trim(const std::string &value)
{
   size_t begin = value.find_first_not_of(" \t\r\n");
   size_t end = value.find_last_not_of(" \t\r\n");
   if (begin == std::string::npos)
      return {};
   return value.substr(begin, end - begin + 1);
}

void
collect_mapper_instances_from_manifest(const char *path,
                                       std::vector<std::string> *out)
{
   FILE *f = fopen(path, "re");
   if (!f)
      return;

   fseek(f, 0, SEEK_END);
   long size = ftell(f);
   fseek(f, 0, SEEK_SET);
   if (size <= 0 || size > 512 * 1024) {
      fclose(f);
      return;
   }

   std::vector<char> buf((size_t)size + 1, 0);
   if (fread(buf.data(), 1, (size_t)size, f) != (size_t)size) {
      fclose(f);
      return;
   }
   fclose(f);

   /* Scan <hal ...>...</hal> blocks and pick native entries whose name is
    * "mapper", collecting every declared <instance>.
    */
   const char *cursor = buf.data();
   while (const char *hal = strstr(cursor, "<hal ")) {
      const char *hal_end = strstr(hal, "</hal>");
      if (!hal_end)
         break;

      std::string block(hal, (size_t)(hal_end - hal));
      if (block.find("format=\"native\"") != std::string::npos &&
          block.find("<name>mapper</name>") != std::string::npos) {
         const char *p = block.c_str();
         while (const char *inst = strstr(p, "<instance>")) {
            const char *inst_end = strstr(inst, "</instance>");
            if (!inst_end)
               break;
            push_unique(out, trim(std::string(inst + 10, (size_t)(inst_end - inst - 10))));
            p = inst_end + 11;
         }
      }

      cursor = hal_end + 6;
   }
}

void
collect_mapper_suffixes_from_hw_dirs(std::vector<std::string> *out)
{
   for (const char *dir : MAPPER_HW_DIRS) {
      DIR *d = opendir(dir);
      if (!d)
         continue;

      const size_t prefix_len = strlen("mapper.");
      while (struct dirent *e = readdir(d)) {
         const char *name = e->d_name;
         size_t len = strlen(name);
         if (len <= prefix_len + 3 || strncmp(name, "mapper.", prefix_len) != 0 ||
             strcmp(name + len - 3, ".so") != 0)
            continue;
         push_unique(out, std::string(name + prefix_len, len - prefix_len - 3));
      }
      closedir(d);
   }
}

struct stablec_gralloc *
try_load_mapper(const char *path)
{
   void *so = dlopen(path, RTLD_NOW | RTLD_LOCAL);
   if (!so)
      return nullptr;

   auto load = (AIMapper_loadIMapperFn)dlsym(so, "AIMapper_loadIMapper");
   if (!load) {
      dlclose(so);
      return nullptr;
   }

   AIMapper *mapper = nullptr;
   if (load(&mapper) != AIMAPPER_ERROR_NONE || !mapper ||
       mapper->version < AIMAPPER_VERSION_5) {
      dlclose(so);
      return nullptr;
   }

   struct stablec_gralloc *gr =
      (struct stablec_gralloc *)calloc(1, sizeof(*gr));
   if (!gr) {
      dlclose(so);
      return nullptr;
   }

   gr->so = so;
   gr->mapper = mapper;
   gr->base.ops = stablec_gralloc_ops;
   gr->base.type = U_GRALLOC_TYPE_STABLEC;

   mesa_logi("AIMapper stable-C v%u loaded from %s", mapper->version, path);
   return gr;
}

} /* anonymous namespace */

extern "C" struct u_gralloc *
u_gralloc_stablec_api_create(void)
{
   std::vector<std::string> suffixes;
   for (const char *manifest : VINTF_MANIFEST_PATHS)
      collect_mapper_instances_from_manifest(manifest, &suffixes);
   collect_mapper_suffixes_from_hw_dirs(&suffixes);
   for (const char *fallback : MAPPER_SUFFIX_FALLBACKS)
      push_unique(&suffixes, fallback);

   for (const auto &suffix : suffixes) {
      char path[256];
      for (const char *dir : MAPPER_HW_DIRS) {
         snprintf(path, sizeof(path), "%s/mapper.%s.so", dir, suffix.c_str());
         if (struct stablec_gralloc *gr = try_load_mapper(path))
            return &gr->base;
      }

      snprintf(path, sizeof(path), "mapper.%s.so", suffix.c_str());
      if (struct stablec_gralloc *gr = try_load_mapper(path))
         return &gr->base;
   }

   mesa_logw("No AIMapper stable-C HAL found (checked VINTF manifests and %zu suffix candidates)",
             suffixes.size());
   return NULL;
}
