/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 2026
 * SPDX-License-Identifier: MIT
 *
 * Android gralloc 4 metadata through the HIDL IMapper@4.0 passthrough
 * contract, without linking any framework or HIDL runtime library.
 *
 * Devices that launched with gralloc 4 advertise the passthrough HAL in
 * their VINTF manifest (android.hardware.graphics.mapper@4.0::IMapper).
 * By the HIDL specification, a passthrough implementation is a vendor hw
 * module exporting the extern-C entry point "HIDL_FETCH_IMapper", and the
 * returned object implements exactly the virtual table that hidl-gen
 * generates from hardware/interfaces/graphics/mapper/4.0/IMapper.hal.
 *
 * This backend therefore:
 *   - discovers the module through the VINTF manifest (including
 *     per-HAL manifest fragments) and the vendor hw module directories,
 *   - loads it with dlopen in the caller's vendor linker namespace (the
 *     module and its dependencies are SELinux same_process_hal_file,
 *     unlike framework-private libui.so, which an app-loaded driver must
 *     never attempt to map),
 *   - resolves only HIDL_FETCH_IMapper (a C symbol; no C++ linkage names
 *     are ever resolved), and
 *   - queries only the standard gralloc 4 metadata keys
 *     (android.hardware.graphics.common StandardMetadataType), decoding
 *     the byte stream with the encoding documented in
 *     frameworks/native/libs/gralloc/types (identical to the stable-C
 *     wire format already used by the AIMapper backend).
 *
 * The interface class layout is frozen per HIDL versioning rules, so the
 * virtual slots below hold for every conformant IMapper@4.0
 * implementation.  They were determined empirically against the live
 * object on this device: the vtable was symbolized at run time
 * (slot1=interfaceChain, slot14=importBuffer, slot15=freeBuffer,
 * slot23=get on android.hardware.graphics.mapper@4.0-impl-qti-display.so,
 * Qualcomm gralloc 4, LineageOS 23.2, SM8550).  The interfaceChain
 * self-check below doubles as the layout verifier: an implementation
 * with a different prologue (extra virtual-dtor slots from a vendor
 * derived interface) will fail the check at slot1 and be refused rather
 * than called through wrong slots.
 *
 * Handles are queried directly as registered by the framework's own
 * GraphicBuffer import in each client process; the backend never performs
 * a competing importBuffer/freeBuffer cycle (doing so was observed to
 * unregister SurfaceFlinger's own reference to shared buffers).  If a
 * vendor ever refuses metadata on an unimported handle, queries fail
 * closed with a logged error instead.
 *
 * Before a buffer is ever queried, the backend self-checks by calling
 * interfaceChain through the same vtable mechanism and the same callback
 * ABI as every later call, requiring the returned interface list to
 * declare android.hardware.graphics.mapper@4.0::IMapper; on any mismatch
 * it refuses to load (fail closed) with an error log, so a mispredicted
 * environment degrades to the next backend instead of corrupting memory.
 */

#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <dirent.h>

#include <functional>
#include <string>
#include <vector>

#include "drm-uapi/drm_fourcc.h"
#include "util/log.h"

#include "u_gralloc_internal.h"

namespace {

/* android.hardware.graphics.common@1.2 StandardMetadataType (identical
 * numbering to the stable-C backend and StandardMetadataType.aidl).
 */
enum standard_metadata_type : int64_t {
   MD_BUFFER_ID = 1,
   MD_WIDTH = 3,
   MD_HEIGHT = 4,
   MD_LAYER_COUNT = 5,
   MD_PIXEL_FORMAT_FOURCC = 7,
   MD_DRM_PIXEL_FORMAT_MODIFIER = 8,
   MD_USAGE = 9,
   MD_ALLOCATION_SIZE = 10,
   MD_PROTECTED_CONTENT = 11,
   MD_COMPRESSION = 12,
   MD_CHROMA_SITING = 14,
   MD_PLANE_LAYOUTS = 15,
   MD_DATASPACE = 17,
};

constexpr char STANDARD_METADATA_NAME[] =
   "android.hardware.graphics.common.StandardMetadataType";

/* android.hardware.graphics.common.PlaneLayoutComponentType (standard
 * ExtendableType name and the Y/Cb/Cr values used to order YUV planes).
 */
constexpr char STANDARD_PLANE_COMPONENT_TYPE[] =
   "android.hardware.graphics.common.PlaneLayoutComponentType";
constexpr int64_t PLANE_COMPONENT_Y = INT64_C(1) << 0;
constexpr int64_t PLANE_COMPONENT_CB = INT64_C(1) << 1;
constexpr int64_t PLANE_COMPONENT_CR = INT64_C(1) << 2;

/* android.hardware.graphics.mapper@4.0 Error (int32_t-backed enum). */
enum pt_error : int32_t {
   PT_ERROR_NONE = 0,
   PT_ERROR_BAD_DESCRIPTOR = 1,
   PT_ERROR_BAD_BUFFER = 2,
   PT_ERROR_BAD_VALUE = 3,
   PT_ERROR_NO_RESOURCES = 5,
   PT_ERROR_UNSUPPORTED = 7,
};

/*
 * Layout mirrors of the HIDL support types, copied member-for-member from
 * AOSP system/libhidl android15-release base/include/hidl/HidlSupport.h
 * and .../graphics/mapper/4.0 MetadataType:
 *
 *   hidl_handle { hidl_pointer<const native_handle_t> mHandle;
 *                 bool mOwnsHandle; uint8_t mPad[7]; }
 *   hidl_string { hidl_pointer<const char> mBuffer;
 *                 uint32_t mSize; bool mOwnsBuffer; uint8_t mPad[3]; }
 *   hidl_vec<T> { T *mBuffer;  (kOffsetOfBuffer == 0, static_assert'd)
 *                 uint32_t mSize; bool mOwnsBuffer; uint8_t mPad[3]; }
 *   IMapper::MetadataType { hidl_string name; int64_t value; }
 *
 * These PODs are only ever read or filled externally by the vendor
 * implementation; nothing here invokes a member function of the real
 * classes, so no libhidlbase symbol is ever referenced.
 */
struct PtHidlString {
   const char *mBuffer;
   uint32_t mSize;
   bool mOwnsBuffer;
   uint8_t mPad[3];
};

struct PtHidlVecU8 {
   uint8_t *mBuffer;
   uint32_t mSize;
   bool mOwnsBuffer;
   uint8_t mPad[3];
};

struct PtMetadataType {
   PtHidlString name;
   int64_t value;
};

/*
 * Return<T> is returned through a caller-provided buffer because the real
 * class is larger than 16 bytes (android15-release libhidl:
 * return_status { Status{int32,int32,std::string} ; mutable bool } = 40
 * bytes, Return<T> adds the value).  A POD mirror of the same size
 * produces the identical AAPCS64 calling convention (hidden x8 sret
 * pointer) without ever constructing or destroying the real object: the
 * passthrough return path leaves the embedded std::string in its inline,
 * allocation-free state, so nothing needs to be destructed.
 */
struct PtReturnVoid {
   alignas(8) uint8_t storage[40];
};

struct PtReturnError {
   alignas(8) uint8_t storage[48];
};

struct PtHidlVecString {
   const PtHidlString *mBuffer;
   uint32_t mSize;
   bool mOwnsBuffer;
   uint8_t mPad[3];
};

static_assert(sizeof(PtHidlString) == 16);
static_assert(sizeof(PtHidlVecU8) == 16);
static_assert(sizeof(PtHidlVecString) == 16);
static_assert(sizeof(PtMetadataType) == 24);

/*
 * Virtual table slots of IMapper@4.0 (interface chain:
 * [dtor pair][IBase@1.0-1.2 virtuals][~IMapper pair][IMapper virtuals];
 * the dtor pair lives inside the 12/13/39/40 relocations of the QtiMapper
 * blob and the IMapper method group starts after it):
 *   1: IBase interfaceChain(std::function<void(const hidl_vec<hidl_string>&)>)
 *  14: importBuffer(const hidl_handle&, std::function<void(Error, void*)>)
 *  15: freeBuffer(void*) -> Return<Error>
 *  23: get(void*, const MetadataType&,
 *          std::function<void(Error, const hidl_vec<uint8_t>&)>)
 *
 * Verified at run time against the live object returned by
 * HIDL_FETCH_IMapper on this interface (vptr-symbolized probe:
 * slot1=interfaceChain ... slot23=get, slot24=set, matching
 * IMapper@4.0's declaration order exactly).
 */
enum pt_vtable_slot : uint32_t {
   SLOT_INTERFACE_CHAIN = 1,
   SLOT_GET = 23,
};

constexpr char EXPECTED_INTERFACE[] =
   "android.hardware.graphics.mapper@4.0::IMapper";

/*
 * Callbacks arrive through std::function objects constructed by this
 * module (statically linked libc++) and invoked by the vendor
 * implementation.  The layout mirrors above make the argument types
 * identical; the interfaceDescriptor self-check at load time exercises
 * exactly this callback mechanism before any buffer is touched.
 */
using GetCb =
   std::function<void(int32_t error, const PtHidlVecU8 &metadata)>;
using ChainCb = std::function<void(const PtHidlVecString &interfaces)>;

typedef PtReturnVoid (*pt_get_fn)(void *self, void *buffer,
                                  const PtMetadataType *metadata_type,
                                  GetCb cb);
typedef PtReturnVoid (*pt_chain_fn)(void *self, ChainCb cb);

typedef void *(*pt_fetch_fn)(const char *instance);

struct pt_gralloc {
   struct u_gralloc base;
   void *so;
   void *mapper;

   pt_get_fn get;
};

PtMetadataType
make_metadata_type(int64_t key)
{
   PtMetadataType md;
   md.name.mBuffer = STANDARD_METADATA_NAME;
   md.name.mSize = (uint32_t)strlen(STANDARD_METADATA_NAME);
   md.name.mOwnsBuffer = false;
   memset(md.name.mPad, 0, sizeof(md.name.mPad));
   md.value = key;
   return md;
}

/* One synchronous get() round-trip; returns the payload after the
 * validated [type-name][value] header, or false.
 */
bool
pt_get_blob(const struct pt_gralloc *gr, void *buffer, int64_t key,
            std::vector<uint8_t> *payload)
{
   bool called = false;
   int32_t error = PT_ERROR_UNSUPPORTED;
   std::vector<uint8_t> bytes;

   GetCb cb = [&](int32_t err, const PtHidlVecU8 &metadata) {
      called = true;
      error = err;
      if (err == PT_ERROR_NONE && metadata.mBuffer && metadata.mSize) {
         bytes.assign(metadata.mBuffer,
                      metadata.mBuffer + metadata.mSize);
      }
   };

   PtMetadataType md = make_metadata_type(key);
   gr->get(gr->mapper, buffer, &md, cb);

   if (!called || error != PT_ERROR_NONE || bytes.empty())
      return false;

   /* decodeMetadataType: int64 name length, name bytes, int64 value. */
   size_t pos = 0;
   auto read_i64 = [&](int64_t *out) -> bool {
      if (pos + sizeof(int64_t) > bytes.size())
         return false;
      memcpy(out, bytes.data() + pos, sizeof(int64_t));
      pos += sizeof(int64_t);
      return true;
   };

   int64_t name_len;
   if (!read_i64(&name_len) || name_len < 0 ||
       (size_t)name_len != strlen(STANDARD_METADATA_NAME) ||
       pos + (size_t)name_len > bytes.size())
      return false;
   if (memcmp(bytes.data() + pos, STANDARD_METADATA_NAME, (size_t)name_len) != 0)
      return false;
   pos += (size_t)name_len;

   int64_t type_value;
   if (!read_i64(&type_value) || type_value != key)
      return false;

   payload->assign(bytes.begin() + (long)pos, bytes.end());
   return true;
}

bool
pt_get_int64(const struct pt_gralloc *gr, void *buffer, int64_t key,
             int64_t *out)
{
   std::vector<uint8_t> payload;
   if (!pt_get_blob(gr, buffer, key, &payload))
      return false;
   if (payload.size() < sizeof(int64_t))
      return false;
   memcpy(out, payload.data(), sizeof(int64_t));
   return true;
}

bool
pt_get_uint32(const struct pt_gralloc *gr, void *buffer, int64_t key,
              uint32_t *out)
{
   std::vector<uint8_t> payload;
   if (!pt_get_blob(gr, buffer, key, &payload))
      return false;
   if (payload.size() < sizeof(uint32_t))
      return false;
   memcpy(out, payload.data(), sizeof(uint32_t));
   return true;
}

struct PtReader {
   const uint8_t *pos;
   const uint8_t *end;
   bool ok = true;

   bool read_i64(int64_t *out)
   {
      if ((size_t)(end - pos) < sizeof(int64_t)) {
         ok = false;
         return false;
      }
      memcpy(out, pos, sizeof(int64_t));
      pos += sizeof(int64_t);
      return true;
   }

   bool read_extendable(int64_t *value, bool *standard_component_type)
   {
      int64_t name_len;
      if (!read_i64(&name_len) || name_len < 0 ||
          (size_t)name_len > (size_t)(end - pos)) {
         ok = false;
         return false;
      }
      *standard_component_type =
         name_len == (int64_t)(sizeof(STANDARD_PLANE_COMPONENT_TYPE) - 1) &&
         !memcmp(pos, STANDARD_PLANE_COMPONENT_TYPE, (size_t)name_len);
      pos += (size_t)name_len;
      return read_i64(value);
   }

   bool skip_extendable()
   {
      int64_t value;
      bool standard;
      return read_extendable(&value, &standard);
   }
};

/* Reads the standard ExtendableType (name string + int64 value) payload of a
 * key whose value space is an extendable enum, e.g. COMPRESSION.  Only the
 * numeric value is returned; the standard enum backing value NONE is 0, and a
 * non-zero value under any (vendor) type name still means "not NONE".
 */
bool
pt_get_extendable_value(const struct pt_gralloc *gr, void *buffer, int64_t key,
                        int64_t *out)
{
   std::vector<uint8_t> payload;
   if (!pt_get_blob(gr, buffer, key, &payload))
      return false;

   PtReader reader{payload.data(), payload.data() + payload.size()};
   int64_t value;
   bool standard_component_type;
   if (!reader.read_extendable(&value, &standard_component_type) || !reader.ok)
      return false;

   *out = value;
   return true;
}

struct PtPlaneLayout {
   int64_t offsetInBytes;
   int64_t strideInBytes;

   /* Standard PlaneLayoutComponentType value when the plane has exactly one
    * component, else -1.
    */
   int64_t component_value = -1;
};

/* decodePlaneLayouts: int64 numPlanes, then per plane: int64
 * numComponents, per component ExtendableType + offsetInBits +
 * sizeInBits, then offsetInBytes, sampleIncrementInBits,
 * strideInBytes, widthInSamples, heightInSamples, totalSizeInBytes,
 * horizontalSubsampling, verticalSubsampling (all int64).
 */
bool
pt_get_plane_layouts(const struct pt_gralloc *gr, void *buffer,
                     std::vector<PtPlaneLayout> *out)
{
   std::vector<uint8_t> payload;
   if (!pt_get_blob(gr, buffer, MD_PLANE_LAYOUTS, &payload))
      return false;

   PtReader reader{payload.data(), payload.data() + payload.size()};
   int64_t num_planes;
   if (!reader.read_i64(&num_planes) || num_planes <= 0 || num_planes > 4)
      return false;

   for (int64_t i = 0; i < num_planes && reader.ok; i++) {
      int64_t num_components;
      if (!reader.read_i64(&num_components) || num_components < 0 ||
          num_components > 8)
         return false;
      int64_t component_value = -1;
      for (int64_t j = 0; j < num_components; j++) {
         int64_t value;
         bool standard_component = false;
         if (!reader.read_extendable(&value, &standard_component))
            return false;
         if (num_components == 1 && standard_component)
            component_value = value;
         int64_t offset_in_bits, size_in_bits;
         if (!reader.read_i64(&offset_in_bits) ||
             !reader.read_i64(&size_in_bits))
            return false;
      }

      PtPlaneLayout plane;
      int64_t v;
      if (!reader.read_i64(&plane.offsetInBytes))
         return false;
      if (!reader.read_i64(&v)) /* sampleIncrementInBits */
         return false;
      if (!reader.read_i64(&plane.strideInBytes))
         return false;
      if (!reader.read_i64(&v)) /* widthInSamples */
         return false;
      if (!reader.read_i64(&v)) /* heightInSamples */
         return false;
      if (!reader.read_i64(&v)) /* totalSizeInBytes */
         return false;
      if (!reader.read_i64(&v)) /* horizontalSubsampling */
         return false;
      if (!reader.read_i64(&v)) /* verticalSubsampling */
         return false;

      plane.component_value = component_value;
      out->push_back(plane);
   }

   return reader.ok && (int64_t)out->size() == num_planes;
}

/* QTI reports PlaneLayouts in logical Y-Cb-Cr order even for YV12, whose
 * storage and DRM fourcc order is Y-Cr-Cb.  vk_android swaps the two chroma
 * planes of DRM_FORMAT_YVU420 back into Vulkan's Y-Cb-Cr plane order, so
 * hand it DRM order here.  Only a plain three-plane layout whose standard
 * PlaneLayoutComponentType identifies every plane is rewritten; anything
 * else keeps the reported order.
 */
static bool
pt_normalize_yv12_plane_order(std::vector<PtPlaneLayout> *planes)
{
   if (planes->size() != 3)
      return false;

   int y = -1, cb = -1, cr = -1;
   for (size_t i = 0; i < planes->size(); i++) {
      if ((*planes)[i].component_value == PLANE_COMPONENT_Y)
         y = (int)i;
      else if ((*planes)[i].component_value == PLANE_COMPONENT_CB)
         cb = (int)i;
      else if ((*planes)[i].component_value == PLANE_COMPONENT_CR)
         cr = (int)i;
      else
         return false;
   }
   if (y < 0 || cb < 0 || cr < 0)
      return false;

   const PtPlaneLayout yv12[3] = { (*planes)[y], (*planes)[cr], (*planes)[cb] };
   planes->assign(yv12, yv12 + 3);
   return true;
}

/*
 * u_gralloc backend interface
 */

static int
pt_get_buffer_basic_info(struct u_gralloc *gralloc,
                         struct u_gralloc_buffer_handle *hnd,
                         struct u_gralloc_buffer_basic_info *out)
{
   struct pt_gralloc *gr = (struct pt_gralloc *)gralloc;

   if (!hnd || !hnd->handle || hnd->handle->numFds < 1)
      return -EINVAL;

   /* Pass the caller's handle directly. In every client process the
    * buffer is already registered with the passthrough mapper by the
    * framework's own GraphicBuffer import; calling importBuffer here
    * too would register a *second* reference whose matching
    * freeBuffer() can tear down the framework's registration
    * (observed to break SurfaceFlinger composition). This mirrors
    * how libui's own clients query buffers they already imported.
    */
   void *imported = (void *)(uintptr_t)hnd->handle;

   int ret = -EINVAL;
   do {
      uint32_t fourcc = 0;
      bool fourcc_known =
         pt_get_uint32(gr, imported, MD_PIXEL_FORMAT_FOURCC, &fourcc) &&
         fourcc != 0;
      if (!fourcc_known) {
         /* PIXEL_FORMAT_FOURCC is optional standard metadata: an
          * implementation may leave it unsupported, or report
          * DRM_FORMAT_INVALID, for a format it does not map (observed with
          * FP16 on gralloc-4 era vendors).  Report the absence through the
          * flags instead of refusing the buffer; consumers resolve the
          * format from the platform-level buffer format and fail closed
          * if they cannot.
          */
         mesa_logw_once("u_gralloc: no DRM fourcc in the standard metadata; "
                        "buffer format must be resolved by the consumer");
         fourcc = 0;
      }

      int64_t modifier = -1;
      if (!pt_get_int64(gr, imported, MD_DRM_PIXEL_FORMAT_MODIFIER, &modifier))
         break;

      int64_t allocation_size = 0;
      if (!pt_get_int64(gr, imported, MD_ALLOCATION_SIZE, &allocation_size) ||
          allocation_size <= 0)
         break;

       int64_t layer_count = 1;
       if (!pt_get_int64(gr, imported, MD_LAYER_COUNT, &layer_count) ||
           layer_count <= 0)
          layer_count = 1;

      const bool modifier_claims_linear =
         modifier == 0 || modifier == (int64_t)DRM_FORMAT_MOD_INVALID;

      /* Cross-check a linear/unknown modifier claim against the independent
       * standard COMPRESSION key: a non-NONE value means the buffer is
       * compressed, so the modifier claim must not be trusted and the
       * vendor plane geometry must not be consumed verbatim.
       */
      bool metadata_compressed = false;
      if (modifier_claims_linear) {
         int64_t compression = 0;
         if (pt_get_extendable_value(gr, imported, MD_COMPRESSION, &compression))
            metadata_compressed = compression != 0;
      }

      /*
        * Vendor layout trust policy (platform-neutral):
        *  - any compressed (non-linear, non-invalid) modifier: vendor
        *    PLANE_LAYOUTS may report only part of the planes (e.g.
        *    omitting a metadata plane) and must not be consumed verbatim;
        *    report the modifier verbatim and leave the geometry
        *    unverified so consumers derive the layout from
        *    (fourcc, modifier, dims) with their own authoritative rules,
        *  - a "linear" modifier contradicted either by the independent
        *    COMPRESSION metadata or by a first plane that does not start
        *    at offset 0 is self-contradictory: flag it (leaving
        *    interpretation to consumers that have authoritative layout
        *    knowledge), and
        *  - plain linear buffers keep their verbatim vendor layouts.
        */
      const bool compressed = !modifier_claims_linear || metadata_compressed;

      std::vector<PtPlaneLayout> planes;
      bool have_planes = pt_get_plane_layouts(gr, imported, &planes);
      bool contradictory = metadata_compressed;
      if (have_planes && !compressed && planes[0].offsetInBytes != 0) {
         contradictory = true;
         have_planes = false;
      }
      if (compressed)
         have_planes = false;

      if (have_planes && !compressed && fourcc == DRM_FORMAT_YVU420 &&
          !pt_normalize_yv12_plane_order(&planes))
         mesa_logw_once("u_gralloc: YV12 plane components missing; keeping reported order");

      memset(out, 0, sizeof(*out));
      out->drm_fourcc = fourcc;
      out->modifier = modifier == (int64_t)DRM_FORMAT_MOD_INVALID
                         ? DRM_FORMAT_MOD_INVALID
                         : (uint64_t)modifier;
      out->alloc_size = (uint64_t)allocation_size;
      out->layer_count = (uint64_t)layer_count;
      if (!fourcc_known)
         out->flags |= U_GRALLOC_BUFFER_INFO_FOURCC_UNVERIFIED;

      if (have_planes) {
         out->num_planes = (int)planes.size();
         const int numFds = hnd->handle->numFds;
         bool valid = true;
         for (int i = 0; i < out->num_planes; i++) {
            const PtPlaneLayout &plane = planes[i];
            if (plane.offsetInBytes < 0 || plane.offsetInBytes > INT32_MAX ||
                plane.strideInBytes <= 0 || plane.strideInBytes > INT32_MAX) {
               valid = false;
               break;
            }
            out->offsets[i] = (int)plane.offsetInBytes;
            out->strides[i] = (int)plane.strideInBytes;
            out->fds[i] = (numFds == out->num_planes)
                             ? hnd->handle->data[i]
                             : hnd->handle->data[0];
         }
         if (!valid) {
            ret = -EINVAL;
            break;
         }
      } else {
         /*
          * The vendor implementation does not expose plane layouts
          * (optional per spec).  Expose only what the standard interface
          * guarantees and mark the geometry unverified so consumers
          * derive it from their own layout rules or fail closed.
          */
         if (hnd->pixel_stride > 0) {
            out->num_planes = (fourcc == DRM_FORMAT_NV21 ||
                               fourcc == DRM_FORMAT_NV12 ||
                               fourcc == DRM_FORMAT_P010)
                                 ? 2
                                 : 1;
            if (fourcc == DRM_FORMAT_YVU420)
               out->num_planes = 3;
            out->offsets[0] = 0;
            out->strides[0] = hnd->pixel_stride;
            for (int i = 0; i < out->num_planes; i++)
               out->fds[i] = hnd->handle->data[0];
         }
         out->flags |= U_GRALLOC_BUFFER_INFO_PLANES_UNVERIFIED;
      }

      if (contradictory)
         out->flags |= U_GRALLOC_BUFFER_INFO_PLANES_CONTRADICTORY;

      ret = 0;

   } while (0);

   return ret;
}

static int
pt_get_buffer_color_info(struct u_gralloc *gralloc,
                         struct u_gralloc_buffer_handle *hnd,
                         struct u_gralloc_buffer_color_info *out)
{
   struct pt_gralloc *gr = (struct pt_gralloc *)gralloc;

   out->yuv_color_space = __DRI_YUV_COLOR_SPACE_ITU_REC601;
   out->sample_range = __DRI_YUV_NARROW_RANGE;
   out->horizontal_siting = __DRI_YUV_CHROMA_SITING_0_5;
   out->vertical_siting = __DRI_YUV_CHROMA_SITING_0_5;

   void *imported = (void *)(uintptr_t)hnd->handle;

   uint32_t dataspace = 0;
   if (pt_get_uint32(gr, imported, MD_DATASPACE, &dataspace)) {
      switch (dataspace & (63u << 16)) {
      case 1u << 16 /* BT709 */:
         out->yuv_color_space = __DRI_YUV_COLOR_SPACE_ITU_REC709;
         break;
      case 6u << 16 /* BT2020 */:
      case 7u << 16 /* BT2020_CL */:
         out->yuv_color_space = __DRI_YUV_COLOR_SPACE_ITU_REC2020;
         break;
      default:
         break;
      }
      if ((dataspace & (7u << 27)) == 1u << 27 /* RANGE_FULL */)
         out->sample_range = __DRI_YUV_FULL_RANGE;
   }

   std::vector<uint8_t> payload;
   if (pt_get_blob(gr, imported, MD_CHROMA_SITING, &payload)) {
      PtReader reader{payload.data(), payload.data() + payload.size()};
      int64_t name_len;
      if (reader.read_i64(&name_len) && name_len >= 0 &&
          (size_t)name_len <= (size_t)(reader.end - reader.pos)) {
         /* ExtendableType: name bytes then int64 value.  The value space
          * is the standard ChromaSiting enum across the @1.2/HIDL and
          * AIDL eras, so the qualified name itself is only skipped.
          */
         reader.pos += (size_t)name_len;
         int64_t value = 0;
         if (reader.read_i64(&value)) {
            switch (value) {
            case 3 /* COSITED_HORIZONTAL */:
               out->horizontal_siting = __DRI_YUV_CHROMA_SITING_0;
               break;
            case 4 /* COSITED_VERTICAL */:
               out->vertical_siting = __DRI_YUV_CHROMA_SITING_0;
               break;
            case 5 /* COSITED_BOTH */:
               out->horizontal_siting = __DRI_YUV_CHROMA_SITING_0;
               out->vertical_siting = __DRI_YUV_CHROMA_SITING_0;
               break;
            default:
               break;
            }
         }
      }
   }

   return 0;
}

static int
pt_get_front_rendering_usage(struct u_gralloc *gralloc, uint64_t *out_usage)
{
   (void)gralloc;
   /* BufferUsage::FRONT_BUFFER (android.hardware.graphics.common) */
   *out_usage = 1ULL << 32;
   return 0;
}

static int
pt_destroy(struct u_gralloc *gralloc)
{
   /*
    * Keep the passthrough module loaded for the process lifetime: the
    * returned mapper object lives in it and framework clients never
    * unload passthrough HALs either.
    */
   free(gralloc);
   return 0;
}

/*
 * Discovery
 */

/* New- and old-style vendor manifests are all scanned, main files first,
 * then per-HAL fragments (mirrors the stable-C backend).
 */
constexpr const char *VINTF_MANIFEST_PATHS[] = {
   "/vendor/etc/vintf/manifest.xml",
   "/vendor/etc/vintf/manifest_odm.xml",
   "/odm/etc/vintf/manifest.xml",
};

constexpr const char *VINTF_MANIFEST_DIRS[] = {
   "/vendor/etc/vintf/manifest",
   "/odm/etc/vintf/manifest",
};

constexpr const char *MAPPER_HW_DIRS[] = {
   "/vendor/lib64/hw",
   "/odm/lib64/hw",
   "/vendor/lib/hw",
};

constexpr const char MAPPER_IMPL_PREFIX[] =
   "android.hardware.graphics.mapper@4.0-impl";

/* Collect "android.hardware.graphics.mapper@4.0-impl*.so" path candidates
 * declared passthrough in a VINTF manifest file (native format="hidl"
 * hal blocks naming android.hardware.graphics.mapper with version 4.0).
 * The manifest confirms the device declares the interface; the file name
 * comes from the hw directory scan because the fragment does not embed
 * it.
 */
bool
manifest_declares_mapper_4_0(const char *path,
                             std::vector<std::string> *instances)
{
   FILE *f = fopen(path, "re");
   if (!f)
      return false;

   fseek(f, 0, SEEK_END);
   long size = ftell(f);
   fseek(f, 0, SEEK_SET);
   if (size <= 0 || size > 512 * 1024) {
      fclose(f);
      return false;
   }

   std::vector<char> buf((size_t)size + 1, 0);
   if (fread(buf.data(), 1, (size_t)size, f) != (size_t)size) {
      fclose(f);
      return false;
   }
   fclose(f);

   const char *cursor = buf.data();
   bool found = false;
   while (const char *hal = strstr(cursor, "<hal ")) {
      const char *hal_end = strstr(hal, "</hal>");
      if (!hal_end)
         break;
      std::string block(hal, (size_t)(hal_end - hal));
      if (block.find("format=\"hidl\"") != std::string::npos &&
          block.find("<name>android.hardware.graphics.mapper</name>") !=
               std::string::npos &&
          block.find("<version>4.0</version>") != std::string::npos) {
         /* Collect every declared passthrough instance name. */
         const char *p = block.c_str();
         while (const char *inst = strstr(p, "<instance>")) {
            const char *inst_end = strstr(inst, "</instance>");
            if (!inst_end)
               break;
            std::string name(inst + 10, (size_t)(inst_end - inst - 10));
            size_t begin = name.find_first_not_of(" \t\r\n");
            size_t end = name.find_last_not_of(" \t\r\n");
            if (begin != std::string::npos)
               instances->push_back(name.substr(begin, end - begin + 1));
            p = inst_end + 11;
         }
         found = true;
      }
      cursor = hal_end + 6;
   }
   return found;
}

void
collect_mapper_candidates(std::vector<std::string> *out,
                          std::vector<std::string> *instances)
{
   bool declared = false;
   for (const char *manifest : VINTF_MANIFEST_PATHS) {
      if (manifest_declares_mapper_4_0(manifest, instances))
         declared = true;
   }
   for (const char *dir : VINTF_MANIFEST_DIRS) {
      DIR *d = opendir(dir);
      if (!d)
         continue;
      while (struct dirent *e = readdir(d)) {
         size_t len = strlen(e->d_name);
         if (len > 4 && strcmp(e->d_name + len - 4, ".xml") == 0) {
            char path[512];
            snprintf(path, sizeof(path), "%s/%s", dir, e->d_name);
            if (manifest_declares_mapper_4_0(path, instances))
               declared = true;
         }
      }
      closedir(d);
   }

   /* The hw module scan is authoritative for the file name; the VINTF
    * declaration is logged to keep the selection observable but does not
    * gate the probe (some ROMs ship a mapper@4.0 passthrough without a
    * device manifest entry, e.g. after framework/AIDL allocator
    * migration, while the gralloc4 metadata transport remains HIDL).
    */
   if (!declared)
      mesa_logi("u_gralloc: no VINTF declaration of "
                "android.hardware.graphics.mapper@4.0; probing hw modules anyway");

   for (const char *dir : MAPPER_HW_DIRS) {
      DIR *d = opendir(dir);
      if (!d)
         continue;
      while (struct dirent *e = readdir(d)) {
         const char *name = e->d_name;
         size_t len = strlen(name);
         if (len <= sizeof(MAPPER_IMPL_PREFIX) + 2 ||
             strncmp(name, MAPPER_IMPL_PREFIX,
                     sizeof(MAPPER_IMPL_PREFIX) - 1) != 0 ||
             strcmp(name + len - 3, ".so") != 0)
            continue;
         char path[512];
         snprintf(path, sizeof(path), "%s/%s", dir, name);
         bool dup = false;
         for (const auto &existing : *out) {
            if (existing == path) {
               dup = true;
               break;
            }
         }
         if (!dup)
            out->push_back(path);
      }
      closedir(d);
   }
}

/* Bind the vtable slots through the returned object pointer. */
bool
pt_bind_mapper(struct pt_gralloc *gr, void *mapper)
{
   void **vtable = *(void ***)mapper;

   gr->mapper = mapper;
   gr->get = (pt_get_fn)vtable[SLOT_GET];
   gr->base.ops.get_buffer_basic_info = pt_get_buffer_basic_info;
   gr->base.ops.get_buffer_color_info = pt_get_buffer_color_info;
   gr->base.ops.get_front_rendering_usage = pt_get_front_rendering_usage;
   gr->base.ops.destroy = pt_destroy;
   gr->base.type = U_GRALLOC_TYPE_IMAPPER4_PT;
   return true;
}

/*
 * Self-check: call interfaceChain (a read-only IBase method) through the
 * same vtable + std::function callback machinery that every metadata
 * query will use, and require the returned interface list to declare
 * android.hardware.graphics.mapper@4.0::IMapper.  A mismatch (wrong
 * slot mapping, callback ABI divergence, foreign object type) surfaces
 * here, before any buffer is touched, as a refusal to load rather than
 * as corrupted buffer information.
 */
bool
pt_self_check(void *mapper)
{
   void **vtable = *(void ***)mapper;
   auto chain = (pt_chain_fn)vtable[SLOT_INTERFACE_CHAIN];

   bool called = false;
   bool declares_mapper = false;
   std::string first;

   chain(mapper, [&](const PtHidlVecString &interfaces) {
      called = true;
      for (uint32_t i = 0; i < interfaces.mSize && i < 16; i++) {
         const PtHidlString &s = interfaces.mBuffer[i];
         if (!s.mBuffer || s.mSize == 0 || s.mSize > 256)
            continue;
         std::string name(s.mBuffer, s.mSize);
         if (i == 0)
            first = name;
         if (name == EXPECTED_INTERFACE)
            declares_mapper = true;
      }
   });

   if (!called) {
      mesa_loge("u_gralloc: IMapper@4.0 passthrough self-check: "
                "interfaceChain callback never ran");
      return false;
   }
   if (!declares_mapper) {
      mesa_loge("u_gralloc: IMapper@4.0 passthrough self-check: interface "
                "chain (first=\"%s\") does not declare %s",
                first.c_str(), EXPECTED_INTERFACE);
      return false;
   }
   return true;
}

struct u_gralloc *
try_open_mapper(const char *path, const std::vector<std::string> &instances)
{
   void *so = dlopen(path, RTLD_NOW | RTLD_LOCAL);
   if (!so) {
      mesa_logw("u_gralloc: dlopen(%s) failed: %s", path, dlerror());
      return NULL;
   }

   auto fetch = (pt_fetch_fn)dlsym(so, "HIDL_FETCH_IMapper");
   if (!fetch) {
      mesa_logw("u_gralloc: %s does not export HIDL_FETCH_IMapper", path);
      dlclose(so);
      return NULL;
   }

   std::vector<std::string> names = instances;
   if (names.empty())
      names.push_back("default");

   for (const auto &name : names) {
      void *mapper = fetch(name.c_str());
      if (!mapper) {
         mesa_logw("u_gralloc: HIDL_FETCH_IMapper(\"%s\") returned null "
                   "for %s",
                   name.c_str(), path);
         continue;
      }

      if (!pt_self_check(mapper))
         continue;

      struct pt_gralloc *gr = (struct pt_gralloc *)calloc(1, sizeof(*gr));
      if (!gr) {
         mesa_loge("u_gralloc: out of memory opening %s", path);
         continue;
      }

      gr->so = so;
      pt_bind_mapper(gr, mapper);

      mesa_logi("u_gralloc: gralloc4 metadata via IMapper@4.0 passthrough "
                "contract from %s instance \"%s\"",
                path, name.c_str());
      return &gr->base;
   }

   dlclose(so);
   return NULL;
}

} /* anonymous namespace */

extern "C" struct u_gralloc *
u_gralloc_imapper4_pt_api_create(void)
{
   std::vector<std::string> candidates;
   std::vector<std::string> instances;
   collect_mapper_candidates(&candidates, &instances);

   /* De-duplicate instance names. */
   for (size_t i = 0; i < instances.size();) {
      bool dup = false;
      for (size_t j = 0; j < i; j++) {
         if (instances[j] == instances[i]) {
            dup = true;
            break;
         }
      }
      if (dup)
         instances.erase(instances.begin() + (long)i);
      else
         i++;
   }

   if (candidates.empty()) {
      mesa_logi("u_gralloc: no android.hardware.graphics.mapper@4.0-impl*.so "
                "hw module found");
      return NULL;
   }

   for (const auto &path : candidates) {
      struct u_gralloc *gr = try_open_mapper(path.c_str(), instances);
      if (gr)
         return gr;
   }

   mesa_logw("u_gralloc: all %zu IMapper@4.0 passthrough candidates failed",
             candidates.size());
   return NULL;
}
