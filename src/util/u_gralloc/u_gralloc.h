/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 2026 NXP
 * Copyright (C) 2022 Roman Stratiienko (r.stratiienko@gmail.com)
 * SPDX-License-Identifier: MIT
 */

#ifndef U_GRALLOC_H
#define U_GRALLOC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <cutils/native_handle.h>

#include <stdbool.h>

#include "util/macros.h"
#include "gallium/include/mesa_interface.h"

struct u_gralloc;

/* Both Vulkan and EGL API exposes HAL format / pixel stride which is required
 * by the fallback implementation.
 */
struct u_gralloc_buffer_handle {
   const native_handle_t *handle;
   int hal_format;
   int pixel_stride;
};

/* Flag for u_gralloc_buffer_basic_info.flags: plane offsets/strides were not
 * authoritatively provided by the gralloc metadata service and must not be
 * used for explicit-layout imports; consumers needing exact compressed plane
 * geometry should recompute it with their layout library or fail closed.
 */
#define U_GRALLOC_BUFFER_INFO_PLANES_UNVERIFIED (1u << 0)

/* Flag for u_gralloc_buffer_basic_info.flags: the reported modifier is
 * contradicted by independent standard metadata, either by plane geometry
 * (a "linear" buffer whose first plane does not start at offset 0) or by a
 * non-NONE COMPRESSION value combined with a linear/unknown modifier.  The
 * modifier claim is untrustworthy; consumers with authoritative layout
 * knowledge may reinterpret it.
 */
#define U_GRALLOC_BUFFER_INFO_PLANES_CONTRADICTORY (1u << 1)

/* Flag for u_gralloc_buffer_basic_info.flags: the standard PIXEL_FORMAT_FOURCC
 * key was absent, unsupported, or reported as DRM_FORMAT_INVALID (0).  The
 * buffer format cannot be resolved from that scalar key; consumers resolve it
 * from the platform buffer format (e.g. the Vulkan/Android format equivalence
 * table) or from verified plane layouts, and fail closed otherwise.
 */
#define U_GRALLOC_BUFFER_INFO_FOURCC_UNVERIFIED (1u << 2)

struct u_gralloc_buffer_basic_info {
   uint32_t drm_fourcc;
   uint64_t modifier;

   int num_planes;
   int fds[4];
   int offsets[4];
   int strides[4];

   uint64_t alloc_size;
   uint64_t layer_count;

   uint32_t flags;
};

struct u_gralloc_buffer_color_info {
   enum __DRIYUVColorSpace yuv_color_space;
   enum __DRISampleRange sample_range;
   enum __DRIChromaSiting horizontal_siting;
   enum __DRIChromaSiting vertical_siting;
};

enum u_gralloc_type {
   U_GRALLOC_TYPE_AUTO,
   U_GRALLOC_TYPE_GRALLOC4,
   U_GRALLOC_TYPE_CROS,
   U_GRALLOC_TYPE_LIBDRM,
   U_GRALLOC_TYPE_QCOM,
   U_GRALLOC_TYPE_FALLBACK,
   U_GRALLOC_TYPE_STABLEC,
   U_GRALLOC_TYPE_IMAPPER4_PT,
   U_GRALLOC_TYPE_COUNT,
};

struct u_gralloc *u_gralloc_create(enum u_gralloc_type type);

void u_gralloc_destroy(struct u_gralloc **gralloc);

int u_gralloc_get_buffer_basic_info(
   struct u_gralloc *gralloc,
   struct u_gralloc_buffer_handle *hnd,
   struct u_gralloc_buffer_basic_info *out);

int u_gralloc_get_buffer_color_info(
   struct u_gralloc *gralloc,
   struct u_gralloc_buffer_handle *hnd,
   struct u_gralloc_buffer_color_info *out);

int u_gralloc_get_front_rendering_usage(struct u_gralloc *gralloc,
                                        uint64_t *out_usage);

int u_gralloc_get_type(struct u_gralloc *gralloc);

#ifdef __cplusplus
}
#endif

#endif /* U_GRALLOC_H */
