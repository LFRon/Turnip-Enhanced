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
#include <stdint.h>

#include "util/macros.h"
#include "gallium/include/mesa_interface.h"

struct u_gralloc;

/* Vulkan AHB/ANB and EGL native-buffer imports expose the public buffer
 * description alongside the native handle.  Width, height, and layer_count
 * may be zero for callers which cannot provide them; metadata backends use
 * non-zero values to cross-check allocator-reported plane geometry.  Usage is
 * valid only when has_usage is set because zero is itself a valid usage.
 */
struct u_gralloc_buffer_handle {
   const native_handle_t *handle;
   int hal_format;
   int pixel_stride;
   uint32_t width;
   uint32_t height;
   uint32_t layer_count;
   uint64_t usage;
   bool has_usage;
};

struct u_gralloc_modifier_plane_layout {
   uint64_t data_offset;
   uint64_t data_size;
   uint64_t metadata_size;
   uint32_t metadata_row_pitch;
};

struct u_gralloc_buffer_basic_info {
   uint32_t drm_fourcc;
   uint64_t modifier;

   int num_planes;
   int fds[4];
   int offsets[4];
   int strides[4];

   uint64_t alloc_size;
   uint64_t layer_count;

   /* For compressed layouts, offsets[] addresses the beginning of the
    * modifier plane (including auxiliary metadata).  modifier_plane_layouts[]
    * independently describes its metadata and primary-data regions so a
    * driver can validate its modifier-specific layout calculation before the
    * buffer is accessed by the GPU.  A zero metadata_row_pitch means that the
    * mapper did not expose a separate metadata plane (as with QCOM RGB UBWC).
    */
   struct u_gralloc_modifier_plane_layout modifier_plane_layouts[4];
   bool has_explicit_modifier_layout;
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
   /* Runtime QTI gralloc plane-metadata bridge.  Keep this after existing
    * public enum values; backend preference is defined by u_grallocs[].
    */
   U_GRALLOC_TYPE_QTI_METADATA,
   U_GRALLOC_TYPE_COUNT,
};

enum u_gralloc_capability {
   /* The backend reports an authoritative DRM FourCC, modifier, and explicit
    * per-plane layout for YUV buffers instead of inferring them from a HAL
    * format or a private native-handle layout.
    */
   U_GRALLOC_CAP_EXPLICIT_YUV_LAYOUT = 1u << 0,

   /* The backend is a verified QTI gralloc ABI where producer private bit 0
    * requests a UBWC allocation.  This is intentionally narrower than merely
    * recognizing QCOM-compressed buffer metadata: the private usage bit must
    * never be sent to an unrelated Android gralloc implementation.
    */
   U_GRALLOC_CAP_QCOM_SWAPCHAIN_UBWC = 1u << 1,
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

uint32_t u_gralloc_get_capabilities(struct u_gralloc *gralloc);

#ifdef __cplusplus
}
#endif

#endif /* U_GRALLOC_H */
