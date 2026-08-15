/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 * SPDX-License-Identifier: MIT
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 */

#ifndef TU_FORMATS_H
#define TU_FORMATS_H

#include <stdbool.h>

#include "util/format/u_format.h"
#include "vulkan/vulkan_core.h"

#include "common/fd6_hw.h"

struct tu_physical_device;

struct tu_native_format
{
   enum a6xx_format fmt : 8;
   enum a3xx_color_swap swap : 8;
};

/* Adreno's native three-plane descriptor cannot represent every legal
 * Android YV12 layout.  Besides weak chroma-plane address alignment, Android
 * permits 16-byte row pitches which ordinary Adreno 2D texture accesses may
 * round to a larger hardware granularity.  Keep this format on Mesa's
 * bounded per-plane YCbCr lowering path instead of truncating an address or
 * letting the texture unit walk past the end of a plane.
 */
static inline bool
tu_format_uses_software_ycbcr(VkFormat format)
{
   return format == VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM;
}

struct tu_native_format tu6_format_vtx(enum pipe_format format);
struct tu_native_format tu6_format_color(enum pipe_format format, enum a6xx_tile_mode tile_mode,
                                         bool is_mutable);
struct tu_native_format tu6_format_texture(enum pipe_format format, enum a6xx_tile_mode tile_mode,
                                           bool is_mutable);

bool tu6_mutable_format_list_ubwc_compatible(const struct fd_dev_info *info,
                                             const VkImageFormatListCreateInfo *fmt_list);

bool tu_android_gralloc_ubwc_possible(
   struct tu_physical_device *physical_device,
   const VkPhysicalDeviceImageFormatInfo2 *info);

bool tu_external_format_resolve_supported(const struct fd_dev_info *info,
                                          VkFormat format,
                                          enum a6xx_tile_mode tile_mode);

#endif /* TU_FORMATS_H */
