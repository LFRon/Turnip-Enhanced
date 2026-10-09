/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef U_GRALLOC_QTI_SEMANTICS_H
#define U_GRALLOC_QTI_SEMANTICS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

/* Qualcomm-specific semantics of standard gralloc metadata values: the
 * ten-bit YUV packing of video and camera buffers is encoded in vendor bits
 * of the DRM modifier (and in vendor HAL formats).  Keep the interpretation
 * in one data-driven place; consumers only see the resulting mode.
 */
enum u_gralloc_qti_yuv10_mode {
   U_GRALLOC_QTI_YUV10_NONE = 0,
   U_GRALLOC_QTI_YUV10_TIGHT,
   U_GRALLOC_QTI_YUV10_P010,
};

enum u_gralloc_qti_yuv10_mode
u_gralloc_qti_get_yuv10_mode(uint32_t hal_format, uint32_t drm_fourcc,
                             uint64_t modifier);

#ifdef __cplusplus
}
#endif

#endif
