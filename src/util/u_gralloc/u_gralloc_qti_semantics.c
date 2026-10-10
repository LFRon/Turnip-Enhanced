/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 2026
 * SPDX-License-Identifier: MIT
 */

#include "u_gralloc_qti_semantics.h"

#include "drm-uapi/drm_fourcc.h"
#include "util/macros.h"

/* Vendor HAL format codes (public gralloc sources: QtiGrallocDefs.h). */
#define QTI_HAL_PIXEL_FORMAT_YCbCr_420_P010        0x00000036u
#define QTI_HAL_PIXEL_FORMAT_YCbCr_420_P010_UBWC   0x00000124u
#define QTI_HAL_PIXEL_FORMAT_NV12_ENCODEABLE       0x00000102u
#define QTI_HAL_PIXEL_FORMAT_YCbCr_420_SP          0x00000109u
#define QTI_HAL_PIXEL_FORMAT_NV21_ZSL              0x00000113u
#define QTI_HAL_PIXEL_FORMAT_YCrCb_420_SP_VENUS    0x00000114u
#define QTI_HAL_PIXEL_FORMAT_NV21_ENCODEABLE       0x7fa30c00u
#define QTI_HAL_PIXEL_FORMAT_YCrCb_420_SP_ADRENO   0x7fa30c01u
#define QTI_HAL_PIXEL_FORMAT_YCbCr_420_SP_VENUS    0x7fa30c04u
#define QTI_HAL_PIXEL_FORMAT_YCbCr_420_SP_VENUS_UBWC 0x7fa30c06u

static const struct u_gralloc_qti_legacy_format qti_legacy_formats[] = {
   /* 8-bit two-plane semi-planar */
   {QTI_HAL_PIXEL_FORMAT_NV12_ENCODEABLE, DRM_FORMAT_NV12},
   {QTI_HAL_PIXEL_FORMAT_YCbCr_420_SP, DRM_FORMAT_NV12},
   {QTI_HAL_PIXEL_FORMAT_YCbCr_420_SP_VENUS, DRM_FORMAT_NV12},
   {QTI_HAL_PIXEL_FORMAT_YCbCr_420_SP_VENUS_UBWC, DRM_FORMAT_NV12},
   {QTI_HAL_PIXEL_FORMAT_NV21_ENCODEABLE, DRM_FORMAT_NV21},
   {QTI_HAL_PIXEL_FORMAT_YCrCb_420_SP_ADRENO, DRM_FORMAT_NV21},
   {QTI_HAL_PIXEL_FORMAT_NV21_ZSL, DRM_FORMAT_NV21},
   {QTI_HAL_PIXEL_FORMAT_YCrCb_420_SP_VENUS, DRM_FORMAT_NV21},
   /* Packed ten-bit.  This HAL generation has no modifier reporting, so
    * the packing identity comes from the format code itself: both P010
    * variants have a native Vulkan format.  TP10 (4/3 packed) has none
    * and is deliberately not listed, so it keeps failing closed.
    */
   {QTI_HAL_PIXEL_FORMAT_YCbCr_420_P010, DRM_FORMAT_P010},
   {QTI_HAL_PIXEL_FORMAT_YCbCr_420_P010_UBWC, DRM_FORMAT_P010},
};

/* Other codes of the same generation are deliberately not listed and fail
 * closed: YCbCr_420_SP_TILED (0x7fa30c03, vendor plane info unsupported),
 * YCbCr_420_SP_4R_UBWC (0x7fa30c07, no NV12-equivalent layout),
 * YCbCr_420_P010_VENUS (0x7fa30c0a, linear ten-bit with no NV12 identity),
 * YCbCr_420_TP10_UBWC (0x7fa30c09, no Vulkan format),
 * the FLEX family (camera-utils internals) and NV12_HEIF.
 */

const struct u_gralloc_qti_legacy_format *
u_gralloc_qti_legacy_format_of(uint32_t hal_format)
{
   for (unsigned i = 0; i < ARRAY_SIZE(qti_legacy_formats); i++) {
      if (qti_legacy_formats[i].hal_format == hal_format)
         return &qti_legacy_formats[i];
   }

   return NULL;
}
