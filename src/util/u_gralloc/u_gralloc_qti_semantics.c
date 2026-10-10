/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 2026
 * SPDX-License-Identifier: MIT
 */

#include "u_gralloc_qti_semantics.h"

#include "drm-uapi/drm_fourcc.h"
#include "util/macros.h"

/* Vendor-private modifier bits (documented in the kernel UAPI sde_drm.h)
 * and vendor HAL format codes (public gralloc sources) used to classify the
 * ten-bit packing of NV12-looking buffers.
 */
#define QTI_HAL_PIXEL_FORMAT_YCbCr_420_P010        0x00000036u
#define QTI_HAL_PIXEL_FORMAT_YCbCr_420_P010_UBWC   0x00000124u
#define QTI_HAL_PIXEL_FORMAT_YCbCr_420_P010_VENUS  0x7fa30c0au
#define QTI_HAL_PIXEL_FORMAT_YCbCr_420_TP10_UBWC   0x7fa30c09u

/* Additional codes for the legacy format table below (QtiGrallocDefs.h). */
#define QTI_HAL_PIXEL_FORMAT_NV12_ENCODEABLE          0x00000102u
#define QTI_HAL_PIXEL_FORMAT_YCbCr_420_SP             0x00000109u
#define QTI_HAL_PIXEL_FORMAT_NV21_ZSL                 0x00000113u
#define QTI_HAL_PIXEL_FORMAT_YCrCb_420_SP_VENUS       0x00000114u
#define QTI_HAL_PIXEL_FORMAT_NV21_ENCODEABLE          0x7fa30c00u
#define QTI_HAL_PIXEL_FORMAT_YCrCb_420_SP_ADRENO      0x7fa30c01u
#define QTI_HAL_PIXEL_FORMAT_YCbCr_420_SP_VENUS       0x7fa30c04u
#define QTI_HAL_PIXEL_FORMAT_YCbCr_420_SP_VENUS_UBWC  0x7fa30c06u

#define QTI_DRM_FORMAT_MOD_COMPRESSED (UINT64_C(1) << 0)
#define QTI_DRM_FORMAT_MOD_DX         (UINT64_C(1) << 1)
#define QTI_DRM_FORMAT_MOD_TIGHT      (UINT64_C(1) << 2)

static bool
qti_hal_format_is_10bit(uint32_t hal_format)
{
   switch (hal_format) {
   case QTI_HAL_PIXEL_FORMAT_YCbCr_420_P010:
   case QTI_HAL_PIXEL_FORMAT_YCbCr_420_P010_UBWC:
   case QTI_HAL_PIXEL_FORMAT_YCbCr_420_P010_VENUS:
   case QTI_HAL_PIXEL_FORMAT_YCbCr_420_TP10_UBWC:
      return true;
   default:
      return false;
   }
}

enum u_gralloc_qti_yuv10_mode
u_gralloc_qti_get_yuv10_mode(uint32_t hal_format, uint32_t drm_fourcc,
                             uint64_t modifier)
{
   if (drm_fourcc != DRM_FORMAT_NV12)
      return U_GRALLOC_QTI_YUV10_NONE;

   if ((modifier >> 56) != DRM_FORMAT_MOD_VENDOR_QCOM ||
       !(modifier & QTI_DRM_FORMAT_MOD_COMPRESSED))
      return U_GRALLOC_QTI_YUV10_NONE;

   if (modifier & QTI_DRM_FORMAT_MOD_TIGHT)
      return U_GRALLOC_QTI_YUV10_TIGHT;

   if (modifier & QTI_DRM_FORMAT_MOD_DX)
      return U_GRALLOC_QTI_YUV10_P010;

   if (qti_hal_format_is_10bit(hal_format)) {
      return hal_format == QTI_HAL_PIXEL_FORMAT_YCbCr_420_TP10_UBWC
                ? U_GRALLOC_QTI_YUV10_TIGHT
                : U_GRALLOC_QTI_YUV10_P010;
   }

   return U_GRALLOC_QTI_YUV10_NONE;
}

static const struct u_gralloc_qti_legacy_format qti_legacy_formats[] = {
   /* 8-bit two-plane semi-planar */
   {QTI_HAL_PIXEL_FORMAT_NV12_ENCODEABLE, DRM_FORMAT_NV12, false},
   {QTI_HAL_PIXEL_FORMAT_YCbCr_420_SP, DRM_FORMAT_NV12, false},
   {QTI_HAL_PIXEL_FORMAT_YCbCr_420_SP_VENUS, DRM_FORMAT_NV12, false},
   {QTI_HAL_PIXEL_FORMAT_YCbCr_420_SP_VENUS_UBWC, DRM_FORMAT_NV12, false},
   {QTI_HAL_PIXEL_FORMAT_NV21_ENCODEABLE, DRM_FORMAT_NV21, false},
   {QTI_HAL_PIXEL_FORMAT_YCrCb_420_SP_ADRENO, DRM_FORMAT_NV21, false},
   {QTI_HAL_PIXEL_FORMAT_NV21_ZSL, DRM_FORMAT_NV21, false},
   {QTI_HAL_PIXEL_FORMAT_YCrCb_420_SP_VENUS, DRM_FORMAT_NV21, false},
   /* Packed ten-bit, transported under the NV12 identity */
   {QTI_HAL_PIXEL_FORMAT_YCbCr_420_TP10_UBWC, DRM_FORMAT_NV12, true},
   {QTI_HAL_PIXEL_FORMAT_YCbCr_420_P010_UBWC, DRM_FORMAT_NV12, true},
};

/* Other codes of the same generation are deliberately not listed and fail
 * closed: YCbCr_420_SP_TILED (0x7fa30c03, vendor plane info unsupported),
 * YCbCr_420_SP_4R_UBWC (0x7fa30c07, no NV12-equivalent layout),
 * YCbCr_420_P010_VENUS (0x7fa30c0a, linear ten-bit with no NV12 identity),
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
