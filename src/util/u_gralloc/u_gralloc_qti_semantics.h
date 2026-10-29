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

#include "drm-uapi/drm_fourcc.h"

/* Qualcomm repurposes the low 56 bits of its vendor DRM modifier as flags
 * (display/drm/sde_drm.h), so they combine with the generic QCOM COMPRESSED
 * and TILED tokens gralloc reports:
 *
 *   0x1 COMPRESSED  UBWC, compressed
 *   0x2 DX          ten-bit variant of the base format
 *   0x4 TIGHT       ten-bit, 4/3 packed ("TP10")
 *   0x8 TILE        UBWC, compression disabled
 *
 * The vendor byte must be masked off before testing a flag:
 * fourcc_mod_code(QCOM, flag) carries the vendor byte as well, so ANDing a
 * modifier with it yields a non-zero vendor byte for *every* QCOM modifier
 * and the test would always succeed.
 */
#define U_GRALLOC_QTI_MOD_COMPRESSED (UINT64_C(1) << 0)
#define U_GRALLOC_QTI_MOD_DX         (UINT64_C(1) << 1)
#define U_GRALLOC_QTI_MOD_TIGHT      (UINT64_C(1) << 2)
#define U_GRALLOC_QTI_MOD_TILE       (UINT64_C(1) << 3)
#define U_GRALLOC_QTI_MOD_FLAGS_MASK (UINT64_C(0x00ffffffffffffff))

/* Flags carried by a QTI vendor modifier, or 0 for any other modifier. */
static inline uint64_t
u_gralloc_qti_modifier_flags(uint64_t modifier)
{
   return fourcc_mod_is_vendor(modifier, QCOM)
             ? (modifier & U_GRALLOC_QTI_MOD_FLAGS_MASK)
             : 0;
}

/* Legacy (gralloc1-era, pre-gralloc4) Qualcomm HAL format codes and the DRM
 * identity of the buffers they describe.  That HAL generation has no
 * DRM-format reporting at all, so the identity comes from the public vendor
 * format definitions (QtiGrallocDefs.h) and the vendor plane-info rules
 * (gr_utils.cpp GetYUVPlaneInfo).  Only the legacy backends consult this
 * table; the standard transports never resolve formats through it.
 */
struct u_gralloc_qti_legacy_format {
   uint32_t hal_format; /* QTI HAL_PIXEL_FORMAT_* code */
   uint32_t drm_fourcc; /* DRM_FORMAT_* identity */
};

/* Returns NULL when the HAL format is not a legacy Qualcomm format. */
const struct u_gralloc_qti_legacy_format *
u_gralloc_qti_legacy_format_of(uint32_t hal_format);

#ifdef __cplusplus
}
#endif

#endif
