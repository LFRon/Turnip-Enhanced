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
