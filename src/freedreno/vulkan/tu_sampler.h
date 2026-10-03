/*
 * Copyright © 2014 Valentine Burley
 * SPDX-License-Identifier: MIT
 */

#ifndef TU_SAMPLER_H
#define TU_SAMPLER_H

#include "tu_common.h"

#include "vk_sampler.h"
#include "vk_ycbcr_conversion.h"

struct tu_sampler {
   struct vk_sampler vk;

   uint32_t descriptor[A6XX_TEX_SAMP_DWORDS];

   /* Number of image/sampler descriptor pairs reserved by an immutable
    * combined-image sampler.  Software YV12 reconstruction reuses those
    * slots for bounded plane and metadata buffer descriptors.
    */
   uint8_t plane_count;

   bool custom_border_color;
   uint32_t border_color_index;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(tu_sampler, vk.base, VkSampler,
                               VK_OBJECT_TYPE_SAMPLER)

#endif /* TU_SAMPLER_H */
