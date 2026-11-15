/*
 * Copyright © 2026 Mesa contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef TU_NIR_LOWER_RAY_TRACING_H
#define TU_NIR_LOWER_RAY_TRACING_H

#include <stdbool.h>
#include <stdint.h>

#include "compiler/nir/nir.h"
#include "vulkan/vulkan_core.h"

#include "tu_ray_tracing_pipeline.h"

struct tu_ray_tracing_pipeline;

struct tu_rt_stage_func {
   VkShaderStageFlagBits stage;
};

struct tu_rt_lowering_info {
   struct tu_ray_tracing_pipeline *pipeline;
   struct tu_rt_stage_func *stages;
   uint32_t stage_count;
   uint32_t params_offset;

   /* Stage function clones, indexed as
    * [stage * levels * callable_levels + level * callable_levels + call_depth].
    * Raygen stages only have a clone at level 0.  Callable stages are cloned
    * once per callable nesting depth at level 0: executeCallableEXT does not
    * increment the ray recursion depth and callable shaders have no ray
    * context.  Other stages are cloned at levels 1..levels-1, call depth 0,
    * since they are only reachable through traceRayEXT.
    */
   nir_function **funcs;
   uint32_t levels;
   uint32_t callable_levels;

   /* Filled in by tu_nir_lower_ray_tracing(). */
   nir_variable *shader_record_ptr[TU_RT_MAX_RECURSION_DEPTH + 1];

   /* Dword offsets of the payload/callable data slot used by the stage at
    * each recursion level.
    */
   nir_variable *payload_ptr[TU_RT_MAX_RECURSION_DEPTH + 1];
   nir_variable *callable_ptr[TU_RT_MAX_RECURSION_DEPTH + 1];
};

bool
tu_nir_lower_ray_tracing(nir_shader *shader, struct tu_rt_lowering_info *info);

void
tu_nir_build_raygen_dispatch(nir_shader *shader,
                             const struct tu_rt_lowering_info *info);

#endif /* TU_NIR_LOWER_RAY_TRACING_H */
