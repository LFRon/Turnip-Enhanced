/*
 * Copyright © 2026 Mesa contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef TU_RAY_TRACING_PIPELINE_H
#define TU_RAY_TRACING_PIPELINE_H

#include <stdint.h>
#include <vulkan/vulkan_core.h>

struct tu_device;
struct nir_shader;

#define TU_RT_HANDLE_SIZE 8
#define TU_RT_WORKGROUP_SIZE 32
/* Fixed persistent grid for indirect trace dispatches.  The shader reads the
 * real launch dimensions from the indirect command and walks a strided ray
 * sequence, so the host never has to divide a GPU-side 3D launch size.
 */
#define TU_RT_INDIRECT_GROUPS 1024

/* Maximum ray tracing recursion depth supported by the monolithic pipeline.
 * The actual per-pipeline depth is min(maxPipelineRayRecursionDepth, this).
 */
#define TU_RT_MAX_RECURSION_DEPTH 4

/* Maximum nesting of executeCallableEXT calls.  Unlike traceRayEXT, callable
 * calls do not count towards maxPipelineRayRecursionDepth: the API only bounds
 * them by the implementation stack size.  The monolithic pipeline has to clone
 * callable stages per callable nesting level, so it supports a finite number of
 * frames.  The initial callable shader is depth 0.
 */
#define TU_RT_MAX_CALLABLE_DEPTH 4

/* Per-invocation call-data storage.  Every outgoing ray payload/callable
 * variable gets its own slot, sized to the largest such variable in the
 * pipeline; the incoming views are redirected to the slot of the call site at
 * run time.  Hit attributes are shared by the hit group and live after the
 * call-data slots.
 */
#define TU_RT_MIN_CALL_DATA_SIZE 16
#define TU_RT_HIT_ATTR_SIZE 64

struct tu_rt_trace_params {
   uint64_t raygen_sbt;
   uint64_t miss_sbt;
   uint64_t hit_sbt;
   uint64_t callable_sbt;
   uint32_t raygen_stride;
   uint32_t miss_stride;
   uint32_t hit_stride;
   uint32_t callable_stride;
   uint32_t launch_size[3];
   uint32_t flags;
   uint64_t indirect_addr;
};

struct tu_rt_group {
   VkRayTracingShaderGroupTypeKHR type;
   uint32_t general;
   uint32_t any_hit;
   uint32_t closest_hit;
   uint32_t intersection;
   uint32_t handle;
};

struct tu_rt_stage {
   uint32_t stack_size;
   VkShaderStageFlagBits stage;
   struct nir_shader *nir;
};

struct tu_ray_tracing_pipeline {
   /* Owns the prepared NIR shaders of the stages. */
   void *mem_ctx;

   /* Number of bytes the application owns at the start of push constants.
    * Turnip appends its trace parameters after this range.
    */
   uint32_t app_push_size;
   uint32_t params_offset;

   uint32_t group_count;
   struct tu_rt_group *groups;

   uint32_t stage_count;
   struct tu_rt_stage *stages;

   uint32_t max_recursion_depth;
   VkPipelineCreateFlags2KHR create_flags;

   VkDeviceSize stack_size;
};

void
tu_ray_tracing_pipeline_destroy(struct tu_device *dev,
                                struct tu_ray_tracing_pipeline *rt);

#endif /* TU_RAY_TRACING_PIPELINE_H */
