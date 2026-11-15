/*
 * Copyright © 2026 Mesa contributors
 * SPDX-License-Identifier: MIT
 */

#include "tu_ray_tracing_pipeline.h"

#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_control_flow.h"
#include "compiler/spirv/spirv.h"
#include "nir/nir.h"
#include "util/mesa-blake3.h"
#include "util/u_atomic.h"
#include "vk_nir.h"
#include "vk_pipeline.h"
#include "vk_util.h"

#include "tu_cmd_buffer.h"
#include "tu_common.h"
#include "tu_device.h"
#include "tu_nir_lower_ray_tracing.h"
#include "tu_pipeline.h"
#include "tu_shader.h"


static uint32_t rt_group_handle_counter;

void
tu_ray_tracing_pipeline_destroy(struct tu_device *dev,
                                struct tu_ray_tracing_pipeline *rt)
{
   if (!rt)
      return;
   free(rt->groups);
   free(rt->stages);
   ralloc_free(rt->mem_ctx);
   free(rt);
}

static struct nir_function *
clone_stage_function(nir_shader *shader, nir_shader *stage, const char *name)
{
   nir_function_impl *entry = nir_shader_get_entrypoint(stage);
   nir_function_impl *copy = nir_function_impl_clone(shader, entry);
   struct hash_table *var_remap = _mesa_pointer_hash_table_create(NULL);

   nir_foreach_block (block, copy) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;

         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (deref->deref_type != nir_deref_type_var)
            continue;

         /* Function-local variables are cloned by
          * nir_function_impl_clone() and must not be added to the shader.
          */
         if (deref->var->data.mode == nir_var_function_temp)
            continue;

         struct hash_entry *entry = _mesa_hash_table_search(var_remap, deref->var);
         if (!entry) {
            nir_variable *new_var = nir_variable_clone(deref->var, shader);
            nir_shader_add_variable(shader, new_var);
            entry = _mesa_hash_table_insert(var_remap, deref->var, new_var);
         }
         deref->var = static_cast<nir_variable *>(entry->data);
      }
   }

   _mesa_hash_table_destroy(var_remap, NULL);

   nir_function *func = nir_function_create(shader, name);
   nir_function_set_impl(func, copy);
   return func;
}


static nir_shader *
prepare_stage(struct tu_device *dev, void *mem_ctx,
              const VkRayTracingPipelineCreateInfoKHR *create_info,
              uint32_t stage_idx, VkPipelineCreateFlags2KHR flags)
{
   const VkPipelineShaderStageCreateInfo *stage_info =
      &create_info->pStages[stage_idx];
   mesa_shader_stage stage = vk_to_mesa_shader_stage(stage_info->stage);
   struct tu_shader_key key = {};

   nir_shader *nir = tu_spirv_to_nir(dev, mem_ctx, flags, stage_info, &key,
                                    stage);
   if (!nir)
      return NULL;

   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_inline_functions);

   return nir;
}

static VkResult
tu_rt_setup_group(struct tu_device *dev,
                  const VkRayTracingShaderGroupCreateInfoKHR *ginfo,
                  uint32_t stage_count, struct tu_rt_group *group)
{
   group->type = ginfo->type;
   group->general = UINT32_MAX;
   group->any_hit = UINT32_MAX;
   group->closest_hit = UINT32_MAX;
   group->intersection = UINT32_MAX;

   if (ginfo->pShaderGroupCaptureReplayHandle) {
      uint64_t replay_handle;
      memcpy(&replay_handle, ginfo->pShaderGroupCaptureReplayHandle,
             sizeof(replay_handle));
      group->handle = (uint32_t)replay_handle;
   } else {
      group->handle = p_atomic_inc_return(&rt_group_handle_counter);
   }

   switch (ginfo->type) {
   case VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR:
      if (ginfo->generalShader != VK_SHADER_UNUSED_KHR)
         group->general = ginfo->generalShader;
      break;
   case VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR:
      if (ginfo->anyHitShader != VK_SHADER_UNUSED_KHR)
         group->any_hit = ginfo->anyHitShader;
      if (ginfo->closestHitShader != VK_SHADER_UNUSED_KHR)
         group->closest_hit = ginfo->closestHitShader;
      break;
   case VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR:
      if (ginfo->anyHitShader != VK_SHADER_UNUSED_KHR)
         group->any_hit = ginfo->anyHitShader;
      if (ginfo->closestHitShader != VK_SHADER_UNUSED_KHR)
         group->closest_hit = ginfo->closestHitShader;
      if (ginfo->intersectionShader != VK_SHADER_UNUSED_KHR)
         group->intersection = ginfo->intersectionShader;
      break;
   default:
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "unsupported ray tracing shader group type");
   }

   if ((group->general != UINT32_MAX && group->general >= stage_count) ||
       (group->any_hit != UINT32_MAX && group->any_hit >= stage_count) ||
       (group->closest_hit != UINT32_MAX && group->closest_hit >= stage_count) ||
       (group->intersection != UINT32_MAX && group->intersection >= stage_count))
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "invalid ray tracing shader group index");

   return VK_SUCCESS;
}

static VkResult
tu_create_ray_tracing_pipeline(
   struct tu_device *dev,
   VkDeferredOperationKHR deferred_operation,
   VkPipelineCache pipeline_cache,
   const VkRayTracingPipelineCreateInfoKHR *create_info,
   VkPipelineCreateFlags2KHR flags,
   const VkAllocationCallbacks *alloc,
   VkPipeline *out_pipeline)
{
   (void) deferred_operation;
   (void) pipeline_cache;

   const struct tu_physical_device *phys = dev->physical_device;
   if (!phys->info->props.has_ray_intersection ||
       (phys->info->props.has_sw_fuse && !phys->has_raytracing))
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "ray tracing is not supported on this device");

   VK_FROM_HANDLE(tu_pipeline_layout, layout, create_info->layout);

   if (create_info->maxPipelineRayRecursionDepth > TU_RT_MAX_RECURSION_DEPTH)
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "maxPipelineRayRecursionDepth exceeds driver limit");

   const VkPipelineLibraryCreateInfoKHR *library_info =
      create_info->pLibraryInfo;
   uint32_t library_count =
      library_info ? library_info->libraryCount : 0;

   uint32_t own_stage_count = create_info->stageCount;
   uint32_t own_group_count = create_info->groupCount;
   uint32_t total_stage_count = own_stage_count;
   uint32_t total_group_count = own_group_count;

   for (uint32_t i = 0; i < library_count; i++) {
      VK_FROM_HANDLE(tu_pipeline, lib, library_info->pLibraries[i]);
      if (!lib || !lib->rt)
         return vk_errorf(dev, VK_ERROR_INITIALIZATION_FAILED,
                          "invalid ray tracing pipeline library");
      total_stage_count += lib->rt->stage_count;
      total_group_count += lib->rt->group_count;
   }

   void *mem_ctx = ralloc_context(NULL);
   if (!mem_ctx)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   struct tu_ray_tracing_pipeline *rt =
      static_cast<struct tu_ray_tracing_pipeline *>(calloc(1, sizeof(*rt)));
   if (!rt) {
      ralloc_free(mem_ctx);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   rt->mem_ctx = ralloc_context(NULL);
   rt->group_count = total_group_count;
   rt->stage_count = total_stage_count;
   rt->create_flags = flags;
   rt->max_recursion_depth = create_info->maxPipelineRayRecursionDepth;
   const uint32_t rt_levels = rt->max_recursion_depth + 1;

   rt->groups = static_cast<struct tu_rt_group *>(
      calloc(rt->group_count ? rt->group_count : 1, sizeof(*rt->groups)));
   rt->stages = static_cast<struct tu_rt_stage *>(
      calloc(rt->stage_count ? rt->stage_count : 1, sizeof(*rt->stages)));
   if (!rt->mem_ctx || !rt->groups || !rt->stages) {
      ralloc_free(mem_ctx);
      tu_ray_tracing_pipeline_destroy(dev, rt);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   for (uint32_t i = 0; i < own_stage_count; i++) {
      struct nir_shader *nir =
         prepare_stage(dev, rt->mem_ctx, create_info, i, flags);
      if (!nir) {
         ralloc_free(mem_ctx);
         tu_ray_tracing_pipeline_destroy(dev, rt);
         return VK_ERROR_UNKNOWN;
      }

      rt->stages[i].stage = create_info->pStages[i].stage;
      /* The driver keeps its traversal state in registers/local memory, but
       * the API still requires a non-zero stack size to be reported for
       * applications that explicitly manage the ray tracing stack.
       */
      rt->stages[i].stack_size = 16;
      rt->stages[i].nir = nir;
   }

   for (uint32_t i = 0; i < own_group_count; i++) {
      VkResult result = tu_rt_setup_group(
         dev, &create_info->pGroups[i], own_stage_count, &rt->groups[i]);
      if (result != VK_SUCCESS) {
         ralloc_free(mem_ctx);
         tu_ray_tracing_pipeline_destroy(dev, rt);
         return result;
      }
   }

   /* Shader groups defined in linked libraries come after the groups of the
    * pipeline itself.  Shader indices are relative to each library, so they
    * have to be rebased to the merged stage array.
    */
   uint32_t stage_offset = own_stage_count;
   uint32_t group_offset = own_group_count;
   for (uint32_t i = 0; i < library_count; i++) {
      VK_FROM_HANDLE(tu_pipeline, lib, library_info->pLibraries[i]);
      struct tu_ray_tracing_pipeline *lib_rt = lib->rt;

      for (uint32_t j = 0; j < lib_rt->stage_count; j++)
         rt->stages[stage_offset + j] = lib_rt->stages[j];

      for (uint32_t j = 0; j < lib_rt->group_count; j++) {
         struct tu_rt_group group = lib_rt->groups[j];
         if (group.general != UINT32_MAX)
            group.general += stage_offset;
         if (group.any_hit != UINT32_MAX)
            group.any_hit += stage_offset;
         if (group.closest_hit != UINT32_MAX)
            group.closest_hit += stage_offset;
         if (group.intersection != UINT32_MAX)
            group.intersection += stage_offset;
         group.handle = group_offset + j;
         rt->groups[group_offset + j] = group;
      }

      stage_offset += lib_rt->stage_count;
      group_offset += lib_rt->group_count;
   }

   uint32_t app_push_size = layout ? layout->push_constant_size : 0;
   uint32_t params_offset = align(app_push_size, 16);

   if (params_offset + sizeof(struct tu_rt_trace_params) > MAX_PUSH_CONSTANTS_SIZE) {
      ralloc_free(mem_ctx);
      tu_ray_tracing_pipeline_destroy(dev, rt);
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "ray tracing push constants exceed driver limit");
   }
   rt->app_push_size = app_push_size;
   rt->params_offset = params_offset;

   struct tu_compute_pipeline *compute = NULL;
   VkResult result;

   if (flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR) {
      /* A pipeline library is only a container for stages and groups.  It is
       * never bound, so compile a trivial compute shader just to have a valid
       * pipeline object for the API.  In particular do not try to build the
       * monolithic shader here: the library may not have a raygen stage and
       * may have been created with a null pipeline layout.
       */
      nir_builder dummy = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, ir3_get_compiler_options(dev->compiler),
         "tu ray tracing pipeline library");
      dummy.shader->info.workgroup_size[0] = 1;

      struct tu_pipeline_layout empty_layout = {};
      tu_pipeline_layout_init(&empty_layout);
      result = tu_compute_pipeline_create_from_nir(
         dev, pipeline_cache, &empty_layout, flags, alloc, dummy.shader,
         &compute);
   } else {
      nir_builder builder = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, ir3_get_compiler_options(dev->compiler),
         "tu ray tracing pipeline");
      nir_builder *b = &builder;
      b->shader->info.workgroup_size[0] = 1;

      struct tu_rt_lowering_info info = {
         .pipeline = rt,
         .stages = static_cast<struct tu_rt_stage_func *>(
            calloc(rt->stage_count ? rt->stage_count : 1, sizeof(*info.stages))),
         .stage_count = rt->stage_count,
         .params_offset = params_offset,
         .levels = rt_levels,
         .callable_levels = TU_RT_MAX_CALLABLE_DEPTH + 1,
      };
      info.funcs = static_cast<nir_function **>(
         calloc((rt->stage_count ? rt->stage_count : 1) * rt_levels *
                   info.callable_levels,
                sizeof(*info.funcs)));
      if (!info.stages || !info.funcs) {
         free(info.funcs);
         free(info.stages);
         ralloc_free(mem_ctx);
         tu_ray_tracing_pipeline_destroy(dev, rt);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }

      for (uint32_t i = 0; i < rt->stage_count; i++) {
         info.stages[i].stage = rt->stages[i].stage;

         /* Raygen only runs at level 0.  Callable shaders have no ray context
          * and cannot call traceRayEXT, so they are cloned once per supported
          * callable nesting depth at level 0.  All other stages first run as
          * the target of a traceRayEXT call from level 0, i.e. at level 1 or
          * deeper.
          */
         uint32_t first_level =
            rt->stages[i].stage == VK_SHADER_STAGE_RAYGEN_BIT_KHR ? 0 : 1;
         uint32_t first_call_depth = 0;
         uint32_t last_call_depth = 1;
         uint32_t last_level = rt_levels;
         if (rt->stages[i].stage == VK_SHADER_STAGE_CALLABLE_BIT_KHR) {
            first_level = 0;
            last_level = 1;
            last_call_depth = info.callable_levels;
         } else if (rt->stages[i].stage == VK_SHADER_STAGE_RAYGEN_BIT_KHR) {
            last_level = 1;
         }
         for (uint32_t level = first_level; level < last_level; level++) {
            for (uint32_t call_depth = first_call_depth;
                 call_depth < last_call_depth; call_depth++) {
               char *name = ralloc_asprintf(
                  b->shader, "rt_%s_%u_L%u_C%u",
                  _mesa_shader_stage_to_string(rt->stages[i].nir->info.stage),
                  i, level, call_depth);
               info.funcs[(i * rt_levels + level) * info.callable_levels +
                          call_depth] =
                  clone_stage_function(b->shader, rt->stages[i].nir, name);
            }
         }
      }

      tu_nir_lower_ray_tracing(b->shader, &info);
      tu_nir_build_raygen_dispatch(b->shader, &info);

      NIR_PASS(_, b->shader, nir_lower_continue_constructs);
      NIR_PASS(_, b->shader, nir_lower_returns);
      NIR_PASS(_, b->shader, nir_inline_functions);
      nir_function *entry_func = nir_shader_get_entrypoint(b->shader)->function;
      nir_foreach_function_safe (func, b->shader) {
         if (func != entry_func)
            exec_node_remove(&func->node);
      }
      NIR_PASS(_, b->shader, nir_remove_dead_derefs);
      NIR_PASS(_, b->shader, nir_remove_dead_variables,
               nir_var_shader_temp, NULL);
      NIR_PASS(_, b->shader, nir_lower_global_vars_to_local);
      NIR_PASS(_, b->shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b->shader, nir_lower_vars_to_explicit_types,
               nir_var_function_temp, glsl_get_natural_size_align_bytes);
      NIR_PASS(_, b->shader, nir_lower_explicit_io,
               nir_var_function_temp, nir_address_format_32bit_offset);
      NIR_PASS(_, b->shader, nir_remove_dead_variables,
               nir_var_function_temp, NULL);

      struct tu_pipeline_layout rt_layout = {};
      if (layout)
         rt_layout = *layout;
      rt_layout.push_constant_size =
         params_offset + sizeof(struct tu_rt_trace_params);
      tu_pipeline_layout_init(&rt_layout);

      result = tu_compute_pipeline_create_from_nir(
         dev, pipeline_cache, &rt_layout, flags, alloc, b->shader, &compute);

      free(info.funcs);
      free(info.stages);
   }

   if (result != VK_SUCCESS) {
      ralloc_free(mem_ctx);
      tu_ray_tracing_pipeline_destroy(dev, rt);
      return result;
   }

   compute->base.rt = rt;
   rt->stack_size = 16;
   *out_pipeline = tu_pipeline_to_handle(&compute->base);

   ralloc_free(mem_ctx);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
tu_CreateRayTracingPipelinesKHR(
   VkDevice device,
   VkDeferredOperationKHR deferredOperation,
   VkPipelineCache pipelineCache,
   uint32_t createInfoCount,
   const VkRayTracingPipelineCreateInfoKHR *pCreateInfos,
   const VkAllocationCallbacks *pAllocator,
   VkPipeline *pPipelines)
{
   VK_FROM_HANDLE(tu_device, dev, device);
   VkResult result = VK_SUCCESS;
   uint32_t i = 0;

   for (; i < createInfoCount; i++) {
      VkPipelineCreateFlags2KHR flags =
         vk_rt_pipeline_create_flags(&pCreateInfos[i]);

      VkPipelineCreationFeedbackCreateInfo *feedback =
         const_cast<VkPipelineCreationFeedbackCreateInfo *>(
            vk_find_struct_const(pCreateInfos[i].pNext,
                                 PIPELINE_CREATION_FEEDBACK_CREATE_INFO));
      if (feedback) {
         if (feedback->pPipelineCreationFeedback) {
            memset(feedback->pPipelineCreationFeedback, 0,
                   sizeof(*feedback->pPipelineCreationFeedback));
         }
         if (feedback->pPipelineStageCreationFeedbacks) {
            memset(feedback->pPipelineStageCreationFeedbacks, 0,
                   pCreateInfos[i].stageCount *
                      sizeof(*feedback->pPipelineStageCreationFeedbacks));
         }
      }

      VkResult tmp = tu_create_ray_tracing_pipeline(
         dev, deferredOperation, pipelineCache, &pCreateInfos[i], flags,
         pAllocator, &pPipelines[i]);
      if (tmp != VK_SUCCESS) {
         if (result == VK_SUCCESS)
            result = tmp;
         pPipelines[i] = VK_NULL_HANDLE;
         if (flags & VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT_KHR)
            break;
      }
   }

   for (; i < createInfoCount; i++)
      pPipelines[i] = VK_NULL_HANDLE;

   /* Pipeline creation is always synchronous, but applications that pass a
    * deferred operation expect the deferred result protocol.  Reporting the
    * operation as deferred and completing it immediately is valid and avoids
    * breaking applications that do not handle VK_OPERATION_NOT_DEFERRED_KHR.
    */
   if (result == VK_SUCCESS && deferredOperation != VK_NULL_HANDLE)
      return VK_OPERATION_DEFERRED_KHR;

   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
tu_GetRayTracingShaderGroupHandlesKHR(
   VkDevice device,
   VkPipeline pipeline,
   uint32_t firstGroup,
   uint32_t groupCount,
   size_t dataSize,
   void *pData)
{
   VK_FROM_HANDLE(tu_pipeline, tu_pipe, pipeline);
   if (!tu_pipe->rt ||
       firstGroup > tu_pipe->rt->group_count ||
       groupCount > tu_pipe->rt->group_count - firstGroup ||
       dataSize < (size_t)groupCount * TU_RT_HANDLE_SIZE)
      return VK_ERROR_INITIALIZATION_FAILED;

   uint8_t *data = static_cast<uint8_t *>(pData);
   if (dataSize)
      memset(data, 0, dataSize);
   for (uint32_t i = 0; i < groupCount; i++) {
      uint64_t handle = tu_pipe->rt->groups[firstGroup + i].handle;
      memcpy(data + i * TU_RT_HANDLE_SIZE, &handle, sizeof(handle));
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
tu_GetRayTracingCaptureReplayShaderGroupHandlesKHR(
   VkDevice device,
   VkPipeline pipeline,
   uint32_t firstGroup,
   uint32_t groupCount,
   size_t dataSize,
   void *pData)
{
   /* Group handles are deterministic numerical indices, which makes them a
    * valid capture replay payload as well.  There is no additional opaque
    * data to save.
    */
   return tu_GetRayTracingShaderGroupHandlesKHR(device, pipeline, firstGroup,
                                                groupCount, dataSize, pData);
}

VKAPI_ATTR VkDeviceSize VKAPI_CALL
tu_GetRayTracingShaderGroupStackSizeKHR(
   VkDevice device,
   VkPipeline pipeline,
   uint32_t group,
   VkShaderGroupShaderKHR groupShader)
{
   VK_FROM_HANDLE(tu_pipeline, tu_pipe, pipeline);
   if (!tu_pipe->rt || group >= tu_pipe->rt->group_count)
      return 0;

   const struct tu_rt_group *g = &tu_pipe->rt->groups[group];
   uint32_t stage_idx = UINT32_MAX;
   switch (groupShader) {
   case VK_SHADER_GROUP_SHADER_GENERAL_KHR:
      stage_idx = g->general;
      break;
   case VK_SHADER_GROUP_SHADER_CLOSEST_HIT_KHR:
      stage_idx = g->closest_hit;
      break;
   case VK_SHADER_GROUP_SHADER_ANY_HIT_KHR:
      stage_idx = g->any_hit;
      break;
   case VK_SHADER_GROUP_SHADER_INTERSECTION_KHR:
      stage_idx = g->intersection;
      break;
   default:
      return 0;
   }

   return stage_idx < tu_pipe->rt->stage_count ?
      tu_pipe->rt->stages[stage_idx].stack_size : 0;
}

VKAPI_ATTR void VKAPI_CALL
tu_CmdSetRayTracingPipelineStackSizeKHR(VkCommandBuffer commandBuffer,
                                        uint32_t pipelineStackSize)
{
   VK_FROM_HANDLE(tu_cmd_buffer, cmd, commandBuffer);

   /* The monolithic implementation does not use the application-visible
    * traversal stack, but keep the dynamic state value for completeness.
    */
   cmd->state.rt_stack_size = pipelineStackSize;
}

static void
fill_trace_params(struct tu_cmd_buffer *cmd,
                  const VkStridedDeviceAddressRegionKHR *raygen,
                  const VkStridedDeviceAddressRegionKHR *miss,
                  const VkStridedDeviceAddressRegionKHR *hit,
                  const VkStridedDeviceAddressRegionKHR *callable,
                  uint32_t width, uint32_t height, uint32_t depth,
                  bool indirect, bool indirect2, VkDeviceAddress indirect_addr)
{
   struct tu_ray_tracing_pipeline *rt = cmd->state.rt;
   struct tu_rt_trace_params params = {
      .raygen_sbt = raygen ? raygen->deviceAddress : 0,
      .miss_sbt = miss ? miss->deviceAddress : 0,
      .hit_sbt = hit ? hit->deviceAddress : 0,
      .callable_sbt = callable ? callable->deviceAddress : 0,
      .raygen_stride = raygen ? raygen->stride : 0,
      .miss_stride = miss ? miss->stride : 0,
      .hit_stride = hit ? hit->stride : 0,
      .callable_stride = callable ? callable->stride : 0,
      .launch_size = { width, height, depth },
      .flags = (indirect ? 1 : 0) | (indirect2 ? 2 : 0),
      .indirect_addr = indirect_addr,
   };
   memcpy(reinterpret_cast<char *>(cmd->push_constants) + rt->params_offset,
          &params, sizeof(params));
   cmd->state.dirty |= TU_CMD_DIRTY_SHADER_CONSTS;
}

VKAPI_ATTR void VKAPI_CALL
tu_CmdTraceRaysKHR(VkCommandBuffer commandBuffer,
                   const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
                   const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
                   const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
                   const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
                   uint32_t width, uint32_t height, uint32_t depth)
{
   VK_FROM_HANDLE(tu_cmd_buffer, cmd, commandBuffer);
   if (!cmd->state.rt || width == 0 || height == 0 || depth == 0)
      return;

   fill_trace_params(cmd, pRaygenShaderBindingTable, pMissShaderBindingTable,
                     pHitShaderBindingTable, pCallableShaderBindingTable,
                     width, height, depth, false, false, 0);
   tu_dispatch_rt_trace(commandBuffer, width, height, depth);
}

VKAPI_ATTR void VKAPI_CALL
tu_CmdTraceRaysIndirectKHR(VkCommandBuffer commandBuffer,
                           const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
                           const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
                           const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
                           const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
                           VkDeviceAddress indirectDeviceAddress)
{
   VK_FROM_HANDLE(tu_cmd_buffer, cmd, commandBuffer);
   if (!cmd->state.rt)
      return;

   fill_trace_params(cmd, pRaygenShaderBindingTable, pMissShaderBindingTable,
                     pHitShaderBindingTable, pCallableShaderBindingTable,
                     0, 0, 0, true, false, indirectDeviceAddress);
   tu_dispatch_rt_trace_indirect(commandBuffer, indirectDeviceAddress);
}

VKAPI_ATTR void VKAPI_CALL
tu_CmdTraceRaysIndirect2KHR(VkCommandBuffer commandBuffer,
                            VkDeviceAddress indirectDeviceAddress)
{
   VK_FROM_HANDLE(tu_cmd_buffer, cmd, commandBuffer);
   if (!cmd->state.rt)
      return;

   fill_trace_params(cmd, NULL, NULL, NULL, NULL,
                     0, 0, 0, true, true, indirectDeviceAddress);
   tu_dispatch_rt_trace_indirect2(commandBuffer, indirectDeviceAddress);
}
