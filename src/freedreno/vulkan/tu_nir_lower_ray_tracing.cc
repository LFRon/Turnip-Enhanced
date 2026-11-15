/*
 * Copyright © 2026 Mesa contributors
 * SPDX-License-Identifier: MIT
 *
 * Monolithic ray-tracing pipeline lowering for Turnip.
 *
 * All ray-tracing stages are cloned into one compute shader.  Ray query
 * traversal uses the hardware ray_intersection instruction; payload and hit
 * attributes live in a per-invocation local array so no cross-shader ABI is
 * needed.
 */

#include "tu_nir_lower_ray_tracing.h"

#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_control_flow.h"
#include "compiler/nir/nir_deref.h"
#include "util/u_dynarray.h"
#include "compiler/spirv/spirv.h"
#include "ir3/ir3_nir.h"

#include "bvh/tu_bvh_defines.h"
#include "tu_ray_traversal.h"
#include "tu_ray_tracing_pipeline.h"

/* Bounded per-level stack.  Overflow is handled by walking parent pointers
 * encoded in the BVH instead of silently dropping nodes, so this only affects
 * performance, not correctness.  16 entries matches RADV's MAX_STACK_ENTRY_COUNT
 * and keeps the shared-memory footprint small.
 */
#define TU_RT_MAX_STACK 16
#define TU_RT_INVALID_NODE TU_RAY_BVH_INVALID_NODE
#define TU_RT_NO_INSTANCE_ROOT TU_RAY_BVH_NO_INSTANCE_ROOT

enum tu_rt_call_data_kind {
   TU_RT_CALL_DATA_PAYLOAD,
   TU_RT_CALL_DATA_CALLABLE,
};

struct tu_rt_state {
   nir_variable *rt_data;
   uint32_t hit_attr_offset;
   nir_variable *payload_ptr;
   nir_variable *callable_ptr;

   nir_variable *tlas_base;
   nir_variable *root_bvh_base;
   nir_variable *bvh_base;
   nir_variable *current_node;
   nir_variable *previous_node;
   nir_variable *stack_ptr;
   nir_variable *top_stack;
   nir_variable *stack_low_watermark;
   nir_variable *instance_top_node;
   nir_variable *instance_bottom_node;
   nir_variable *stack;
   nir_variable *hit;

   nir_variable *origin;
   nir_variable *dir;
   nir_variable *world_origin;
   nir_variable *world_dir;
   nir_variable *tmin;
   nir_variable *tmax;
   nir_variable *ray_flags;
   nir_variable *node_mask;
   nir_variable *api_ray_flags;
   nir_variable *cull_mask;
   nir_variable *instance_flags;

   nir_variable *sbt_offset;
   nir_variable *sbt_stride;
   nir_variable *miss_index;

   nir_variable *instance_id;
   nir_variable *instance_sbt_offset_and_flags;

   nir_variable *shader_record_ptr;
   nir_variable *primitive_id;
   nir_variable *geometry_id;
   nir_variable *hit_kind;
   nir_variable *sbt_index;
   nir_variable *primitive_bvh_base;
   nir_variable *primitive_node;

   nir_variable *accept;
   nir_variable *terminate;
   nir_variable *opaque;
   nir_variable *report_t;
   nir_variable *report_kind;
};

struct tu_rt_lower_ctx {
   struct tu_rt_lowering_info *info;
   struct tu_rt_state traversal_states[TU_RT_MAX_RECURSION_DEPTH + 1];
   struct tu_rt_state hit_states[TU_RT_MAX_RECURSION_DEPTH + 1];
   struct tu_rt_state callable_states[TU_RT_MAX_CALLABLE_DEPTH + 1];
   struct tu_rt_state *state;
   uint32_t level;
   uint32_t levels;
   uint32_t callable_depth;
   uint32_t callable_levels;
   bool in_callable_stage;
   uint32_t hit_attr_base;
   struct hash_table *func_to_stage;
   struct hash_table *var_slots;
   uint32_t rt_data_size;

   /* If the pipeline has exactly one outgoing payload/callable slot, every
    * incoming view ultimately aliases that slot.  Use a compile-time dword
    * offset instead of the dynamic payload_ptr/callable_ptr indirection.
    */
   int32_t fixed_payload_slot;
   int32_t fixed_callable_slot;

   bool has_any_hit;
   bool has_closest_hit;
   bool has_intersection;
   bool has_miss;
   bool has_callable;
};

static nir_variable *
rt_var(nir_shader *shader, const struct glsl_type *type, const char *name)
{
   return nir_variable_create(shader, nir_var_shader_temp, type, name);
}

static nir_variable *
rt_shared_var(nir_shader *shader, const struct glsl_type *type,
              const char *name)
{
   return nir_variable_create(shader, nir_var_mem_shared, type, name);
}

static void
rt_state_init_one(nir_shader *shader, struct tu_rt_state *s,
                  nir_variable *rt_data, uint32_t hit_attr_offset,
                  bool traversal)
{
   const struct glsl_type *u32 = glsl_uint_type();
   const struct glsl_type *uvec2 = glsl_uvec2_type();
   const struct glsl_type *vec3 = glsl_vec_type(3);
   const struct glsl_type *u64 = glsl_uint64_t_type();

   *s = (struct tu_rt_state) {};
   s->rt_data = rt_data;
   s->hit_attr_offset = hit_attr_offset;
   s->payload_ptr = rt_var(shader, u32, "payload_ptr");
   s->callable_ptr = rt_var(shader, u32, "callable_ptr");

   s->tlas_base = rt_var(shader, uvec2, "tlas_base");
   if (traversal) {
      s->root_bvh_base = rt_var(shader, uvec2, "root_bvh_base");
      s->bvh_base = rt_var(shader, uvec2, "bvh_base");
      s->current_node = rt_var(shader, u32, "current_node");
      s->previous_node = rt_var(shader, u32, "previous_node");
      s->stack_ptr = rt_var(shader, u32, "stack_ptr");
      s->top_stack = rt_var(shader, u32, "top_stack");
      s->stack_low_watermark = rt_var(shader, u32, "stack_low_watermark");
      s->instance_top_node = rt_var(shader, u32, "instance_top_node");
      s->instance_bottom_node = rt_var(shader, u32, "instance_bottom_node");
      /* The traversal stack is per invocation.  It lives in shared memory so
       * that the persistent 32-thread raygen workgroup does not multiply
       * local-memory usage.  The array is slot-major: all invocations' entries
       * for a given stack depth are adjacent, matching RADV's LDS layout and
       * avoiding bank conflicts when the warp pushes/pops the same depth.
       */
      s->stack = rt_shared_var(shader,
         glsl_array_type(glsl_uint_type(),
                         TU_RT_MAX_STACK * TU_RT_WORKGROUP_SIZE, 0),
         "traversal_stack");
      s->hit = rt_var(shader, glsl_bool_type(), "hit");
   }

   s->origin = rt_var(shader, vec3, "origin");
   s->dir = rt_var(shader, vec3, "dir");
   s->world_origin = rt_var(shader, vec3, "world_origin");
   s->world_dir = rt_var(shader, vec3, "world_dir");
   s->tmin = rt_var(shader, glsl_float_type(), "tmin");
   s->tmax = rt_var(shader, glsl_float_type(), "tmax");
   s->ray_flags = rt_var(shader, u32, "ray_flags");
   s->node_mask = rt_var(shader, u32, "node_mask");
   s->api_ray_flags = rt_var(shader, u32, "api_ray_flags");
   s->cull_mask = rt_var(shader, u32, "cull_mask");
   s->instance_flags = rt_var(shader, u32, "instance_flags");

   s->sbt_offset = rt_var(shader, u32, "sbt_offset");
   s->sbt_stride = rt_var(shader, u32, "sbt_stride");
   s->miss_index = rt_var(shader, u32, "miss_index");

   s->instance_id = rt_var(shader, u32, "instance_id");
   s->instance_sbt_offset_and_flags =
      rt_var(shader, u32, "instance_sbt_offset_and_flags");

   s->shader_record_ptr = rt_var(shader, u64, "shader_record_ptr");
   s->primitive_id = rt_var(shader, u32, "primitive_id");
   s->geometry_id = rt_var(shader, u32, "geometry_id");
   s->hit_kind = rt_var(shader, u32, "hit_kind");
   s->sbt_index = rt_var(shader, u32, "sbt_index");
   s->primitive_bvh_base = rt_var(shader, uvec2, "primitive_bvh_base");
   s->primitive_node = rt_var(shader, u32, "primitive_node");

   s->accept = rt_var(shader, glsl_bool_type(), "accept");
   s->terminate = rt_var(shader, glsl_bool_type(), "terminate");
   s->opaque = rt_var(shader, glsl_bool_type(), "opaque");
   s->report_t = rt_var(shader, glsl_float_type(), "report_t");
   s->report_kind = rt_var(shader, u32, "report_kind");
}

static void
rt_state_init(nir_shader *shader, struct tu_rt_state *traversal_states,
              struct tu_rt_state *hit_states,
              struct tu_rt_state *callable_states, uint32_t levels,
              uint32_t rt_data_size, uint32_t hit_attr_base)
{
   nir_variable *rt_data = rt_var(
      shader, glsl_array_type(glsl_uint_type(), rt_data_size / 4, 0), "rt_data");

   for (uint32_t level = 0; level < levels; level++) {
      uint32_t hit_attr_offset =
         hit_attr_base + level * TU_RT_HIT_ATTR_SIZE;
      rt_state_init_one(shader, &traversal_states[level], rt_data,
                        hit_attr_offset, true);
      rt_state_init_one(shader, &hit_states[level], rt_data, hit_attr_offset,
                        false);
   }

   /* Callable shaders only see callable data and their shader record; they do
    * not have a ray context.  Give each callable nesting depth its own state
    * so a nested callable can never clobber its caller's record.
    */
   for (uint32_t depth = 0; depth < TU_RT_MAX_CALLABLE_DEPTH + 1; depth++)
      rt_state_init_one(shader, &callable_states[depth], rt_data,
                        hit_attr_base, false);
}

static nir_def *
load_vec(nir_builder *b, nir_variable *var)
{
   return nir_load_var(b, var);
}

static nir_def *
load_u32(nir_builder *b, nir_variable *var)
{
   return nir_load_var(b, var);
}

static nir_def *
load_u64(nir_builder *b, nir_variable *var)
{
   return nir_load_var(b, var);
}

static void
store_u32(nir_builder *b, nir_variable *var, nir_def *v)
{
   nir_store_var(b, var, v, 1);
}

static void
store_uvec2(nir_builder *b, nir_variable *var, nir_def *v)
{
   nir_store_var(b, var, v, 3);
}

static nir_def *
pack_uvec2(nir_builder *b, nir_def *v)
{
   return nir_pack_64_2x32(b, v);
}

static nir_def *
unpack_u64(nir_builder *b, nir_def *v)
{
   return nir_unpack_64_2x32(b, v);
}

static nir_def *
iadd_imm(nir_builder *b, nir_def *v, uint32_t imm)
{
   return nir_iadd_imm(b, v, imm);
}

static nir_def *
load_push(nir_builder *b, const struct tu_rt_lowering_info *info,
          uint32_t offset, unsigned components)
{
   return nir_load_push_constant(b, components, 32,
                                 nir_imm_int(b, info->params_offset + offset));
}

static nir_def *
load_trace_u32(nir_builder *b, const struct tu_rt_lowering_info *info,
               uint32_t offset)
{
   return load_push(b, info, offset, 1);
}

static nir_def *
load_trace_u64(nir_builder *b, const struct tu_rt_lowering_info *info,
               uint32_t offset)
{
   return pack_uvec2(b, load_push(b, info, offset, 2));
}

static nir_def *
load_tlas_record(nir_builder *b, nir_def *base, nir_def *record,
                 uint32_t offset, unsigned components)
{
   nir_def *byte_offset =
      nir_iadd_imm(b, nir_imul_imm(b, record, AS_RECORD_SIZE), offset);
   nir_def *addr = nir_iadd(b, pack_uvec2(b, base), nir_u2u64(b, byte_offset));
   return nir_load_global(b, components, 32, addr,
                          .access = static_cast<gl_access_qualifier>(
                                       ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                          .align_mul = 4);
}

static bool
is_as_descriptor(nir_def *accel)
{
   nir_intrinsic_instr *intr = nir_def_as_intrinsic_or_null(accel);
   return intr != NULL &&
      intr->intrinsic == nir_intrinsic_load_vulkan_descriptor;
}

static nir_def *
load_as_header(nir_builder *b, nir_def *accel)
{
   if (is_as_descriptor(accel)) {
      return nir_load_uav_ir3(
         b, 2, 32, accel,
         nir_vec2(b, nir_imm_int(b, 0),
                  nir_imm_int(b, offsetof(struct tu_accel_struct_header,
                                          self_ptr) / 4)),
         .access = static_cast<gl_access_qualifier>(
                      ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
         .align_mul = AS_RECORD_SIZE,
         .align_offset = offsetof(struct tu_accel_struct_header, self_ptr));
   }
   return unpack_u64(b, accel);
}

static nir_def *
load_as_bvh_ptr(nir_builder *b, nir_def *header, nir_def *accel)
{
   if (is_as_descriptor(accel)) {
      return nir_load_uav_ir3(
         b, 2, 32, accel,
         nir_vec2(b, nir_imm_int(b, 0),
                  nir_imm_int(b, offsetof(struct tu_accel_struct_header,
                                          bvh_ptr) / 4)),
         .access = static_cast<gl_access_qualifier>(
                      ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
         .align_mul = AS_RECORD_SIZE,
         .align_offset = offsetof(struct tu_accel_struct_header, bvh_ptr));
   }

   nir_def *addr = pack_uvec2(b, header);
   return nir_load_global(b, 2, 32,
                          nir_iadd_imm(b, addr, offsetof(struct tu_accel_struct_header,
                                                         bvh_ptr)),
                          .align_mul = 8);
}

static nir_def *
load_instance(nir_builder *b, const struct tu_rt_state *s,
              nir_def *instance, uint32_t offset, unsigned components)
{
   return load_tlas_record(b, nir_load_var(b, s->tlas_base),
                           iadd_imm(b, instance, 1), offset, components);
}

static nir_def *
load_leaf_geometry_id(nir_builder *b, nir_def *base, nir_def *node)
{
   nir_def *offset =
      nir_iadd_imm(b, nir_imul_imm(b, node, sizeof(struct tu_leaf_node)),
                   offsetof(struct tu_leaf_node, geometry_id));
   nir_def *addr = nir_iadd(b, pack_uvec2(b, base), nir_u2u64(b, offset));
   return nir_load_global(b, 1, 32, addr,
                          .access = static_cast<gl_access_qualifier>(
                                       ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                          .align_mul = sizeof(struct tu_leaf_node),
                          .align_offset = offsetof(struct tu_leaf_node, geometry_id));
}

static nir_def *
load_leaf_addr(nir_builder *b, nir_def *base, nir_def *node)
{
   nir_def *offset = nir_imul_imm(b, node, sizeof(struct tu_leaf_node));
   return nir_iadd(b, pack_uvec2(b, base), nir_u2u64(b, offset));
}

static nir_def *
build_vec3_mat_mult(nir_builder *b, nir_def *vec, nir_def *matrix[],
                    bool translation)
{
   nir_def *result_components[3] = {
      nir_channel(b, matrix[0], 3),
      nir_channel(b, matrix[1], 3),
      nir_channel(b, matrix[2], 3),
   };
   for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 3; ++j) {
         nir_def *v = nir_fmul(b, nir_channels(b, vec, 1 << j),
                               nir_channels(b, matrix[i], 1 << j));
         result_components[i] = (translation || j) ?
            nir_fadd(b, result_components[i], v) : v;
      }
   }
   return nir_vec(b, result_components, 3);
}

static void
load_wto_matrix(nir_builder *b, const struct tu_rt_state *s,
                nir_def *instance, nir_def *rows[3])
{
   for (unsigned i = 0; i < 3; i++)
      rows[i] = load_instance(b, s, instance,
                              offsetof(struct tu_instance_descriptor,
                                       wto_matrix.values) + i * 16, 4);
}


static nir_def *
load_indirect2_field(nir_builder *b, const struct tu_rt_lowering_info *info,
                     uint32_t offset, unsigned components)
{
   nir_def *addr = load_trace_u64(
      b, info, offsetof(struct tu_rt_trace_params, indirect_addr));
   return nir_load_global(b, components, 32, nir_iadd_imm(b, addr, offset),
                          .align_mul = 4);
}

static nir_def *
load_sbt_base(nir_builder *b, const struct tu_rt_lowering_info *info,
              uint32_t push_offset, uint32_t indirect2_offset)
{
   nir_def *flags = load_trace_u32(b, info,
                                   offsetof(struct tu_rt_trace_params, flags));
   nir_variable *result =
      nir_local_variable_create(b->impl, glsl_uint64_t_type(), "sbt_base");
   nir_push_if(b, nir_test_mask(b, flags, 2));
   {
      nir_def *base = indirect2_offset == UINT32_MAX ?
         nir_imm_int64(b, 0) :
         pack_uvec2(b, load_indirect2_field(b, info, indirect2_offset, 2));
      nir_store_var(b, result, base, 1);
   }
   nir_push_else(b, NULL);
   {
      nir_store_var(b, result, load_trace_u64(b, info, push_offset), 1);
   }
   nir_pop_if(b, NULL);
   return nir_load_var(b, result);
}

static nir_def *
load_sbt_stride(nir_builder *b, const struct tu_rt_lowering_info *info,
                uint32_t push_offset, uint32_t indirect2_offset)
{
   nir_def *flags = load_trace_u32(b, info,
                                   offsetof(struct tu_rt_trace_params, flags));
   nir_variable *result =
      nir_local_variable_create(b->impl, glsl_uint_type(), "sbt_stride");
   nir_push_if(b, nir_test_mask(b, flags, 2));
   {
      nir_def *stride = indirect2_offset == UINT32_MAX ?
         nir_imm_int(b, 0) :
         nir_load_global(b, 1, 32,
                         nir_iadd_imm(b,
                            load_trace_u64(b, info,
                               offsetof(struct tu_rt_trace_params, indirect_addr)),
                            indirect2_offset),
                         .align_mul = 4);
      nir_store_var(b, result, stride, 1);
   }
   nir_push_else(b, NULL);
   {
      nir_store_var(b, result, load_trace_u32(b, info, push_offset), 1);
   }
   nir_pop_if(b, NULL);
   return nir_load_var(b, result);
}

/* Load the ray launch dimensions from the direct push constant or from the
 * indirect command.  The persistent raygen loop uses this to derive the total
 * number of rays and to turn a linear invocation index back into a 3D
 * gl_LaunchIDEXT.
 */
static nir_def *
load_launch_size(nir_builder *b, const struct tu_rt_lowering_info *info)
{
   nir_def *indirect = load_trace_u32(b, info,
                                      offsetof(struct tu_rt_trace_params, flags));
   nir_variable *launch_size =
      nir_local_variable_create(b->impl,
                                glsl_vector_type(GLSL_TYPE_UINT, 3),
                                "rt_launch_size");
   nir_push_if(b, nir_test_mask(b, indirect, 3));
   {
      nir_def *addr = load_trace_u64(b, info,
                                     offsetof(struct tu_rt_trace_params,
                                              indirect_addr));
      nir_def *off = nir_bcsel(b, nir_test_mask(b, indirect, 2),
                               nir_imm_int(b, offsetof(VkTraceRaysIndirectCommand2KHR,
                                                       width)),
                               nir_imm_int(b, 0));
      nir_store_var(b, launch_size,
                    nir_load_global(b, 3, 32, nir_iadd(b, addr, nir_u2u64(b, off)),
                                    .align_mul = 4), 7);
   }
   nir_push_else(b, NULL);
   {
      nir_store_var(b, launch_size,
                    load_push(b, info,
                              offsetof(struct tu_rt_trace_params, launch_size), 3), 7);
   }
   nir_pop_if(b, NULL);
   return nir_load_var(b, launch_size);
}

/* Follow a deref chain back to the root variable.  The SPIR-V payload
 * operands of traceRay/executeCallable are always pointers to variables, so
 * this is stable even after the shader has been through some NIR passes.
 */
static nir_variable *
deref_root_var(nir_def *def)
{
   nir_deref_instr *deref = nir_def_as_deref_or_null(def);

   while (deref && deref->deref_type != nir_deref_type_var)
      deref = nir_def_as_deref_or_null(deref->parent.ssa);

   return deref ? deref->var : NULL;
}

static nir_def *
call_data_source(nir_intrinsic_instr *intr,
                 enum tu_rt_call_data_kind *kind)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_trace_ray:
      *kind = TU_RT_CALL_DATA_PAYLOAD;
      return intr->src[10].ssa;
   case nir_intrinsic_execute_callable:
      *kind = TU_RT_CALL_DATA_CALLABLE;
      return intr->src[1].ssa;
   default:
      return NULL;
   }
}

/* Assign every outgoing ray payload/callable variable its own slot.  Incoming
 * variables are redirected through payload_ptr/callable_ptr instead, so that
 * a callee sees exactly the memory of the call site that invoked it.
 */
static void
collect_call_data_slots(nir_shader *shader, struct tu_rt_lower_ctx *ctx)
{
   struct hash_table *seen = _mesa_pointer_hash_table_create(NULL);
   struct util_dynarray payload_vars = UTIL_DYNARRAY_INIT;
   struct util_dynarray callable_vars = UTIL_DYNARRAY_INIT;

   nir_foreach_function (function, shader) {
      if (!function->impl)
         continue;

      nir_foreach_block (block, function->impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            enum tu_rt_call_data_kind kind;
            nir_def *src = call_data_source(intr, &kind);
            if (!src)
               continue;

            nir_variable *var = deref_root_var(src);
            if (!var || _mesa_hash_table_search(seen, var))
               continue;

            /* Incoming variables are forwarded through the caller's payload
             * or callable pointer at run time; they do not own a storage slot.
             */
            if (var->data.mode & nir_var_shader_call_data_in)
               continue;

            _mesa_hash_table_insert(seen, var, NULL);
            util_dynarray_append(kind == TU_RT_CALL_DATA_PAYLOAD ?
                                 &payload_vars : &callable_vars, var);
         }
      }
   }

   /* Size the payload and callable slots independently.  Incoming views must
    * match the type of the outgoing variable at the call site, so the outgoing
    * sizes are sufficient; the minimum keeps a well-defined home for malformed
    * pipelines and prevents the hit attribute slot from overlapping slot 0.
    */
   uint32_t payload_slot_size = TU_RT_MIN_CALL_DATA_SIZE;
   uint32_t callable_slot_size = TU_RT_MIN_CALL_DATA_SIZE;
   util_dynarray_foreach (&payload_vars, nir_variable *, var) {
      unsigned size, alignment;
      glsl_get_natural_size_align_bytes((*var)->type, &size, &alignment);
      payload_slot_size = MAX2(payload_slot_size, align(size, 16));
   }
   util_dynarray_foreach (&callable_vars, nir_variable *, var) {
      unsigned size, alignment;
      glsl_get_natural_size_align_bytes((*var)->type, &size, &alignment);
      callable_slot_size = MAX2(callable_slot_size, align(size, 16));
   }

   uint32_t offset = payload_vars.size ? 0 : payload_slot_size;
   util_dynarray_foreach (&payload_vars, nir_variable *, var) {
      _mesa_hash_table_insert(ctx->var_slots, *var,
                              (void *)(uintptr_t)(offset + 1));
      offset += payload_slot_size;
   }
   util_dynarray_foreach (&callable_vars, nir_variable *, var) {
      _mesa_hash_table_insert(ctx->var_slots, *var,
                              (void *)(uintptr_t)(offset + 1));
      offset += callable_slot_size;
   }

   ctx->hit_attr_base = align(offset, 16);
   ctx->rt_data_size = ctx->hit_attr_base +
                       ctx->levels * TU_RT_HIT_ATTR_SIZE;

   if (payload_vars.size == 1) {
      nir_variable *var =
         *util_dynarray_element(&payload_vars, nir_variable *, 0);
      struct hash_entry *entry = _mesa_hash_table_search(ctx->var_slots, var);
      if (entry)
         ctx->fixed_payload_slot =
            (int32_t)(((uint32_t)(uintptr_t)entry->data - 1) / 4);
   }
   if (callable_vars.size == 1) {
      nir_variable *var =
         *util_dynarray_element(&callable_vars, nir_variable *, 0);
      struct hash_entry *entry = _mesa_hash_table_search(ctx->var_slots, var);
      if (entry)
         ctx->fixed_callable_slot =
            (int32_t)(((uint32_t)(uintptr_t)entry->data - 1) / 4);
   }

   util_dynarray_fini(&payload_vars);
   util_dynarray_fini(&callable_vars);
   _mesa_hash_table_destroy(seen, NULL);
}

static nir_def *
call_data_ptr(nir_builder *b, struct tu_rt_lower_ctx *ctx, nir_def *src,
              enum tu_rt_call_data_kind kind)
{
   nir_variable *var = deref_root_var(src);

   /* An outgoing variable owns a fixed storage slot.  An incoming variable
    * (RayPayloadInKHR/IncomingCallableDataKHR) has no slot of its own: passing
    * it to a nested trace/callable forwards the caller's live data pointer.
    */
   if (var && !(var->data.mode & nir_var_shader_call_data_in)) {
      struct hash_entry *entry = _mesa_hash_table_search(ctx->var_slots, var);
      if (entry) {
         uint32_t slot = (uint32_t)(uintptr_t)entry->data - 1;
         return nir_imm_int(b, slot / 4);
      }
   }

   if (kind == TU_RT_CALL_DATA_PAYLOAD && ctx->fixed_payload_slot >= 0)
      return nir_imm_int(b, ctx->fixed_payload_slot);
   if (kind == TU_RT_CALL_DATA_CALLABLE && ctx->fixed_callable_slot >= 0)
      return nir_imm_int(b, ctx->fixed_callable_slot);

   return load_u32(b, kind == TU_RT_CALL_DATA_PAYLOAD ?
                      ctx->state->payload_ptr : ctx->state->callable_ptr);
}

enum tu_rt_stage_kind {
   TU_RT_GENERAL,
   TU_RT_ANY_HIT,
   TU_RT_CLOSEST_HIT,
   TU_RT_INTERSECTION,
};

static nir_function *
group_stage(const struct tu_ray_tracing_pipeline *pipeline,
            const struct tu_rt_lowering_info *info,
            const struct tu_rt_group *group, enum tu_rt_stage_kind kind,
            uint32_t level, uint32_t call_depth)
{
   uint32_t stage_idx;
   switch (kind) {
   case TU_RT_GENERAL:
      stage_idx = group->general;
      break;
   case TU_RT_ANY_HIT:
      stage_idx = group->any_hit;
      break;
   case TU_RT_CLOSEST_HIT:
      stage_idx = group->closest_hit;
      break;
   case TU_RT_INTERSECTION:
      stage_idx = group->intersection;
      break;
   default:
      return NULL;
   }
   if (stage_idx >= info->stage_count || level >= info->levels ||
       call_depth >= info->callable_levels)
      return NULL;
   return info->funcs[(stage_idx * info->levels + level) *
                         info->callable_levels +
                      call_depth];
}

static void
call_stage(nir_builder *b, const struct tu_rt_state *s, nir_function *func,
           nir_def *record_ptr)
{
   nir_store_var(b, s->shader_record_ptr, record_ptr, 1);
   nir_build_call(b, func, 0, NULL);
}

static bool
pipeline_has_stage(const struct tu_rt_lowering_info *info,
                   enum tu_rt_stage_kind kind)
{
   if (info->levels <= 1)
      return false;

   for (uint32_t i = 0; i < info->pipeline->group_count; i++) {
      if (group_stage(info->pipeline, info, &info->pipeline->groups[i], kind,
                      1, 0))
         return true;
   }
   return false;
}

static bool
pipeline_has_general_stage(const struct tu_rt_lowering_info *info,
                           VkShaderStageFlagBits stage)
{
   for (uint32_t i = 0; i < info->pipeline->group_count; i++) {
      uint32_t general = info->pipeline->groups[i].general;
      if (general < info->stage_count &&
          info->stages[general].stage == stage)
         return true;
   }
   return false;
}

enum tu_rt_dispatch_kind {
   TU_RT_DISPATCH_RAYGEN,
   TU_RT_DISPATCH_MISS,
   TU_RT_DISPATCH_CALLABLE,
   TU_RT_DISPATCH_ANY_HIT,
   TU_RT_DISPATCH_CLOSEST_HIT,
   TU_RT_DISPATCH_INTERSECTION,
};

struct tu_rt_candidate {
   const struct tu_rt_group *group;
   nir_function *func;
};

static int
candidate_compare(const void *va, const void *vb)
{
   const struct tu_rt_candidate *a =
      static_cast<const struct tu_rt_candidate *>(va);
   const struct tu_rt_candidate *b =
      static_cast<const struct tu_rt_candidate *>(vb);
   return (a->group->handle > b->group->handle) -
          (a->group->handle < b->group->handle);
}

static nir_function *
dispatch_group_stage(struct tu_rt_lower_ctx *ctx,
                     const struct tu_rt_group *group,
                     enum tu_rt_dispatch_kind kind,
                     uint32_t level, uint32_t call_depth)
{
   const struct tu_rt_lowering_info *info = ctx->info;
   uint32_t stage_idx;

   switch (kind) {
   case TU_RT_DISPATCH_RAYGEN:
      stage_idx = group->general;
      if (stage_idx < info->stage_count &&
          info->stages[stage_idx].stage != VK_SHADER_STAGE_RAYGEN_BIT_KHR)
         return NULL;
      level = 0;
      call_depth = 0;
      break;
   case TU_RT_DISPATCH_MISS:
      stage_idx = group->general;
      if (stage_idx < info->stage_count &&
          info->stages[stage_idx].stage != VK_SHADER_STAGE_MISS_BIT_KHR)
         return NULL;
      break;
   case TU_RT_DISPATCH_CALLABLE:
      stage_idx = group->general;
      if (stage_idx < info->stage_count &&
          info->stages[stage_idx].stage != VK_SHADER_STAGE_CALLABLE_BIT_KHR)
         return NULL;
      level = 0;
      break;
   case TU_RT_DISPATCH_ANY_HIT:
      stage_idx = group->any_hit;
      break;
   case TU_RT_DISPATCH_CLOSEST_HIT:
      stage_idx = group->closest_hit;
      break;
   case TU_RT_DISPATCH_INTERSECTION:
      stage_idx = group->intersection;
      break;
   default:
      return NULL;
   }

   if (stage_idx >= info->stage_count || level >= info->levels ||
       call_depth >= info->callable_levels)
      return NULL;
   return info->funcs[(stage_idx * info->levels + level) *
                         info->callable_levels +
                      call_depth];
}

static void
collect_candidates(struct tu_rt_lower_ctx *ctx, enum tu_rt_dispatch_kind kind,
                   uint32_t level, uint32_t call_depth,
                   struct util_dynarray *candidates)
{
   const struct tu_ray_tracing_pipeline *pipeline = ctx->info->pipeline;

   util_dynarray_clear(candidates);
   for (uint32_t i = 0; i < pipeline->group_count; i++) {
      const struct tu_rt_group *group = &pipeline->groups[i];
      nir_function *func =
         dispatch_group_stage(ctx, group, kind, level, call_depth);
      if (!func || !func->impl)
         continue;
      struct tu_rt_candidate candidate = { group, func };
      util_dynarray_append(candidates, candidate);
   }
   uint32_t count =
      util_dynarray_num_elements(candidates, struct tu_rt_candidate);
   if (count > 1) {
      qsort(candidates->data, count, sizeof(struct tu_rt_candidate),
            candidate_compare);
   }
}

/* Emit a balanced handle search over the stage candidates.  This replaces the
 * old linear "if (handle == group->handle)" chain, which grew with every
 * shader group in the pipeline.
 */
static void
emit_dispatch_tree(nir_builder *b, const struct tu_rt_state *callee,
                   nir_def *handle, nir_def *record,
                   struct util_dynarray *candidates,
                   uint32_t lo, uint32_t hi)
{
   if (lo >= hi)
      return;

   uint32_t mid = lo + (hi - lo) / 2;
   struct tu_rt_candidate *candidate =
      util_dynarray_element(candidates, struct tu_rt_candidate, mid);

   nir_push_if(b, nir_ult(b, handle,
                          nir_imm_int(b, candidate->group->handle)));
   emit_dispatch_tree(b, callee, handle, record, candidates, lo, mid);
   nir_push_else(b, NULL);
   nir_push_if(b, nir_ieq(b, handle,
                          nir_imm_int(b, candidate->group->handle)));
   {
      nir_store_var(b, callee->shader_record_ptr, record, 1);
      nir_build_call(b, candidate->func, 0, NULL);
   }
   nir_push_else(b, NULL);
   emit_dispatch_tree(b, callee, handle, record, candidates, mid + 1, hi);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
}

static void
copy_hit_context(nir_builder *b, const struct tu_rt_state *dst,
                 const struct tu_rt_state *src);

static void
copy_call_result(nir_builder *b, const struct tu_rt_state *caller,
                 const struct tu_rt_state *callee, bool intersection);

static void
call_group_stage(nir_builder *b, struct tu_rt_lower_ctx *ctx,
                 struct tu_rt_state *caller, enum tu_rt_stage_kind kind,
                 uint32_t level, nir_def *index)
{
   const struct tu_rt_lowering_info *info = ctx->info;
   struct tu_rt_state *callee = &ctx->hit_states[level];

   if (caller != callee)
      copy_hit_context(b, callee, caller);

   ASSERTED enum tu_rt_dispatch_kind dispatch_kind;
   switch (kind) {
   case TU_RT_INTERSECTION:
      dispatch_kind = TU_RT_DISPATCH_INTERSECTION;
      break;
   case TU_RT_ANY_HIT:
      dispatch_kind = TU_RT_DISPATCH_ANY_HIT;
      break;
   case TU_RT_CLOSEST_HIT:
      dispatch_kind = TU_RT_DISPATCH_CLOSEST_HIT;
      break;
   default:
      UNREACHABLE("bad hit-group stage kind");
   }

   nir_def *base = load_sbt_base(
      b, info, offsetof(struct tu_rt_trace_params, hit_sbt),
      offsetof(VkTraceRaysIndirectCommand2KHR, hitShaderBindingTableAddress));
   nir_def *stride = load_sbt_stride(
      b, info, offsetof(struct tu_rt_trace_params, hit_stride),
      offsetof(VkTraceRaysIndirectCommand2KHR, hitShaderBindingTableStride));
   nir_def *byte_offset = nir_u2u64(b, nir_imul(b, index, stride));
   nir_def *entry = nir_iadd(b, base, byte_offset);
   nir_def *handle = nir_load_global(b, 1, 32, entry, .align_mul = 4);
   nir_def *record = nir_iadd_imm(b, entry, TU_RT_HANDLE_SIZE);

   struct util_dynarray candidates = UTIL_DYNARRAY_INIT;
   collect_candidates(ctx, dispatch_kind, level, 0, &candidates);
   emit_dispatch_tree(b, callee, handle, record, &candidates,
                      0, util_dynarray_num_elements(&candidates,
                                                    struct tu_rt_candidate));
   util_dynarray_fini(&candidates);

   if (caller != callee && kind == TU_RT_INTERSECTION)
      copy_call_result(b, caller, callee, true);
}

static void
call_miss(nir_builder *b, struct tu_rt_lower_ctx *ctx,
          struct tu_rt_state *caller, uint32_t level, nir_def *index)
{
   const struct tu_rt_lowering_info *info = ctx->info;
   struct tu_rt_state *callee = &ctx->hit_states[level];

   copy_hit_context(b, callee, caller);

   nir_def *base = load_sbt_base(b, info, offsetof(struct tu_rt_trace_params, miss_sbt),
                    offsetof(VkTraceRaysIndirectCommand2KHR, missShaderBindingTableAddress));
   nir_def *stride = load_sbt_stride(b, info, offsetof(struct tu_rt_trace_params, miss_stride),
                                     offsetof(VkTraceRaysIndirectCommand2KHR, missShaderBindingTableStride));
   nir_def *entry = nir_iadd(b, base, nir_u2u64(b, nir_imul(b, index, stride)));
   nir_def *handle = nir_load_global(b, 1, 32, entry, .align_mul = 4);
   nir_def *record = nir_iadd_imm(b, entry, TU_RT_HANDLE_SIZE);

   struct util_dynarray candidates = UTIL_DYNARRAY_INIT;
   collect_candidates(ctx, TU_RT_DISPATCH_MISS, level, 0, &candidates);
   emit_dispatch_tree(b, callee, handle, record, &candidates,
                      0, util_dynarray_num_elements(&candidates,
                                                    struct tu_rt_candidate));
   util_dynarray_fini(&candidates);
}

static void
call_callable(nir_builder *b, struct tu_rt_lower_ctx *ctx,
              struct tu_rt_state *callee, uint32_t level, uint32_t call_depth,
              nir_def *index)
{
   const struct tu_rt_lowering_info *info = ctx->info;

   nir_def *base = load_sbt_base(b, info, offsetof(struct tu_rt_trace_params, callable_sbt),
                    offsetof(VkTraceRaysIndirectCommand2KHR, callableShaderBindingTableAddress));
   nir_def *stride = load_sbt_stride(b, info, offsetof(struct tu_rt_trace_params, callable_stride),
                                     offsetof(VkTraceRaysIndirectCommand2KHR, callableShaderBindingTableStride));
   nir_def *entry = nir_iadd(b, base, nir_u2u64(b, nir_imul(b, index, stride)));
   nir_def *handle = nir_load_global(b, 1, 32, entry, .align_mul = 4);
   nir_def *record = nir_iadd_imm(b, entry, TU_RT_HANDLE_SIZE);

   struct util_dynarray candidates = UTIL_DYNARRAY_INIT;
   collect_candidates(ctx, TU_RT_DISPATCH_CALLABLE, level, call_depth,
                      &candidates);
   emit_dispatch_tree(b, callee, handle, record, &candidates,
                      0, util_dynarray_num_elements(&candidates,
                                                    struct tu_rt_candidate));
   util_dynarray_fini(&candidates);
}

static void
call_closest_hit(nir_builder *b, struct tu_rt_lower_ctx *ctx,
                 struct tu_rt_state *caller, uint32_t level, nir_def *index)
{
   const struct tu_rt_lowering_info *info = ctx->info;
   struct tu_rt_state *callee = &ctx->hit_states[level];

   if (caller != callee)
      copy_hit_context(b, callee, caller);

   nir_def *base = load_sbt_base(b, info, offsetof(struct tu_rt_trace_params, hit_sbt),
                    offsetof(VkTraceRaysIndirectCommand2KHR, hitShaderBindingTableAddress));
   nir_def *stride = load_sbt_stride(b, info, offsetof(struct tu_rt_trace_params, hit_stride),
                                     offsetof(VkTraceRaysIndirectCommand2KHR, hitShaderBindingTableStride));
   nir_def *entry = nir_iadd(b, base, nir_u2u64(b, nir_imul(b, index, stride)));
   nir_def *handle = nir_load_global(b, 1, 32, entry, .align_mul = 4);
   nir_def *record = nir_iadd_imm(b, entry, TU_RT_HANDLE_SIZE);

   struct util_dynarray candidates = UTIL_DYNARRAY_INIT;
   collect_candidates(ctx, TU_RT_DISPATCH_CLOSEST_HIT, level, 0, &candidates);
   emit_dispatch_tree(b, callee, handle, record, &candidates,
                      0, util_dynarray_num_elements(&candidates,
                                                    struct tu_rt_candidate));
   util_dynarray_fini(&candidates);

   if (caller != callee)
      copy_call_result(b, caller, callee, false);
}

static void
call_any_hit(nir_builder *b, struct tu_rt_lower_ctx *ctx,
             struct tu_rt_state *caller, uint32_t level, nir_def *index)
{
   const struct tu_rt_lowering_info *info = ctx->info;
   struct tu_rt_state *callee = &ctx->hit_states[level];

   if (caller != callee)
      copy_hit_context(b, callee, caller);

   nir_variable *saved_record = NULL;
   if (caller == callee) {
      saved_record =
         nir_local_variable_create(b->impl, glsl_uint64_t_type(),
                                   "saved_anyhit_record");
      nir_store_var(b, saved_record, load_u64(b, callee->shader_record_ptr), 1);
   }

   nir_def *base = load_sbt_base(b, info, offsetof(struct tu_rt_trace_params, hit_sbt),
                    offsetof(VkTraceRaysIndirectCommand2KHR, hitShaderBindingTableAddress));
   nir_def *stride = load_sbt_stride(b, info, offsetof(struct tu_rt_trace_params, hit_stride),
                                     offsetof(VkTraceRaysIndirectCommand2KHR, hitShaderBindingTableStride));
   nir_def *entry = nir_iadd(b, base, nir_u2u64(b, nir_imul(b, index, stride)));
   nir_def *handle = nir_load_global(b, 1, 32, entry, .align_mul = 4);
   nir_def *record = nir_iadd_imm(b, entry, TU_RT_HANDLE_SIZE);

   struct util_dynarray candidates = UTIL_DYNARRAY_INIT;
   collect_candidates(ctx, TU_RT_DISPATCH_ANY_HIT, level, 0, &candidates);
   emit_dispatch_tree(b, callee, handle, record, &candidates,
                      0, util_dynarray_num_elements(&candidates,
                                                    struct tu_rt_candidate));
   util_dynarray_fini(&candidates);

   if (caller == callee)
      nir_store_var(b, callee->shader_record_ptr,
                    nir_load_var(b, saved_record), 1);
   else
      copy_call_result(b, caller, callee, false);
}

static void
copy_hit_context(nir_builder *b, const struct tu_rt_state *dst,
                 const struct tu_rt_state *src)
{
   nir_store_var(b, dst->tlas_base, nir_load_var(b, src->tlas_base), 0x3);

   nir_store_var(b, dst->origin, nir_load_var(b, src->origin), 0x7);
   nir_store_var(b, dst->dir, nir_load_var(b, src->dir), 0x7);
   nir_store_var(b, dst->world_origin, nir_load_var(b, src->world_origin), 0x7);
   nir_store_var(b, dst->world_dir, nir_load_var(b, src->world_dir), 0x7);

   store_u32(b, dst->tmin, load_u32(b, src->tmin));
   store_u32(b, dst->tmax, load_u32(b, src->tmax));
   store_u32(b, dst->ray_flags, load_u32(b, src->ray_flags));
   store_u32(b, dst->api_ray_flags, load_u32(b, src->api_ray_flags));
   store_u32(b, dst->cull_mask, load_u32(b, src->cull_mask));
   store_u32(b, dst->instance_flags, load_u32(b, src->instance_flags));

   store_u32(b, dst->sbt_offset, load_u32(b, src->sbt_offset));
   store_u32(b, dst->sbt_stride, load_u32(b, src->sbt_stride));
   store_u32(b, dst->miss_index, load_u32(b, src->miss_index));

   store_u32(b, dst->instance_id, load_u32(b, src->instance_id));
   store_u32(b, dst->instance_sbt_offset_and_flags,
             load_u32(b, src->instance_sbt_offset_and_flags));

   store_u32(b, dst->primitive_id, load_u32(b, src->primitive_id));
   store_u32(b, dst->geometry_id, load_u32(b, src->geometry_id));
   store_u32(b, dst->hit_kind, load_u32(b, src->hit_kind));
   store_u32(b, dst->sbt_index, load_u32(b, src->sbt_index));
   nir_store_var(b, dst->primitive_bvh_base,
                 nir_load_var(b, src->primitive_bvh_base), 0x3);
   store_u32(b, dst->primitive_node, load_u32(b, src->primitive_node));

   nir_store_var(b, dst->accept, nir_load_var(b, src->accept), 1);
   nir_store_var(b, dst->terminate, nir_load_var(b, src->terminate), 1);
   nir_store_var(b, dst->opaque, nir_load_var(b, src->opaque), 1);
   store_u32(b, dst->report_t, load_u32(b, src->report_t));
   store_u32(b, dst->report_kind, load_u32(b, src->report_kind));
}

static void
copy_call_result(nir_builder *b, const struct tu_rt_state *caller,
                 const struct tu_rt_state *callee, bool intersection)
{
   nir_store_var(b, caller->accept, nir_load_var(b, callee->accept), 1);
   nir_store_var(b, caller->terminate, nir_load_var(b, callee->terminate), 1);
   if (intersection) {
      store_u32(b, caller->report_t, load_u32(b, callee->report_t));
      store_u32(b, caller->report_kind, load_u32(b, callee->report_kind));
   }
}

static nir_def *
ray_flags_with_instance(nir_builder *b, const struct tu_rt_state *s)
{
   nir_def *flags = nir_bitfield_insert(b, load_u32(b, s->ray_flags),
                                       load_u32(b, s->node_mask),
                                       nir_imm_int(b, 0), nir_imm_int(b, 4));
   nir_def *instance_flags = load_u32(b, s->instance_flags);
   return nir_ior(b, flags, instance_flags);
}

static nir_def *
stack_index(nir_builder *b, nir_def *ptr)
{
   nir_def *local_id =
      nir_channel(b, nir_load_local_invocation_id(b), 0);
   /* Slot-major: adjacent invocations write adjacent words for the same
    * depth.  The previous invocation-major layout used a 16-word stride,
    * which maps every lane's same-depth entry to very few LDS banks.
    */
   return nir_iadd(b,
      nir_imul_imm(b, nir_umod_imm(b, ptr, TU_RT_MAX_STACK),
                   TU_RT_WORKGROUP_SIZE),
      local_id);
}

static nir_deref_instr *
stack_entry(nir_builder *b, const struct tu_rt_state *s, nir_def *ptr)
{
   return nir_build_deref_array(b, nir_build_deref_var(b, s->stack),
                                stack_index(b, ptr));
}

static void
push_stack(nir_builder *b, const struct tu_rt_state *s, nir_def *node)
{
   nir_def *ptr = load_u32(b, s->stack_ptr);
   nir_store_deref(b, stack_entry(b, s, ptr), node, 0x1);
   ptr = nir_iadd_imm(b, ptr, 1);
   store_u32(b, s->stack_ptr, ptr);

   /* Remember how far the stack wrapped.  When the pointer later falls to or
    * below this watermark, the wrapped entries are no longer valid and the
    * traversal must continue by walking parent pointers instead.
    */
   nir_def *watermark =
      nir_iadd_imm(b, ptr, -(int)TU_RT_MAX_STACK);
   store_u32(b, s->stack_low_watermark,
             nir_imax(b, load_u32(b, s->stack_low_watermark), watermark));
}

static void
store_hit_attrib_vec2(nir_builder *b, const struct tu_rt_state *s,
                      uint32_t offset, nir_def *v)
{
   nir_deref_instr *data = nir_build_deref_var(b, s->rt_data);
   nir_deref_instr *elem =
      nir_build_deref_array_imm(b, data, offset / 4);
   nir_deref_instr *cast =
      nir_build_deref_cast(b, &elem->def, nir_var_function_temp,
                           glsl_vec2_type(), 4);
   nir_store_deref(b, cast, v, 0x3);
}

static void
handle_triangle(nir_builder *b, struct tu_rt_lower_ctx *ctx,
                struct tu_rt_state *s, nir_def *bvh_base, nir_def *node,
                nir_def *result)
{
   uint32_t callee_level = ctx->level + 1;
   nir_def *flags = nir_channel(b, result, 0);
   nir_def *primitive_id = nir_channel(b, result, 1);
   nir_def *t = nir_channel(b, result, 2);
   nir_def *bary = nir_vec2(b, nir_channel(b, result, 3),
                            nir_channel(b, result, 4));

   nir_def *geometry_id = load_leaf_geometry_id(b, bvh_base, node);
   nir_def *instance_sbt = nir_iand_imm(b,
      load_u32(b, s->instance_sbt_offset_and_flags), 0x00ffffff);
   nir_def *sbt_index =
      nir_iadd(b, nir_iadd(b, load_u32(b, s->sbt_offset), instance_sbt),
               nir_imul(b, load_u32(b, s->sbt_stride), geometry_id));

   store_u32(b, s->primitive_id, primitive_id);
   store_u32(b, s->geometry_id, geometry_id);
   store_u32(b, s->sbt_index, sbt_index);
   store_uvec2(b, s->primitive_bvh_base, bvh_base);
   store_u32(b, s->primitive_node, node);
   store_u32(b, s->hit_kind,
             nir_bcsel(b, nir_test_mask(b, flags, TU_INTERSECTION_BACK_FACE),
                       nir_imm_int(b, 0xff), nir_imm_int(b, 0xfe)));

   nir_def *opaque = nir_inot(b, nir_test_mask(b, flags,
                                               TU_INTERSECTION_TYPE_NONOPAQUE));
   nir_variable *prev_tmax =
      nir_local_variable_create(b->impl, glsl_float_type(), "prev_tmax");
   nir_store_var(b, prev_tmax, load_u32(b, s->tmax), 1);
   nir_store_var(b, s->accept, nir_imm_true(b), 1);
   nir_store_var(b, s->opaque, opaque, 1);
   store_hit_attrib_vec2(b, s, s->hit_attr_offset, bary);

   /* Hit attributes (the barycentrics above) and gl_HitTEXT are available to
    * the any-hit shader.  Use the candidate t for both, but restore it if the
    * shader ignores the intersection.
    */
   store_u32(b, s->tmax, t);

   if (ctx->has_any_hit) {
      nir_push_if(b, nir_inot(b, opaque));
      call_any_hit(b, ctx, s, callee_level, sbt_index);
      nir_pop_if(b, NULL);
   }

   nir_push_if(b, load_u32(b, s->accept));
   {
      nir_store_var(b, s->hit, nir_imm_true(b), 1);

      if (ctx->has_closest_hit) {
         nir_def *skip_closest =
            nir_test_mask(b, load_u32(b, s->ray_flags),
                          SpvRayFlagsSkipClosestHitShaderKHRMask << 4);
         nir_push_if(b, nir_inot(b, skip_closest));
         call_closest_hit(b, ctx, s, callee_level, sbt_index);
         nir_pop_if(b, NULL);
      }

      nir_def *terminate =
         nir_ior(b, load_u32(b, s->terminate),
                 nir_test_mask(b, load_u32(b, s->ray_flags),
                               SpvRayFlagsTerminateOnFirstHitKHRMask << 4));
      nir_push_if(b, terminate);
      nir_jump(b, nir_jump_break);
      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   {
      store_u32(b, s->tmax, load_u32(b, prev_tmax));
   }
   nir_pop_if(b, NULL);
}

static void
handle_aabb(nir_builder *b, struct tu_rt_lower_ctx *ctx,
            struct tu_rt_state *s, nir_def *bvh_base, nir_def *node,
            nir_def *result)
{
   uint32_t callee_level = ctx->level + 1;
   nir_def *flags = nir_channel(b, result, 0);
   nir_def *primitive_id = nir_channel(b, result, 1);

   nir_def *geometry_id = load_leaf_geometry_id(b, bvh_base, node);
   nir_def *instance_sbt = nir_iand_imm(b,
      load_u32(b, s->instance_sbt_offset_and_flags), 0x00ffffff);
   nir_def *sbt_index =
      nir_iadd(b, nir_iadd(b, load_u32(b, s->sbt_offset), instance_sbt),
               nir_imul(b, load_u32(b, s->sbt_stride), geometry_id));

   store_u32(b, s->primitive_id, primitive_id);
   store_u32(b, s->geometry_id, geometry_id);
   store_u32(b, s->sbt_index, sbt_index);
   store_uvec2(b, s->primitive_bvh_base, bvh_base);
   store_u32(b, s->primitive_node, node);
   nir_store_var(b, s->opaque,
                   nir_inot(b, nir_test_mask(b, flags, TU_INTERSECTION_TYPE_NONOPAQUE)), 1);

   nir_variable *prev_tmax =
      nir_local_variable_create(b->impl, glsl_float_type(), "prev_tmax");
   nir_store_var(b, prev_tmax, load_u32(b, s->tmax), 1);

   nir_store_var(b, s->accept, nir_imm_false(b), 1);
   store_u32(b, s->report_t, load_u32(b, s->tmax));
   store_u32(b, s->report_kind, nir_imm_int(b, 0));

   if (ctx->has_intersection)
      call_group_stage(b, ctx, s, TU_RT_INTERSECTION, callee_level, sbt_index);

   nir_push_if(b, load_u32(b, s->accept));
   {
      store_u32(b, s->tmax, load_u32(b, s->report_t));
      store_u32(b, s->hit_kind, load_u32(b, s->report_kind));

      /* Procedural AABB hits go through the any-hit stage just like
       * triangles.  The intersection shader has already committed the
       * candidate via reportIntersectionEXT.
       */
      if (ctx->has_any_hit) {
         nir_push_if(b, nir_inot(b, load_u32(b, s->opaque)));
         call_any_hit(b, ctx, s, callee_level, sbt_index);
         nir_pop_if(b, NULL);
      }

      nir_push_if(b, load_u32(b, s->accept));
      {
         nir_store_var(b, s->hit, nir_imm_true(b), 1);

         if (ctx->has_closest_hit) {
            nir_def *skip_closest =
               nir_test_mask(b, load_u32(b, s->ray_flags),
                             SpvRayFlagsSkipClosestHitShaderKHRMask << 4);
            nir_push_if(b, nir_inot(b, skip_closest));
            call_closest_hit(b, ctx, s, callee_level, sbt_index);
            nir_pop_if(b, NULL);
         }

         nir_def *terminate =
            nir_ior(b, load_u32(b, s->terminate),
                    nir_test_mask(b, load_u32(b, s->ray_flags),
                                  SpvRayFlagsTerminateOnFirstHitKHRMask << 4));
         nir_push_if(b, terminate);
         nir_jump(b, nir_jump_break);
         nir_pop_if(b, NULL);
      }
      nir_push_else(b, NULL);
      {
         store_u32(b, s->tmax, load_u32(b, prev_tmax));
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
}

static void
build_ray_traversal(nir_builder *b, struct tu_rt_lower_ctx *ctx,
                    struct tu_rt_state *s,
                    bool skip_triangles, bool skip_aabbs)
{

   store_u32(b, s->current_node, nir_imm_int(b, 0));
   store_u32(b, s->previous_node, nir_imm_int(b, TU_RT_INVALID_NODE));
   store_u32(b, s->stack_ptr, nir_imm_int(b, 0));
   store_u32(b, s->top_stack, nir_imm_int(b, UINT32_MAX));
   store_u32(b, s->stack_low_watermark, nir_imm_int(b, 0));
   store_u32(b, s->instance_top_node, nir_imm_int(b, TU_RT_INVALID_NODE));
   store_u32(b, s->instance_bottom_node,
             nir_imm_int(b, TU_RT_NO_INSTANCE_ROOT));
   store_u32(b, s->node_mask, nir_imm_int(b, 0xf));
   nir_store_var(b, s->hit, nir_imm_false(b), 1);
   nir_store_var(b, s->accept, nir_imm_false(b), 1);
   nir_store_var(b, s->terminate, nir_imm_false(b), 1);
   nir_store_var(b, s->instance_flags, nir_imm_int(b, 0), 1);

   /* Seed the hit-related state as well.  A miss shader has no instance or
    * primitive, but copying the whole ray context into the callee state must
    * not read values that were never written by an earlier trace.
    */
   nir_def *zero2 = nir_vec2(b, nir_imm_int(b, 0), nir_imm_int(b, 0));
   store_u32(b, s->instance_id, nir_imm_int(b, 0));
   store_u32(b, s->instance_sbt_offset_and_flags, nir_imm_int(b, 0));
   store_u32(b, s->primitive_id, nir_imm_int(b, 0));
   store_u32(b, s->geometry_id, nir_imm_int(b, 0));
   store_u32(b, s->hit_kind, nir_imm_int(b, 0));
   store_u32(b, s->sbt_index, nir_imm_int(b, 0));
   store_uvec2(b, s->primitive_bvh_base, zero2);
   store_u32(b, s->primitive_node, nir_imm_int(b, 0));
   nir_store_var(b, s->opaque, nir_imm_false(b), 1);
   store_u32(b, s->report_t, load_u32(b, s->tmax));
   store_u32(b, s->report_kind, nir_imm_int(b, 0));

   nir_loop *loop = nir_push_loop(b);
   {
      nir_push_if(b, nir_ieq_imm(b, load_u32(b, s->current_node),
                                 TU_RT_INVALID_NODE));
      {
         /* Early-exit when both the node queue and the stack are empty.
          * This is the only successful termination path.
          */
         nir_push_if(b, nir_ilt_imm(b, load_u32(b, s->stack_ptr), 1));
         {
            nir_jump(b, nir_jump_break);
         }
         nir_pop_if(b, NULL);

         /* An instance's BLAS has been fully traversed when the stack has
          * unwound back to the level it had when the instance was entered,
          * or when the previous node was the BLAS root.  Restore the world
          * ray and the TLAS before looking for the next TLAS sibling.
          */
         nir_def *stack_instance_exit =
            nir_iand(b,
               nir_ine_imm(b, load_u32(b, s->top_stack), UINT32_MAX),
               nir_ige(b, load_u32(b, s->top_stack),
                          load_u32(b, s->stack_ptr)));
         nir_def *root_instance_exit =
            nir_ieq(b, load_u32(b, s->previous_node),
                       load_u32(b, s->instance_bottom_node));
         nir_push_if(b, nir_ior(b, stack_instance_exit, root_instance_exit));
         {
            store_u32(b, s->top_stack, nir_imm_int(b, UINT32_MAX));
            store_u32(b, s->previous_node,
                      load_u32(b, s->instance_top_node));
            store_u32(b, s->instance_bottom_node,
                      nir_imm_int(b, TU_RT_NO_INSTANCE_ROOT));
            store_uvec2(b, s->bvh_base, load_vec(b, s->root_bvh_base));
            nir_store_var(b, s->origin, load_vec(b, s->world_origin), 7);
            nir_store_var(b, s->dir, load_vec(b, s->world_dir), 7);
            store_u32(b, s->instance_flags, nir_imm_int(b, 0));
            store_u32(b, s->node_mask, nir_imm_int(b, 0xf));
         }
         nir_pop_if(b, NULL);

         /* If the bounded stack wrapped, the entries at the top are gone.
          * Re-intersect the parent of the last processed node and let the
          * hardware tell us which sibling to visit next.  This is the
          * RADV-style stackless overflow path; it trades a little bandwidth
          * for never silently dropping geometry.
          */
         nir_push_if(b, nir_ige(b, load_u32(b, s->stack_low_watermark),
                                   load_u32(b, s->stack_ptr)));
         {
            nir_def *prev = load_u32(b, s->previous_node);
            nir_def *parent = tu_ray_fetch_parent_node(
               b, load_vec(b, s->bvh_base), prev);
            nir_push_if(b, nir_ieq_imm(b, parent, TU_RT_INVALID_NODE));
            {
               nir_jump(b, nir_jump_break);
            }
            nir_pop_if(b, NULL);
            store_u32(b, s->current_node, parent);
            store_u32(b, s->node_mask, nir_imm_int(b, 0xf));
         }
         nir_push_else(b, NULL);
         {
            nir_def *ptr =
               nir_iadd_imm(b, load_u32(b, s->stack_ptr), -1);
            store_u32(b, s->stack_ptr, ptr);
            store_u32(b, s->current_node,
                      nir_load_deref(b, stack_entry(b, s, ptr)));
            store_u32(b, s->previous_node, nir_imm_int(b, TU_RT_INVALID_NODE));
            store_u32(b, s->node_mask, nir_imm_int(b, 0xf));
         }
         nir_pop_if(b, NULL);
      }
      nir_push_else(b, NULL);
      {
         /* A node is about to be visited.  It has no parent-backtracking
          * predecessor yet.
          */
         store_u32(b, s->previous_node, nir_imm_int(b, TU_RT_INVALID_NODE));
      }
      nir_pop_if(b, NULL);

      nir_def *node = load_u32(b, s->current_node);
      nir_def *bvh_base = load_vec(b, s->bvh_base);
      store_u32(b, s->current_node, nir_imm_int(b, TU_RT_INVALID_NODE));
      /* Remember which node we are leaving from when the next iteration has
       * to backtrack through parent pointers.
       */
      store_u32(b, s->previous_node, node);

      nir_def *ray = nir_vec8(b,
         nir_channel(b, load_vec(b, s->origin), 0),
         nir_channel(b, load_vec(b, s->origin), 1),
         nir_channel(b, load_vec(b, s->origin), 2),
         load_u32(b, s->tmin),
         nir_channel(b, load_vec(b, s->dir), 0),
         nir_channel(b, load_vec(b, s->dir), 1),
         nir_channel(b, load_vec(b, s->dir), 2),
         load_u32(b, s->tmax));

      nir_def *result =
         nir_ray_intersection_ir3(b, 32, bvh_base, node, ray,
                                  ray_flags_with_instance(b, s));

      nir_def *out_flags = nir_channel(b, result, 0);
      nir_def *type = nir_ubitfield_extract_imm(b, out_flags, 8, 4);
      nir_def *count = nir_ubitfield_extract_imm(b, out_flags, 4, 4);
      nir_def *is_leaf =
         nir_test_mask(b, type, TU_INTERSECTION_TYPE_LEAF >> 8);
      nir_def *processed_mask = nir_iand_imm(b, out_flags, 0xf);

      nir_push_if(b, nir_iand(b, is_leaf, nir_ine_imm(b, count, 0)));
      {
         nir_push_if(b, nir_test_mask(b, type,
                                      TU_INTERSECTION_TYPE_TLAS >> 8));
         {
            nir_def *instance = nir_channel(b, result, 1);
            store_u32(b, s->instance_id, instance);
            store_u32(b, s->instance_sbt_offset_and_flags,
                      load_instance(b, s, instance,
                                    offsetof(struct tu_instance_descriptor,
                                             sbt_offset_and_flags), 1));
            store_u32(b, s->instance_flags,
                      nir_iand_imm(b, load_u32(b, s->instance_sbt_offset_and_flags),
                                   0xff000000));
            nir_def *blas = load_instance(b, s, instance,
                                          offsetof(struct tu_instance_descriptor,
                                                   bvh_ptr), 2);

            store_u32(b, s->top_stack, load_u32(b, s->stack_ptr));
            store_u32(b, s->instance_top_node, node);
            store_u32(b, s->instance_bottom_node, nir_imm_int(b, 0));
            store_uvec2(b, s->bvh_base, blas);

            nir_def *wto[3];
            load_wto_matrix(b, s, instance, wto);
            nir_store_var(b, s->origin,
                          build_vec3_mat_mult(b, load_vec(b, s->world_origin),
                                              wto, true), 7);
            nir_store_var(b, s->dir,
                          build_vec3_mat_mult(b, load_vec(b, s->world_dir),
                                              wto, false), 7);
            store_u32(b, s->current_node, nir_imm_int(b, 0));
            store_u32(b, s->node_mask, nir_imm_int(b, 0xf));
         }
         nir_push_else(b, NULL);
         {
            /* Procedural AABB leaves are only meaningful if the pipeline has
             * an intersection shader; without one, no reportIntersectionEXT
             * can ever accept the candidate.  Skip the entire AABB path for
             * triangle-only pipelines.  The triangle path is likewise
             * removed when the pipeline was created with SKIP_TRIANGLES.
             */
            const bool handle_aabbs = ctx->has_intersection && !skip_aabbs;
            const bool handle_triangles = !skip_triangles;

            nir_push_if(b, nir_test_mask(b, type,
                                         TU_INTERSECTION_TYPE_AABB >> 8));
            {
               if (handle_aabbs) {
                  nir_def *ray_skip_aabbs =
                     nir_test_mask(b, load_u32(b, s->ray_flags),
                                   SpvRayFlagsSkipAABBsKHRMask << 4);
                  nir_push_if(b, nir_inot(b, ray_skip_aabbs));
                  handle_aabb(b, ctx, s, bvh_base, node, result);
                  nir_pop_if(b, NULL);
               }
            }
            nir_push_else(b, NULL);
            {
               if (handle_triangles) {
                  nir_def *ray_skip_triangles =
                     nir_test_mask(b, load_u32(b, s->ray_flags),
                                   SpvRayFlagsSkipTrianglesKHRMask << 4);
                  nir_push_if(b, nir_inot(b, ray_skip_triangles));
                  handle_triangle(b, ctx, s, bvh_base, node, result);
                  nir_pop_if(b, NULL);
               }
            }
            nir_pop_if(b, NULL);
         }
         nir_pop_if(b, NULL);
      }
      nir_push_else(b, NULL);
      {
         nir_push_if(b, nir_iand(b, nir_inot(b, is_leaf),
                                 nir_ine_imm(b, count, 0)));
         {
            nir_def *id = nir_channel(b, result, 1);
            nir_def *children = nir_channel(b, result, 3);
            nir_def *closest_pos = nir_iand_imm(b, children, 0x1f);
            nir_def *closest =
               nir_iand_imm(b, nir_ishr(b, children, closest_pos), 7);
            store_u32(b, s->current_node, nir_iadd(b, id, closest));
            store_u32(b, s->node_mask, nir_imm_int(b, 0xf));

            /* Push the remaining children, farthest first, so the next
             * closest sibling is on top of the stack.
             */
            for (unsigned i = 0; i < 7; i++) {
               nir_push_if(b, nir_ult(b, nir_imm_int(b, i + 1), count));
               {
                  nir_def *pos = nir_imm_int(b, 8 + i * 3);
                  nir_def *child =
                     nir_iand_imm(b, nir_ishr(b, children, pos), 7);
                  push_stack(b, s, nir_iadd(b, id, child));
               }
               nir_pop_if(b, NULL);
            }
         }
         nir_pop_if(b, NULL);
      }
      nir_pop_if(b, NULL);

      /* A compressed triangle/AABB leaf (or a TLAS leaf that missed) is
       * re-tested in place with the narrowed mask.  A TLAS hit cannot do
       * this because it changes bvh_base and the ray space; the BVH encoder
       * emits one instance per TLAS leaf, so there are no further instances
       * in such a node.
       */
      nir_def *tlas_hit = nir_iand(b,
         nir_test_mask(b, type, TU_INTERSECTION_TYPE_TLAS >> 8),
         nir_ine_imm(b, count, 0));
      nir_def *retest =
         nir_iand(b, is_leaf, nir_ine_imm(b, processed_mask, 0));
      retest = nir_iand(b, retest, nir_inot(b, tlas_hit));
      store_u32(b, s->current_node,
                nir_bcsel(b, retest, node, load_u32(b, s->current_node)));
      store_u32(b, s->node_mask,
                nir_bcsel(b, retest, processed_mask, load_u32(b, s->node_mask)));
   }
   nir_pop_loop(b, loop);
}

static void
lower_trace_ray(nir_builder *b, nir_intrinsic_instr *intr,
                struct tu_rt_lower_ctx *ctx)
{
   struct tu_rt_state *s = &ctx->traversal_states[ctx->level];
   const struct tu_rt_lowering_info *info = ctx->info;
   const struct tu_ray_tracing_pipeline *pipeline = info->pipeline;
   uint32_t callee_level = ctx->level + 1;
   struct tu_rt_state *callee = &ctx->hit_states[callee_level];

   nir_def *accel = intr->src[0].ssa;
   nir_def *flags = intr->src[1].ssa;
   nir_def *cull_mask = intr->src[2].ssa;
   nir_def *sbt_offset = intr->src[3].ssa;
   nir_def *sbt_stride = intr->src[4].ssa;
   nir_def *miss_index = intr->src[5].ssa;
   nir_def *origin = intr->src[6].ssa;
   nir_def *tmin = intr->src[7].ssa;
   nir_def *dir = intr->src[8].ssa;
   nir_def *tmax = intr->src[9].ssa;

   store_u32(b, callee->payload_ptr,
             call_data_ptr(b, ctx, intr->src[10].ssa, TU_RT_CALL_DATA_PAYLOAD));

   nir_def *tlas_header = load_as_header(b, accel);
   nir_def *tlas_bvh = load_as_bvh_ptr(b, tlas_header, accel);
   store_uvec2(b, s->tlas_base, tlas_header);
   store_uvec2(b, s->root_bvh_base, tlas_bvh);
   store_uvec2(b, s->bvh_base, tlas_bvh);
   nir_store_var(b, s->world_origin, origin, 7);
   nir_store_var(b, s->world_dir, dir, 7);
   nir_store_var(b, s->origin, origin, 7);
   nir_store_var(b, s->dir, dir, 7);
   store_u32(b, s->tmin, tmin);
   store_u32(b, s->tmax, tmax);
   store_u32(b, s->api_ray_flags, flags);
   store_u32(b, s->ray_flags,
             nir_ior(b, nir_ishl_imm(b, flags, 4),
                     nir_ior(b, nir_ishl_imm(b, cull_mask, 16),
                             nir_imm_int(b, 0xf))));
   store_u32(b, s->cull_mask, cull_mask);
   store_u32(b, s->sbt_offset, sbt_offset);
   store_u32(b, s->sbt_stride, sbt_stride);
   store_u32(b, s->miss_index, miss_index);

   bool skip_triangles =
      pipeline->create_flags &
      VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_TRIANGLES_BIT_KHR;
   bool skip_aabbs =
      pipeline->create_flags &
      VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_AABBS_BIT_KHR;

   build_ray_traversal(b, ctx, s, skip_triangles, skip_aabbs);

   if (ctx->has_miss) {
      nir_push_if(b, nir_inot(b, load_u32(b, s->hit)));
      call_miss(b, ctx, s, callee_level, miss_index);
      nir_pop_if(b, NULL);
   }
}

static void
lower_execute_callable(nir_builder *b, nir_intrinsic_instr *intr,
                       struct tu_rt_lower_ctx *ctx)
{
   /* executeCallableEXT does not increase the traceRayEXT recursion depth and
    * callable shaders have no ray context.  Each nesting depth owns a separate
    * state, so a nested callable cannot clobber its caller's shader record or
    * incoming callable data; no save/restore is needed.
    */
   uint32_t call_depth =
      ctx->in_callable_stage ? ctx->callable_depth + 1 : 0;
   struct tu_rt_state *callee = &ctx->callable_states[call_depth];

   store_u32(b, callee->callable_ptr,
             call_data_ptr(b, ctx, intr->src[1].ssa, TU_RT_CALL_DATA_CALLABLE));

   if (ctx->has_callable)
      call_callable(b, ctx, callee, 0, call_depth, intr->src[0].ssa);
}

/* Emit an early return from a ray tracing stage.  Everything following the
 * ignore/terminate instruction is unreachable, including the block terminator
 * that NIR generated for the original control flow.  It must be removed so
 * that the new return is the last instruction of the block; otherwise
 * nir_lower_returns() will not see it and inlining will leave a dangling
 * jump in the middle of the block.
 */
static void
lower_stage_return(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_block *block = intr->instr.block;
   nir_instr *instr = nir_instr_next(&intr->instr);
   while (instr) {
      nir_instr *next = nir_instr_next(instr);
      nir_instr_remove(instr);
      instr = next;
   }
   nir_instr_remove(&intr->instr);

   if (nir_block_last_instr(block))
      b->cursor = nir_after_instr(nir_block_last_instr(block));
   else
      b->cursor = nir_before_block(block);
}

static void
lower_report(nir_builder *b, nir_intrinsic_instr *intr,
             struct tu_rt_lower_ctx *ctx)
{
   struct tu_rt_state *s = ctx->state;
   nir_def *t = intr->src[0].ssa;
   nir_def *hit_kind = intr->src[1].ssa;

   nir_def *prev_accept = load_u32(b, s->accept);
   nir_def *prev_tmax = load_u32(b, s->tmax);
   nir_def *prev_kind = load_u32(b, s->hit_kind);

   nir_variable *commit =
      nir_local_variable_create(b->impl, glsl_bool_type(), "commit");
   nir_store_var(b, commit, nir_imm_false(b), 1);

   nir_def *in_range =
      nir_iand(b, nir_fge(b, t, load_u32(b, s->tmin)),
                  nir_fge(b, load_u32(b, s->tmax), t));
   nir_def *terminated =
      nir_iand(b, load_u32(b, s->terminate), load_u32(b, s->accept));

   nir_push_if(b, nir_iand(b, in_range, nir_inot(b, terminated)));
   {
      nir_store_var(b, s->accept, nir_imm_true(b), 1);
      store_u32(b, s->tmax, t);
      store_u32(b, s->report_t, t);
      store_u32(b, s->hit_kind, hit_kind);
      store_u32(b, s->report_kind, hit_kind);

      if (ctx->has_any_hit) {
         nir_push_if(b, nir_inot(b, load_u32(b, s->opaque)));
         call_any_hit(b, ctx, s, ctx->level, load_u32(b, s->sbt_index));
         nir_pop_if(b, NULL);
      }

      nir_push_if(b, load_u32(b, s->accept));
      nir_store_var(b, commit, nir_imm_true(b), 1);
      nir_push_else(b, NULL);
      {
         nir_store_var(b, s->accept, prev_accept, 1);
         store_u32(b, s->tmax, prev_tmax);
         store_u32(b, s->hit_kind, prev_kind);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);

   nir_def_replace(&intr->def, nir_load_var(b, commit));
}

static void
set_current_state(struct tu_rt_lower_ctx *ctx, nir_function_impl *impl)
{
   struct hash_entry *entry =
      _mesa_hash_table_search(ctx->func_to_stage, impl);
   if (entry) {
      uint32_t value = (uint32_t)(uintptr_t)entry->data;
      uint32_t stage_idx = (value & 0xffff) - 1;
      ctx->level = (value >> 16) & 0xff;
      ctx->callable_depth = (value >> 24) & 0xff;
      ctx->in_callable_stage =
         stage_idx < ctx->info->stage_count &&
         ctx->info->stages[stage_idx].stage == VK_SHADER_STAGE_CALLABLE_BIT_KHR;
      ctx->state = ctx->in_callable_stage ?
         &ctx->callable_states[ctx->callable_depth] :
         &ctx->hit_states[ctx->level];
   } else {
      ctx->level = 0;
      ctx->callable_depth = 0;
      ctx->in_callable_stage = false;
      ctx->state = &ctx->hit_states[0];
   }
}

static bool
lower_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct tu_rt_lower_ctx *ctx = static_cast<struct tu_rt_lower_ctx *>(data);
   set_current_state(ctx, b->impl);
   struct tu_rt_state *s = ctx->state;

   if (instr->type == nir_instr_type_jump) {
      return false;
   }
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   nir_def *def = NULL;

   b->cursor = nir_before_instr(instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_trace_ray:
      /* SPIR-V only allows traceRayEXT in ray generation, closest-hit and
       * miss shaders.  A callable shader cannot start a new ray.
       */
      if (ctx->in_callable_stage) {
         nir_instr_remove(instr);
         return true;
      }
      /* A traceRayEXT call executes at ray depth level + 1.  Calls that would
       * exceed the pipeline's limit must never be executed; keep the pipeline
       * creatable but drop the unreachable recursive code.
       */
      if (ctx->level + 1 >= ctx->levels) {
         nir_instr_remove(instr);
         return true;
      }
      lower_trace_ray(b, intr, ctx);
      nir_instr_remove(instr);
      return true;
   case nir_intrinsic_execute_callable: {
      /* Callable calls do not count towards maxPipelineRayRecursionDepth.
       * The monolithic pipeline bounds them by the number of cloned callable
       * frames instead.  A call from a non-callable stage starts a new
       * callable chain at depth 0; nested calls increment the depth.
       */
      uint32_t call_depth =
         ctx->in_callable_stage ? ctx->callable_depth + 1 : 0;
      if (call_depth >= ctx->callable_levels) {
         nir_instr_remove(instr);
         return true;
      }
      lower_execute_callable(b, intr, ctx);
      nir_instr_remove(instr);
      return true;
   }
   case nir_intrinsic_report_ray_intersection:
      lower_report(b, intr, ctx);
      return true;
   case nir_intrinsic_ignore_ray_intersection:
      lower_stage_return(b, intr);
      nir_store_var(b, s->accept, nir_imm_false(b), 1);
      nir_jump(b, nir_jump_return);
      return true;
   case nir_intrinsic_terminate_ray:
      lower_stage_return(b, intr);
      nir_store_var(b, s->accept, nir_imm_true(b), 1);
      nir_store_var(b, s->terminate, nir_imm_true(b), 1);
      nir_jump(b, nir_jump_return);
      return true;
   case nir_intrinsic_accept_ray_intersection:
      nir_store_var(b, s->accept, nir_imm_true(b), 1);
      nir_instr_remove(instr);
      return true;
   case nir_intrinsic_load_ray_launch_id:
      def = load_vec(b, ctx->info->launch_id);
      break;
   case nir_intrinsic_load_ray_launch_size:
      def = load_launch_size(b, ctx->info);
      break;
   case nir_intrinsic_load_shader_record_ptr:
      def = load_u64(b, s->shader_record_ptr);
      break;
   case nir_intrinsic_load_ray_t_min:
      def = load_u32(b, s->tmin);
      break;
   case nir_intrinsic_load_ray_t_max:
      def = load_u32(b, s->tmax);
      break;
   case nir_intrinsic_load_ray_world_origin:
      def = load_vec(b, s->world_origin);
      break;
   case nir_intrinsic_load_ray_world_direction:
      def = load_vec(b, s->world_dir);
      break;
   case nir_intrinsic_load_ray_flags:
      def = load_u32(b, s->api_ray_flags);
      break;
   case nir_intrinsic_load_cull_mask:
      def = load_u32(b, s->cull_mask);
      break;
   case nir_intrinsic_load_instance_id:
      def = load_u32(b, s->instance_id);
      break;
   case nir_intrinsic_load_primitive_id:
      def = load_u32(b, s->primitive_id);
      break;
   case nir_intrinsic_load_ray_geometry_index:
      def = load_u32(b, s->geometry_id);
      break;
   case nir_intrinsic_load_ray_hit_kind:
      def = load_u32(b, s->hit_kind);
      break;
   case nir_intrinsic_load_ray_instance_custom_index:
      def = load_instance(b, s, load_u32(b, s->instance_id),
                          offsetof(struct tu_instance_descriptor,
                                   custom_instance_index), 1);
      break;
   case nir_intrinsic_load_ray_world_to_object: {
      unsigned column = nir_intrinsic_column(intr);
      nir_def *rows[3];
      load_wto_matrix(b, s, load_u32(b, s->instance_id), rows);
      def = nir_vec3(b, nir_channel(b, rows[0], column),
                     nir_channel(b, rows[1], column),
                     nir_channel(b, rows[2], column));
      break;
   }
   case nir_intrinsic_load_ray_object_to_world: {
      unsigned column = nir_intrinsic_column(intr);
      nir_def *rows[3];
      for (unsigned i = 0; i < 3; i++)
         rows[i] = load_instance(b, s, load_u32(b, s->instance_id),
                                 offsetof(struct tu_instance_descriptor,
                                          otw_matrix.values) + i * 16, 4);
      def = nir_vec3(b, nir_channel(b, rows[0], column),
                     nir_channel(b, rows[1], column),
                     nir_channel(b, rows[2], column));
      break;
   }
   case nir_intrinsic_load_ray_object_origin: {
      nir_def *rows[3];
      load_wto_matrix(b, s, load_u32(b, s->instance_id), rows);
      def = build_vec3_mat_mult(b, load_vec(b, s->world_origin), rows, true);
      break;
   }
   case nir_intrinsic_load_ray_object_direction: {
      nir_def *rows[3];
      load_wto_matrix(b, s, load_u32(b, s->instance_id), rows);
      def = build_vec3_mat_mult(b, load_vec(b, s->world_dir), rows, false);
      break;
   }
   case nir_intrinsic_load_ray_triangle_vertex_positions: {
      unsigned column = nir_intrinsic_column(intr);
      nir_def *addr = load_leaf_addr(b, load_vec(b, s->primitive_bvh_base),
                                     load_u32(b, s->primitive_node));
      uint32_t offset = offsetof(struct tu_leaf_node, coords) +
                        column * 3 * sizeof(float);
      def = nir_load_global(b, 3, 32, nir_iadd_imm(b, addr, offset),
                            .access = static_cast<gl_access_qualifier>(
                                         ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                            .align_mul = sizeof(struct tu_leaf_node),
                            .align_offset = offset);
      break;
   }
   default:
      return false;
   }

   if (def)
      nir_def_rewrite_uses(&intr->def, def);
   nir_instr_remove(instr);
   return true;
}

/* ShaderRecordBufferKHR is not a real variable: SPIR-V lowers it to a cast
 * on top of load_shader_record_ptr.  Turn those load/store_deref operations
 * into plain global memory accesses, with the record pointer as the base.
 */
static bool
lower_shader_record_io(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_load_deref &&
       intr->intrinsic != nir_intrinsic_store_deref)
      return false;

   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   if (!deref || !nir_deref_mode_is(deref, nir_var_mem_constant))
      return false;

   nir_deref_path path;
   nir_deref_path_init(&path, deref, NULL);
   nir_deref_instr *root = path.path[0];
   bool is_record =
      root->deref_type == nir_deref_type_cast &&
      nir_def_as_deref_or_null(root->parent.ssa) == NULL;
   if (!is_record) {
      nir_deref_path_finish(&path);
      return false;
   }

   b->cursor = nir_before_instr(instr);
   nir_def *offset = nir_build_deref_offset(
      b, deref, glsl_get_natural_size_align_bytes);
   nir_def *addr = nir_iadd(b, root->parent.ssa, nir_u2u64(b, offset));

   if (intr->intrinsic == nir_intrinsic_load_deref) {
      nir_def *load = nir_load_global(b, intr->def.num_components,
                                      intr->def.bit_size, addr,
                                      .align_mul = 1);
      nir_def_rewrite_uses(&intr->def, load);
   } else {
      nir_store_global(b, intr->src[1].ssa, addr,
                       nir_intrinsic_write_mask(intr));
   }

   nir_deref_path_finish(&path);
   nir_instr_remove(instr);
   return true;
}

static bool
lower_derefs(nir_shader *shader, struct tu_rt_lower_ctx *ctx)
{
   bool progress = false;

   nir_foreach_function (function, shader) {
      if (!function->impl)
         continue;

      set_current_state(ctx, function->impl);
      struct tu_rt_state *s = ctx->state;
      bool callable_stage = ctx->in_callable_stage;

      nir_builder b = nir_builder_create(function->impl);
      nir_foreach_block (block, function->impl) {
         nir_foreach_instr_safe (instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;

            nir_deref_instr *deref = nir_instr_as_deref(instr);

            /* Only replace the root variable dereference.  Child derefs stay
             * intact so struct and array indexing keeps working.
             */
            if (deref->deref_type != nir_deref_type_var)
               continue;
            if (!nir_deref_mode_is_one_of(deref,
                  nir_var_shader_call_data_in |
                  nir_var_shader_call_data_out |
                  nir_var_ray_hit_attrib))
               continue;

            b.cursor = nir_before_instr(instr);

            nir_def *offset;
            nir_variable_mode mode = deref->modes;

            if (mode & nir_var_ray_hit_attrib) {
               uint32_t ray_level = ctx->level ? ctx->level - 1 : 0;
               offset = nir_imm_int(
                  &b, ctx->traversal_states[ray_level].hit_attr_offset / 4);
            } else if (mode & nir_var_shader_call_data_in) {
               int32_t fixed_slot = callable_stage ?
                  ctx->fixed_callable_slot : ctx->fixed_payload_slot;
               offset = fixed_slot >= 0 ?
                  nir_imm_int(&b, fixed_slot) :
                  load_u32(&b, callable_stage ?
                              s->callable_ptr : s->payload_ptr);
            } else {
               struct hash_entry *entry =
                  _mesa_hash_table_search(ctx->var_slots, deref->var);
               uint32_t slot = entry ?
                  (uint32_t)(uintptr_t)entry->data - 1 : 0;
               offset = nir_imm_int(&b, slot / 4);
            }

            nir_deref_instr *data = nir_build_deref_var(&b, s->rt_data);
            nir_deref_instr *elem =
               nir_build_deref_array(&b, data, offset);
            nir_deref_instr *cast =
               nir_build_deref_cast(&b, &elem->def, nir_var_function_temp,
                                    deref->type, 4);
            nir_def_replace(&deref->def, &cast->def);
            progress = true;
         }
      }
   }

   return progress;
}

bool
tu_nir_lower_ray_tracing(nir_shader *shader, struct tu_rt_lowering_info *info)
{
   struct tu_rt_lower_ctx ctx = {
      .info = info,
      .func_to_stage = _mesa_pointer_hash_table_create(NULL),
      .var_slots = _mesa_pointer_hash_table_create(NULL),
      .fixed_payload_slot = -1,
      .fixed_callable_slot = -1,
   };

   ctx.levels = info->levels;
   ctx.callable_levels = info->callable_levels;

   for (uint32_t i = 0; i < info->stage_count; i++) {
      for (uint32_t level = 0; level < info->levels; level++) {
         for (uint32_t call_depth = 0;
              call_depth < info->callable_levels; call_depth++) {
            nir_function *func =
               info->funcs[(i * info->levels + level) *
                              info->callable_levels +
                           call_depth];
            if (!func)
               continue;
            uint32_t value = (call_depth << 24) | (level << 16) | (i + 1);
            _mesa_hash_table_insert(ctx.func_to_stage, func->impl,
                                    (void *)(uintptr_t)value);
         }
      }
   }

   ctx.has_any_hit = pipeline_has_stage(info, TU_RT_ANY_HIT);
   ctx.has_closest_hit = pipeline_has_stage(info, TU_RT_CLOSEST_HIT);
   ctx.has_intersection = pipeline_has_stage(info, TU_RT_INTERSECTION);
   ctx.has_miss = pipeline_has_general_stage(info, VK_SHADER_STAGE_MISS_BIT_KHR);
   ctx.has_callable =
      pipeline_has_general_stage(info, VK_SHADER_STAGE_CALLABLE_BIT_KHR);

   collect_call_data_slots(shader, &ctx);
   info->launch_id = rt_var(
      shader, glsl_vector_type(GLSL_TYPE_UINT, 3), "rt_launch_id");
   rt_state_init(shader, ctx.traversal_states, ctx.hit_states,
                 ctx.callable_states, ctx.levels, ctx.rt_data_size,
                 ctx.hit_attr_base);

   for (uint32_t level = 0; level < ctx.levels; level++) {
      info->shader_record_ptr[level] =
         ctx.hit_states[level].shader_record_ptr;
      info->payload_ptr[level] = ctx.hit_states[level].payload_ptr;
      info->callable_ptr[level] = ctx.hit_states[level].callable_ptr;
   }

   /* Lower the ray tracing intrinsics first: their payload/callable operands
    * still point at the original variables, which lets us map each call site
    * to its outgoing data slot.  lower_derefs() then rewrites all variable
    * accesses (including the ones inside the stages that are now called).
    */
   /* lower_instr() inserts control flow (the traversal loop and stage-call
    * dispatch), so neither block indices nor dominance survive.
    */
   bool progress = nir_shader_instructions_pass(
      shader, lower_instr, nir_metadata_none, &ctx);
   progress |= lower_derefs(shader, &ctx);
   progress |= nir_shader_instructions_pass(
      shader, lower_shader_record_io,
      nir_metadata_block_index | nir_metadata_dominance, NULL);

   _mesa_hash_table_destroy(ctx.var_slots, NULL);
   _mesa_hash_table_destroy(ctx.func_to_stage, NULL);

   /* Child derefs inherited shader_call_data/ray_hit_attrib modes from their
    * original root variable.  They now point into rt_data, so propagate the
    * new local-memory mode down the chain.
    */
   nir_fixup_deref_modes(shader);

   /* All accesses to the original call-data variables now go through rt_data.
    * Drop the now-dead variables; their modes are not valid for the compute
    * shader stage we compile them into.
    */
   nir_remove_dead_derefs(shader);
   nir_remove_dead_variables(shader,
      nir_var_shader_call_data_in |
      nir_var_shader_call_data_out |
      nir_var_ray_hit_attrib, NULL);

   return progress;
}

void
tu_nir_build_raygen_dispatch(nir_shader *shader,
                             const struct tu_rt_lowering_info *info)
{
   const struct tu_ray_tracing_pipeline *pipeline = info->pipeline;
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   nir_def *launch_size = load_launch_size(&b, info);
   nir_def *size_x = nir_channel(&b, launch_size, 0);
   nir_def *size_y = nir_channel(&b, launch_size, 1);
   nir_def *size_z = nir_channel(&b, launch_size, 2);
   nir_def *total_rays =
      nir_imul(&b, nir_imul(&b, size_x, size_y), size_z);

   nir_store_var(&b, info->launch_id,
                 nir_imm_vec3(&b, 0, 0, 0), 7);

   nir_def *base = load_sbt_base(&b, info,
                                 offsetof(struct tu_rt_trace_params, raygen_sbt),
                                 offsetof(VkTraceRaysIndirectCommand2KHR,
                                          raygenShaderRecordAddress));
   nir_def *stride = load_sbt_stride(&b, info,
                                     offsetof(struct tu_rt_trace_params, raygen_stride),
                                     UINT32_MAX);
   nir_def *entry = nir_iadd(&b, base, nir_u2u64(&b, nir_imul(&b, stride,
                                                              nir_imm_int(&b, 0))));
   nir_def *handle = nir_load_global(&b, 1, 32, entry, .align_mul = 4);
   nir_def *record = nir_iadd_imm(&b, entry, TU_RT_HANDLE_SIZE);

   /* Persistent traversal: every invocation walks a strided sequence of rays.
    * This keeps indirect dispatches correct without a host-side divide of the
    * 3D launch size, and gives each invocation a chance to reuse its caches
    * across several rays.
    */
   nir_def *num_groups = nir_load_num_workgroups(&b);
   nir_def *groups_x = nir_channel(&b, num_groups, 0);
   nir_def *groups_y = nir_channel(&b, num_groups, 1);
   nir_def *groups_z = nir_channel(&b, num_groups, 2);
   nir_def *invocations_x = nir_imul_imm(&b, groups_x, TU_RT_WORKGROUP_SIZE);
   nir_def *invocations_xy = nir_imul(&b, invocations_x, groups_y);
   nir_def *total_invocations = nir_imul(&b, invocations_xy, groups_z);

   nir_def *global_id = nir_load_global_invocation_id(&b, 32);
   nir_def *linear_id = nir_channel(&b, global_id, 0);
   linear_id = nir_iadd(&b, linear_id,
      nir_imul(&b, nir_channel(&b, global_id, 1), invocations_x));
   linear_id = nir_iadd(&b, linear_id,
      nir_imul(&b, nir_channel(&b, global_id, 2), invocations_xy));

   /* The loop-carried index must live in a variable so nir_lower_vars_to_ssa
    * can turn it into a phi.  Assigning a new SSA def to the C variable would
    * leave the loop back-edge reading the initial value forever.
    */
   nir_variable *linear_id_var =
      nir_local_variable_create(b.impl, glsl_uint_type(), "rt_linear_id");
   nir_store_var(&b, linear_id_var, linear_id, 1);

   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *linear_id = nir_load_var(&b, linear_id_var);
      nir_push_if(&b, nir_ieq_imm(&b, total_rays, 0));
      {
         nir_jump(&b, nir_jump_break);
      }
      nir_pop_if(&b, NULL);

      nir_push_if(&b, nir_uge(&b, linear_id, total_rays));
      {
         nir_jump(&b, nir_jump_break);
      }
      nir_pop_if(&b, NULL);

      /* Turn the linear ray index back into gl_LaunchIDEXT. */
      nir_def *launch_x = nir_umod(&b, linear_id, size_x);
      nir_def *launch_y = nir_umod(&b,
         nir_udiv(&b, linear_id, size_x), size_y);
      nir_def *launch_z = nir_udiv(&b, linear_id,
         nir_imul(&b, size_x, size_y));
      nir_store_var(&b, info->launch_id,
                    nir_vec3(&b, launch_x, launch_y, launch_z), 7);

      for (uint32_t level = 0; level < info->levels; level++) {
         nir_store_var(&b, info->payload_ptr[level], nir_imm_int(&b, 0), 1);
         nir_store_var(&b, info->callable_ptr[level], nir_imm_int(&b, 0), 1);
      }

      for (uint32_t i = 0; i < pipeline->group_count; i++) {
         const struct tu_rt_group *group = &pipeline->groups[i];
         if (group->general >= info->stage_count)
            continue;
         if (info->stages[group->general].stage != VK_SHADER_STAGE_RAYGEN_BIT_KHR)
            continue;
         nir_function *func =
            info->funcs[(group->general * info->levels + 0) *
                           info->callable_levels +
                        0];
         if (!func || func->impl == NULL)
            continue;

         nir_push_if(&b, nir_ieq(&b, handle, nir_imm_int(&b, group->handle)));
         nir_store_var(&b, info->shader_record_ptr[0], record, 1);
         nir_build_call(&b, func, 0, NULL);
         nir_pop_if(&b, NULL);
      }

      nir_store_var(&b, linear_id_var,
                    nir_iadd(&b, linear_id, total_invocations), 1);
   }
   nir_pop_loop(&b, loop);
}
