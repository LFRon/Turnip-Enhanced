/*
 * Copyright © 2026 Mesa contributors
 * SPDX-License-Identifier: MIT
 *
 * Shared helpers for the Turnip BVH traversal loops used by ray queries and
 * ray-tracing pipelines.  The actual loops have different result handling
 * (ray query candidates vs. SBT stage calls), but the node/parent layout and
 * the overflow-backtracking primitive are identical.
 */

#ifndef TU_RAY_TRAVERSAL_H
#define TU_RAY_TRAVERSAL_H

#include "compiler/nir/nir_builder.h"

#define TU_RAY_BVH_INVALID_NODE 0xffffffffu
/* Explicitly different from TU_RAY_BVH_INVALID_NODE and any real node, so
 * that we never exit an instance when we are not inside one.
 */
#define TU_RAY_BVH_NO_INSTANCE_ROOT 0xfffffffeu

/* The BVH encoder stores each child's parent immediately before the 64-byte
 * node, i.e. at bvh_base - (node * 4 + 4).  Node 0's parent lives just before
 * the BVH and is set to TU_RAY_BVH_INVALID_NODE by the root encoder.
 */
static inline nir_def *
tu_ray_fetch_parent_node(nir_builder *b, nir_def *bvh, nir_def *node)
{
   nir_def *offset = nir_iadd_imm(b, nir_imul_imm(b, node, 4), 4);

   return nir_load_global(
      b, 1, 32, nir_isub(b, nir_pack_64_2x32(b, bvh), nir_u2u64(b, offset)),
      .access = static_cast<gl_access_qualifier>(
                   ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
      .align_mul = 4);
}

#endif /* TU_RAY_TRAVERSAL_H */
