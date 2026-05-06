// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_QNN_TRANSFORMATIONS_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_QNN_TRANSFORMATIONS_H_

#include "litert/c/litert_builder.h"
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Fuses BatchMatmul -> Quantize where the Quantize is a pure requantization
// (input/output element types match) and the BatchMatmul has a single user.
// The pattern is rewritten into a single BatchMatmul op that carries the
// Quantize output's quantization parameters.
LiteRtStatus FuseMatMulRequantTransformation(LiteRtBuilder builder_ptr,
                                             LiteRtOp op);

// Matches the composite KV-swapped attention pattern rooted at the final
// Reshape (R5):
//   IN_TOP -> Mul(M0) -> Reshape(R1) -+-> BatchMatmul(MM1) -+
//                                     +-> BatchMatmul(MM2) -+-> Concat(C1)
//   C1 -> Reshape(R2) -> Add(A1, +IN_LEFT) -> Reshape(R3) -> Softmax(S1)
//   S1 -> Slice(SL) -> BatchMatmul(MM3) -+
//   S1 -> Slice(SR) -> BatchMatmul(MM4) -+-> Add(A2) -> Reshape(R4)
//                                                    -> Transpose(T1)
//                                                    -> Reshape(R5) [op]
//
// Pure matcher. Returns kLiteRtStatusOk on match, kLiteRtStatusPatternNoMatch
// otherwise. Does not modify the graph.
LiteRtStatus MatchCompositeAttentionPattern(LiteRtOp op);

// Matches the same KV-swapped composite attention pattern as
// ApplyCompositeAttentionTransformation. On success, replaces ONLY the
// Q-scale Mul (M0) with an Add that has identical inputs and outputs.
// The replacement Add's output tensor is renamed to the string
// "DBG_KVSwapAttn_QScaleAdd_ReplacedMul" so the node is immediately
// visible in Netron / other graph-viewer tools as an unusual Add in the
// Q-projection path.
//
// Use case: inject a visually distinct marker to confirm the pattern is
// matched and the rewrite pipeline is live, without changing graph
// connectivity. Useful before committing the full MHA->SHA rewrite.
LiteRtStatus DebugReplaceQScaleMulWithAdd(LiteRtBuilder builder_ptr,
                                          LiteRtOp op);
// logs the captured op output shapes, derives (num_attn_heads,
// num_kv_heads, num_attn_per_kv_heads) from the K/Q tensor dims, and logs
// the intended unpack plan.
//
// Phase 1: observation-only. Always returns kLiteRtStatusPatternNoMatch so
// the greedy rewriter does not re-fire on the same op. Actual Split/Reshape
// op construction and the per-head SHA rewrite come in phase 2.
LiteRtStatus ApplyCompositeAttentionTransformation(LiteRtBuilder builder_ptr,
                                                   LiteRtOp op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_QNN_TRANSFORMATIONS_H_
