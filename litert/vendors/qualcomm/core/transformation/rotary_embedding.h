// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_ROTARY_EMBEDDING_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_ROTARY_EMBEDDING_H_

#include <cstddef>
#include <functional>
#include <vector>

#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

namespace qnn {

// Fuses the 9-op decomposed RoPE pattern:
//   Convert -> StridedSlice(x1) -> StridedSlice(x2) -> Concat([x2,x1])
//   -> ElementWiseBinary MUL(x, cos) -> ElementWiseBinary MUL(concat, sin)
//   -> ElementWiseBinary ADD -> Convert -> Transpose(perm=[0,2,1,3])
//
// Replaced by:
//   Unpack(x, axis=2)  →  H × [B,S,D]
//   Reshape each        →  H × [B,1,S,D]
//   cos/sin reshape     →  [B,S,D]  (squeeze head dim=1)
//   cos/sin slice       →  [B,S,D/2] (take +sin half)
//   H × RotaryEmbedding([B,1,S,D], cos, sin)  →  H × [B,1,S,D]
//   Concat(axis=1) all heads  →  [B,H,S,D]
//   Single Convert (output quantize)  →  [B,H,S,D]  (into transpose output)
//
// This eliminates the Transpose by using Pack(axis=1) which stacks each
// [B,1,S,D] as a contiguous block — no strided scatter.
size_t FuseRotaryEmbeddingWithTranspose(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_ROTARY_EMBEDDING_H_
