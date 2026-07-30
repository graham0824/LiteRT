// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_FOLD_BOUNDARY_CAST_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_FOLD_BOUNDARY_CAST_H_

#include <cstddef>
#include <functional>
#include <vector>

#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

namespace qnn {

// Folds a same-scale int8<->uint8 (offset-diff-128) boundary Cast by rewiring
// its producer/consumer to the opposite-signedness tensor and dropping the
// Cast. The two 8-bit tensors are bit-identical (int8_val - (-128) ==
// uint8_val - 0), so this is a numerically exact re-label with no precision
// loss -- the same transform qnn-tflite-converter applies for all backends.
//
// This is required for LPAI, whose adaptor rejects a Cast with differing input
// and output data types (lpai_adaptor_op_handler validate_cast). HTP accepts
// such a Cast, so LiteRT never needed this before; LPAI is the first backend to
// expose the gap.
//
// Matches a single QnnOpCode::kCast. Returns the number of indices to advance.
size_t FoldBoundaryCast(std::function<bool(OpWrapper&)> validate_op_config,
                        std::vector<OpWrapper>& ops, size_t start_index,
                        TensorPool& tensor_pool, size_t pattern_size);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_FOLD_BOUNDARY_CAST_H_
