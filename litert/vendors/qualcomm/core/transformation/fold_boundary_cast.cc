// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/fold_boundary_cast.h"

#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

namespace {

// Rebuilds op `idx` with one tensor slot swapped, validates it, and commits in
// place on success. `is_input` selects whether the swap is on the input or
// output side; `slot` is which tensor index to replace with `replacement`.
// Returns true iff the rewired op validated and was committed.
bool RewireAndValidate(std::function<bool(OpWrapper&)> validate_op_config,
                       std::vector<OpWrapper>& ops, size_t idx, bool is_input,
                       const TensorWrapper& old_tensor,
                       const TensorWrapper& replacement) {
  OpWrapper& op = ops[idx];
  std::vector<ConstTensorWrapperRef> inputs;
  std::vector<ConstTensorWrapperRef> outputs;
  for (size_t i = 0; i < op.GetInputCount(); ++i) {
    const TensorWrapper& t = op.GetInputTensor(i);
    inputs.emplace_back((is_input && t == old_tensor) ? replacement : t);
  }
  for (size_t i = 0; i < op.GetOutputCount(); ++i) {
    const TensorWrapper& t = op.GetOutputTensor(i);
    outputs.emplace_back((!is_input && t == old_tensor) ? replacement : t);
  }
  OpWrapper rewired = CreateOpWithSameParams(op, inputs, outputs);
  if (!validate_op_config(rewired)) {
    return false;
  }
  ops[idx] = std::move(rewired);
  return true;
}

}  // namespace

size_t FoldBoundaryCast(std::function<bool(OpWrapper&)> validate_op_config,
                        std::vector<OpWrapper>& ops, size_t start_index,
                        TensorPool& tensor_pool, size_t pattern_size) {
  OpWrapper& cast = ops[start_index];
  const TensorWrapper& cast_in = cast.GetInputTensor(0);
  const TensorWrapper& cast_out = cast.GetOutputTensor(0);

  // Only same-scale int8<->uint8 (offset-diff-128) boundary casts are exact
  // re-labels. Everything else must keep its Cast.
  if (!cast_in.IsPerTensorQuantWithOffsetDiff(cast_out)) {
    return 1;
  }

  // Egress: the cast writes a subgraph output (uint8). Rewire the producer of
  // cast_in to write cast_out directly, then drop the cast.
  if (cast_out.IsSubgraphOutput()) {
    for (size_t i = 0; i < start_index; ++i) {
      bool produces = false;
      for (size_t o = 0; o < ops[i].GetOutputCount(); ++o) {
        if (ops[i].GetOutputTensor(o) == cast_in) {
          produces = true;
          break;
        }
      }
      if (!produces) {
        continue;
      }
      if (RewireAndValidate(validate_op_config, ops, i, /*is_input=*/false,
                            cast_in, cast_out)) {
        QNN_LOG_INFO("[G2G] Folded egress int8/uint8 boundary Cast.");
        ops.erase(ops.begin() + start_index);
        return 0;
      }
      QNN_LOG_WARNING(
          "[G2G] Egress boundary-Cast producer rejected re-label; keeping "
          "Cast.");
      return 1;
    }
    return 1;
  }

  // Ingress: the cast reads a subgraph input (uint8). Rewire every consumer of
  // cast_out to read cast_in directly, then drop the cast.
  if (cast_in.IsSubgraphInput()) {
    std::vector<size_t> consumers;
    for (size_t i = start_index + 1; i < ops.size(); ++i) {
      for (size_t in = 0; in < ops[i].GetInputCount(); ++in) {
        if (ops[i].GetInputTensor(in) == cast_out) {
          consumers.push_back(i);
          break;
        }
      }
    }
    if (consumers.empty()) {
      return 1;
    }
    // All consumers must accept the re-label, otherwise we cannot drop the Cast
    // (partial rewire would leave a dangling producer-less tensor).
    for (size_t idx : consumers) {
      OpWrapper& consumer = ops[idx];
      std::vector<ConstTensorWrapperRef> inputs;
      std::vector<ConstTensorWrapperRef> outputs;
      for (size_t in = 0; in < consumer.GetInputCount(); ++in) {
        const TensorWrapper& t = consumer.GetInputTensor(in);
        inputs.emplace_back(t == cast_out ? cast_in : t);
      }
      for (size_t o = 0; o < consumer.GetOutputCount(); ++o) {
        outputs.emplace_back(consumer.GetOutputTensor(o));
      }
      OpWrapper probe = CreateOpWithSameParams(consumer, inputs, outputs);
      if (!validate_op_config(probe)) {
        QNN_LOG_WARNING(
            "[G2G] Ingress boundary-Cast consumer rejected re-label; keeping "
            "Cast.");
        return 1;
      }
    }
    for (size_t idx : consumers) {
      RewireAndValidate(validate_op_config, ops, idx, /*is_input=*/true,
                        cast_out, cast_in);
    }
    QNN_LOG_INFO("[G2G] Folded ingress int8/uint8 boundary Cast.");
    ops.erase(ops.begin() + start_index);
    return 0;
  }

  return 1;
}

}  // namespace qnn
