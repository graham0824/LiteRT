// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/masked_softmax.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/select_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reduce_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/softmax_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {
namespace {
    constexpr size_t kAddIndex = 0;
    constexpr size_t kSoftmaxIndex = 1;
}
size_t TransformMaskedSoftmax(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  // Connection check
  bool is_connected = ops[start_index + kAddIndex].GetOutputTensor(0) ==
                      ops[start_index + kSoftmaxIndex].GetInputTensor(0);
  if (!is_connected) {
    return 1;
  }

  if (!IsElementWiseAdd(ops[start_index + kAddIndex])) {
    return 1;
  }

  const auto& input = ops[start_index + kAddIndex].GetInputTensor(0);

  // ReduceMin
  auto min_out_dims = input.GetDimensions();
  min_out_dims.back() = 1u;
  auto& min_out = tensor_pool.CloneNativeTensorFrom(input, min_out_dims);
  const std::vector<std::uint32_t> kOneDim{1};
  const std::vector<std::uint32_t> kAxes{input.GetRank() - 1};
  auto& axes = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, {}, kOneDim, sizeof(std::uint32_t), kAxes.data());

  // Add (B: <= -20)
  auto& add_out = tensor_pool.CloneNativeTensorFrom(min_out);
  static constexpr std::int8_t kNegativeValue = -20;
  auto& negative_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, ScaleOffsetQuantizeParamsWrapper{1.0f, 0},
      kOneDim, sizeof(kNegativeValue), &kNegativeValue);

  // Graph transform
  QNN_LOG_INFO("[G2G] Transform MaskedSoftmax");

  // Construct the new subgraph
  const auto& mask = ops[start_index + kAddIndex].GetInputTensor(1);
  const auto& softmax_in = ops[start_index + kSoftmaxIndex].GetInputTensor(0);
  const auto& output = ops[start_index + kSoftmaxIndex].GetOutputTensor(0);
  std::vector<OpWrapper> new_ops =
      MakeVector(CreateReduceMinOp(input, min_out, axes, true),
                 CreateElementWistAddOp(min_out, negative_tensor, add_out),
                 CreateSelectOp(mask, input, add_out, softmax_in),
                 CreateSoftmaxOp(softmax_in, output, 1.0f));

  if (new_ops.empty()) {
    QNN_LOG_WARNING(
        "[G2G] Transformation failed. Rolling back to the original graph.");
    return 1;
  }
  // Validate new graph.
  bool is_valid =
      std::all_of(new_ops.begin(), new_ops.end(),
                  [validate_op_config](::qnn::OpWrapper& op_wrapper) -> bool {
                    return validate_op_config(op_wrapper);
                  });
  if (is_valid) {
    // Replace the matched pattern with a newly generated subgraph.
    size_t step_size = new_ops.size();
    ops.insert(ops.begin() + start_index + pattern_size,
               std::make_move_iterator(new_ops.begin()),
               std::make_move_iterator(new_ops.end()));
    ops.erase(ops.begin() + start_index,
              ops.begin() + start_index + pattern_size);
    return step_size;
  }
  QNN_LOG_WARNING(
      "[G2G] Validation failed. Rolling back to the original graph.");
  return 1;
}

}  // namespace qnn
