// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/fc_tiling.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>
#include <variant>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/split_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {
namespace {
OpWrapper CloneOpWithIO(
    const OpWrapper& source_op,
    const std::vector<std::optional<qnn::TensorWrapperRef>>& inputs,
    const std::vector<std::optional<qnn::TensorWrapperRef>>& outputs) {
  OpWrapper ret = source_op;
  ret.UpdateTensors(inputs, outputs);
  return ret;
}
}

size_t TileFullyConnected(std::function<bool(OpWrapper&)> validate_op_config,
                          std::vector<OpWrapper>& ops, size_t start_index,
                          TensorPool& tensor_pool, size_t pattern_size) {
  // Check size.
  const auto& input = ops[start_index].GetInputTensor(0);
  const auto& weight = ops[start_index].GetInputTensor(1);
  auto weight_data = weight.GetTensorData<int8_t>();
  QNN_LOG_INFO("[G2G] FC Weight Info:");
  for (size_t i = 0; i < weight.GetRank(); ++i) {
    QNN_LOG_INFO("[G2G] Dim %d: %d", i, weight.GetDimension(i));
  }
  if (!(weight.GetDimension(0) == 12288 || weight.GetDimension(0) == 1536 ||
        weight.GetDimension(0) == 262144)) {
    return 1;
  }
  if (!weight_data.has_value()) return 1;
  // Split
  auto split_output_dims = input.GetDimensions();
  split_output_dims[split_output_dims.size() - 1] /= 2;
  auto& fc_input_0 =
      tensor_pool.CloneNativeTensorFrom(input, split_output_dims);
  auto& fc_input_1 =
      tensor_pool.CloneNativeTensorFrom(input, split_output_dims);
  std::vector<std::uint32_t> split_index_dims = {1};
  std::vector<std::uint32_t> split_index = {
      input.GetDimension(input.GetRank() - 1) / 2};
  auto& split_index_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, {}, split_index_dims,
      sizeof(split_index[0]) * split_index.size(), split_index.data());

  // Graph transform
  QNN_LOG_INFO("[G2G] Tile FC");
  // Construct the new subgraph
  std::vector<int8_t> fc_weight_0;
  fc_weight_0.reserve(weight_data.value().size() / 2);
  std::vector<int8_t> fc_weight_1;
  fc_weight_1.reserve(weight_data.value().size() / 2);
  const size_t j_bound = weight.GetDimension(1);
  for (size_t i = 0; i < weight.GetDimension(0); ++i) {
    for (size_t j = 0; j < j_bound / 2; ++j) {
      fc_weight_0.emplace_back(weight_data.value()[j]);
    }
    for (size_t j = j_bound / 2; j < j_bound; ++j) {
      fc_weight_1.emplace_back(weight_data.value()[j]);
    }
  }

  auto weight_dims = weight.GetDimensions();
  weight_dims[weight.GetRank() - 1] /= 2;

  auto& tiled_weight_0 = tensor_pool.CreateStaticTensor(
      weight.GetDataType(), weight.GetQuantParams(), weight_dims,
      weight.GetTensorBytes(), fc_weight_0.data());
  auto& tiled_weight_1 = tensor_pool.CreateStaticTensor(
      weight.GetDataType(), weight.GetQuantParams(), weight_dims,
      weight.GetTensorBytes(), fc_weight_1.data());
  const auto& output_tensor = ops[start_index].GetOutputTensor(0);
  auto& fc_output_0 = tensor_pool.CloneNativeTensorFrom(output_tensor);
  auto& fc_output_1 = tensor_pool.CloneNativeTensorFrom(output_tensor);

  std::vector<OpWrapper> new_ops = MakeVector(
      CreateSplitOp(input, {fc_input_0, fc_input_1}, input.GetRank() - 1,
                    split_index_tensor),
      CloneOpWithIO(ops[start_index], {fc_input_0, tiled_weight_0},
                    {fc_output_0}),
      CloneOpWithIO(ops[start_index], {fc_input_1, tiled_weight_1},
                    {fc_output_1}),
      CreateElementWiseBinaryOp(fc_output_0, fc_output_1, output_tensor,
                                QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD));

  // Validate new graph.
  bool is_valid =
      std::all_of(new_ops.begin(), new_ops.end(),
                  [validate_op_config](::qnn::OpWrapper& op_wrapper) -> bool {
                    return validate_op_config(op_wrapper);
                  });
  if (is_valid) {
    // Adjust the name to avoid a name collision in the Qnn JSON dump.
    for (size_t i = 0; i < new_ops.size(); ++i) {
      new_ops[i].AddSuffixToName(absl::StrCat("_qcg2g_", i));
    }
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
