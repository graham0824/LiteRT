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
  // Tile FC computation along the K dimension and combine partial results using
  // an adder-tree. This allows the FC operation to scale by splitting large
  // weight into smaller tiles while maintaining the same output shape.
  //
  //      k            k             n
  //   ┌─────┐      ┌─────┐     ┌─────────┐
  // m │  I  │  x   │     │ = m |    O    |
  //   └─────┘    n │  W  │     └─────────┘
  //                │     │
  //                └─────┘
  //
  // For example, if t (num tiles) = 2, we have:
  //
  //   k/t          k/t              n
  //   ┌─┐ ┌─┐      ┌─┐ ┌─┐     ┌─────────┐   ┌─────────┐
  // m │I| |I│  x   │ │ │ │ = m |    O    | + |    O    |
  //   └─┘ └─┘    n │W│ │W│     └─────────┘   └─────────┘
  //                │ │ │ │
  //                └─┘ └─┘
  //
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
  static constexpr uint32_t kNumTiles = 2;
  // Split
  const auto& input = ops[start_index].GetInputTensor(0);
  auto split_dims = input.GetDimensions();
  split_dims.back() /= kNumTiles;
  const std::uint32_t tile_size = split_dims.back();
  std::vector<TensorWrapperRef> fc_inputs;
  fc_inputs.reserve(kNumTiles);
  for (size_t i = 0; i < kNumTiles; ++i) {
    fc_inputs.emplace_back(
        tensor_pool.CloneNativeTensorFrom(input, split_dims));
  }

  std::vector<std::uint32_t> split_index_dims = {kNumTiles - 1};
  std::vector<std::uint32_t> split_index;
  split_index.reserve(split_index_dims[0]);
  for (std::uint32_t i = tile_size; i < input.GetDimensions().back();
       i += tile_size) {
    split_index.emplace_back(i);
  }
  auto& split_index_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, {}, split_index_dims,
      sizeof(split_index[0]) * split_index.size(), split_index.data());
  std::vector<OpWrapper> new_ops;
  new_ops.reserve(2 + kNumTiles);
  new_ops.emplace_back(
      CreateSplitOp(input, fc_inputs, input.GetRank() - 1, split_index_tensor));
  // Graph transform
  QNN_LOG_INFO("[G2G] Tile FC");
  auto weight_dims = weight.GetDimensions();
  weight_dims.back() /= kNumTiles;
  std::vector<TensorWrapperRef> fc_outputs;
  fc_outputs.reserve(fc_inputs.size());
  const auto& output = ops[start_index].GetOutputTensor(0);
  for (size_t op_index = 0; op_index < kNumTiles; ++op_index) {
    std::vector<int8_t> fc_weight;
    fc_weight.reserve(weight.GetTensorNumElements() / kNumTiles);
    for (size_t i = 0; i < weight.GetDimension(0); ++i) {
      for (size_t j = op_index * tile_size; j < (op_index + 1) * tile_size;
           ++j) {
        fc_weight.emplace_back(weight_data.value()[j]);
      }
    }
    auto& tiled_weight = tensor_pool.CreateStaticTensor(
        weight.GetDataType(), weight.GetQuantParams(), weight_dims,
        weight.GetTensorBytes(), fc_weight.data());
    auto& fc_output = fc_outputs.emplace_back(
        tensor_pool.CloneNativeTensorFrom(output));
    new_ops.emplace_back(CloneOpWithIO(
        ops[start_index], {fc_inputs[op_index], tiled_weight}, {fc_output}));
  }
  // TODO (jiunkaiy): Change to adder tree.
  new_ops.emplace_back(
      CreateElementWiseBinaryOp(fc_outputs[0], fc_outputs[1], output,
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
