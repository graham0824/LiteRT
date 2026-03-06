// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/fully_connected_op_builder.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace qnn {

namespace {
constexpr int kBiasIdx = 2;
OpWrapper CreateConvertOp(const TensorWrapper& input,
                          const TensorWrapper& output) {
  OpWrapper op(GetUniqueOpName(QNN_OP_CONVERT), QNN_OP_CONVERT,
               QnnOpCode::kConvert);
  op.AddInputTensor(input);
  op.AddOutputTensor(output);
  return op;
}
}

std::vector<OpWrapper> BuildFullyConnectedOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const bool keep_num_dims,
    bool use_int64_bias_as_int32) {
  std::vector<OpWrapper> res;
  TensorWrapper& input_tensor = inputs[0];
  TensorWrapper& weight_tensor = inputs[1];
  TensorWrapper& output_tensor = outputs[0];
  const bool is_a8w2_per_channel_quant =
      input_tensor.IsQuantI8() && output_tensor.IsQuantI8() &&
      weight_tensor.IsQuantI8() && weight_tensor.IsBwPerChannelQuant(2);
  TensorWrapper* input_ptr = &input_tensor;
  if (is_a8w2_per_channel_quant) {
    QNN_LOG_DEBUG("A8W2: Convert input from A8 to A16.");
    auto& input_sfp16 = tensor_pool.CloneNativeTensorFrom(
        input_tensor, QNN_DATATYPE_SFIXED_POINT_16);
    res.emplace_back(CreateConvertOp(input_tensor, input_sfp16)); 
    input_ptr = &input_sfp16;
  }
  OpWrapper& fully_connected_op = CreateOpWrapper(res, QNN_OP_FULLY_CONNECTED);
  fully_connected_op.AddInputTensor(*input_ptr);
  fully_connected_op.AddInputTensor(weight_tensor);
  if (inputs.size() - 1 >= kBiasIdx) {
    TensorWrapper& bias_tensor = inputs[kBiasIdx];
    if (use_int64_bias_as_int32 && bias_tensor.IsTensorStatic() &&
        bias_tensor.GetDataType() == QNN_DATATYPE_INT_64) {
      auto* converted_bias_tensor =
          tensor_pool.ConvertStaticTensorFrom<std::int32_t>(bias_tensor);
      if (converted_bias_tensor == nullptr) {
        return {};
      }
      fully_connected_op.AddInputTensor(*converted_bias_tensor);
      QNN_LOG_WARNING(
          "Convert bias tensor in fully connected op from int64 to int32.");
    } else {
      fully_connected_op.AddInputTensor(bias_tensor);
    }
  }

  if (keep_num_dims) {
    auto& input_dims = input_tensor.GetDimensions();
    std::uint32_t input_size = std::accumulate(
        input_dims.begin(), input_dims.end(), 1, std::multiplies<>());
    const std::uint32_t num_units = weight_tensor.GetDimension(0);
    const std::uint32_t num_input_elem = weight_tensor.GetDimension(1);

    // input_size must be divisible by num_input_elem. This should be validated
    // by QNN.
    const std::uint32_t batch_size = input_size / num_input_elem;
    // QNN output should always be rank 2
    qnn::TensorWrapper& fully_connected_out = tensor_pool.CloneNativeTensorFrom(
        output_tensor, {batch_size, num_units});
    if (is_a8w2_per_channel_quant) {
      QNN_LOG_DEBUG("A8W2(keep_num_dims): Convert output from A8 to A16.");
      auto& output_sfp16 = tensor_pool.CloneNativeTensorFrom(
          fully_connected_out, QNN_DATATYPE_SFIXED_POINT_16);
      fully_connected_op.AddOutputTensor(output_sfp16);
      res.emplace_back(CreateConvertOp(output_sfp16, fully_connected_out));
    } else {
      fully_connected_op.AddOutputTensor(fully_connected_out);
    }
    qnn::OpWrapper& reshape_op = CreateOpWrapper(res, QNN_OP_RESHAPE);
    reshape_op.AddInputTensor(fully_connected_out);
    reshape_op.AddOutputTensor(output_tensor);
  } else {
    if (is_a8w2_per_channel_quant) {
      QNN_LOG_DEBUG("A8W2: Convert output from A8 to A16.");
      auto& output_sfp16 = tensor_pool.CloneNativeTensorFrom(
          output_tensor, QNN_DATATYPE_SFIXED_POINT_16);
      fully_connected_op.AddOutputTensor(output_sfp16);
      res.emplace_back(CreateConvertOp(output_sfp16, output_tensor));
    } else {
      fully_connected_op.AddOutputTensor(outputs[0]);
    }
  }

  return res;
}

}  // namespace qnn
