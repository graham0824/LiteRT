// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/depthwise_conv2d_op_builder.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

#include "QnnOpDef.h" // from @qairt
#include "QnnTypes.h" // from @qairt
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

namespace {
constexpr size_t kInputIndex = 0;
constexpr size_t kFilterIndex = 1;
constexpr size_t kBiasIndex = 2;
constexpr size_t kOutputIndex = 0;
constexpr size_t kBatchIndex = 0;
constexpr size_t kHeightIndex = 1;
constexpr size_t kWidthIndex = 2;
constexpr size_t kChannelIndex = 3;
constexpr std::uint32_t kConv2dMinStride = 4;
constexpr std::uint32_t kConv2dMinFilterSize = 4;

bool ShouldUseGroupedConv2d(const TensorWrapper &filter_tensor,
                            std::uint32_t stride_h, std::uint32_t stride_w) {
  return stride_h >= kConv2dMinStride && stride_w >= kConv2dMinStride &&
         filter_tensor.GetDimension(kHeightIndex) >= kConv2dMinFilterSize &&
         filter_tensor.GetDimension(kWidthIndex) >= kConv2dMinFilterSize;
}

TensorWrapper *DequantizePerChannelInt8StaticTensor(TensorPool &tensor_pool,
                                                    const TensorWrapper &src) {
  if (!src.IsTensorStatic() || !src.IsQuantI8() || !src.IsPerChannelQuant()) {
    QNN_LOG_ERROR("DepthwiseConv2d filter must be static per-channel int8.");
    return nullptr;
  }

  const auto src_data = src.GetTensorData<std::int8_t>();
  if (!src_data.has_value()) {
    QNN_LOG_ERROR("Failed to get DepthwiseConv2d filter tensor data.");
    return nullptr;
  }

  const auto &quant_params =
      std::get<AxisScaleOffsetQuantizeParamsWrapper>(src.GetQuantParams());
  const auto scales = quant_params.GetScales();
  const auto zero_points = quant_params.GetZeroPoints();
  const auto axis = quant_params.GetAxis();
  if (axis < 0 || static_cast<std::uint32_t>(axis) >= src.GetRank()) {
    QNN_LOG_ERROR("Invalid DepthwiseConv2d filter quantization axis.");
    return nullptr;
  }

  const auto axis_index = static_cast<size_t>(axis);
  const auto &dims = src.GetDimensions();
  const auto channel_count = dims[axis_index];
  if (scales.size() != channel_count || zero_points.size() != channel_count) {
    QNN_LOG_ERROR(
        "DepthwiseConv2d filter quantization params do not match axis size.");
    return nullptr;
  }

  size_t axis_stride = 1;
  for (size_t i = axis_index + 1; i < dims.size(); ++i) {
    axis_stride *= dims[i];
  }

  std::vector<float> dequantized_data;
  dequantized_data.reserve(src_data->size());
  for (size_t i = 0; i < src_data->size(); ++i) {
    const auto channel = (i / axis_stride) % channel_count;
    dequantized_data.emplace_back(
        Dequantize((*src_data)[i], scales[channel], zero_points[channel]));
  }

  return &tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, QuantizeParamsWrapperVariant{},
      src.GetDimensions(), dequantized_data.size() * sizeof(float),
      dequantized_data.data());
}

} // namespace

std::vector<OpWrapper> BuildDepthwiseConv2dOp(
    TensorPool &tensor_pool, const std::vector<TensorWrapperRef> &inputs,
    const std::vector<TensorWrapperRef> &outputs, const std::uint32_t stride_h,
    const std::uint32_t stride_w, const std::uint32_t dilation_h,
    const std::uint32_t dilation_w, const PaddingType padding_type) {
  std::vector<OpWrapper> res;

  TensorWrapper &input_tensor = inputs[kInputIndex];
  TensorWrapper &filter_tensor = inputs[kFilterIndex];
  const bool has_bias = inputs.size() - 1 >= kBiasIndex;
  TensorWrapper *effective_filter_tensor = &filter_tensor;
  if (input_tensor.IsF32() && outputs[kOutputIndex].get().IsF32() &&
      filter_tensor.IsTensorStatic() && filter_tensor.IsQuantI8() &&
      filter_tensor.IsPerChannelQuant() &&
      (!has_bias || inputs[kBiasIndex].get().IsF32())) {
    effective_filter_tensor =
        DequantizePerChannelInt8StaticTensor(tensor_pool, filter_tensor);
    if (effective_filter_tensor == nullptr) {
      return {};
    }
  }

  // 1HWC to HW1C, only need reshape instead of transpose.
  const std::vector<std::uint32_t> reshape_dims{
      effective_filter_tensor->GetDimension(kHeightIndex),
      effective_filter_tensor->GetDimension(kWidthIndex),
      effective_filter_tensor->GetDimension(kBatchIndex),
      effective_filter_tensor->GetDimension(kChannelIndex)};
  TensorWrapper *reshaped_filter_tensor = nullptr;
  if (effective_filter_tensor->IsTensorStatic()) {
    reshaped_filter_tensor = &(tensor_pool.CloneStaticTensorFrom(
        *effective_filter_tensor, reshape_dims));
  } else {
    reshaped_filter_tensor = &(tensor_pool.CloneNativeTensorFrom(
        *effective_filter_tensor, reshape_dims));

    OpWrapper &reshape_op = CreateOpWrapper(res, QNN_OP_RESHAPE);
    reshape_op.AddInputTensor(*effective_filter_tensor);
    reshape_op.AddOutputTensor(*reshaped_filter_tensor);
  }
  TensorWrapper *bias_tensor = nullptr;
  if (has_bias) {
    bias_tensor = &inputs[kBiasIndex].get();
    // QNN only support per-tensor quant for bias,
    // and the scale and offset are both zero.
    bias_tensor->ConvertAxisScaleOffsetToScaleOffset();
  }

  // stride param
  const std::array<std::uint32_t, 2> stride_data{stride_h, stride_w};
  const std::vector<std::uint32_t> stride_shape{2};
  auto &stride_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, stride_shape,
      sizeof(decltype(stride_data)::value_type) * stride_data.size(),
      stride_data.data());

  // dilation param
  const std::array<std::uint32_t, 2> dilation_data{dilation_h, dilation_w};
  const std::vector<std::uint32_t> dilation_shape{2};
  auto &dilation_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, dilation_shape,
      sizeof(decltype(dilation_data)::value_type) * dilation_data.size(),
      dilation_data.data());

  // padding param
  const auto [padding_before_height, padding_after_height] =
      ComputePaddingBeforeAfter(
          input_tensor.GetDimension(kHeightIndex),
          effective_filter_tensor->GetDimension(kHeightIndex), stride_h,
          dilation_h, padding_type);
  const auto [padding_before_width, padding_after_width] =
      ComputePaddingBeforeAfter(
          input_tensor.GetDimension(kWidthIndex),
          effective_filter_tensor->GetDimension(kWidthIndex), stride_w,
          dilation_w, padding_type);
  const std::array<std::uint32_t, 4> padding_data = {
      padding_before_height, padding_after_height, padding_before_width,
      padding_after_width};
  const std::vector<std::uint32_t> padding_shape{2, 2};
  auto &padding_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, padding_shape,
      sizeof(decltype(padding_data)::value_type) * padding_data.size(),
      padding_data.data());

  if (ShouldUseGroupedConv2d(*effective_filter_tensor, stride_h, stride_w)) {
    OpWrapper &conv_op = CreateOpWrapper(res, QNN_OP_CONV_2D);
    conv_op.AddInputTensor(input_tensor);
    conv_op.AddInputTensor(*reshaped_filter_tensor);
    if (bias_tensor) {
      conv_op.AddInputTensor(*bias_tensor);
    }
    conv_op.AddOutputTensor(outputs[kOutputIndex]);
    conv_op.AddScalarParam<std::uint32_t>(
        QNN_OP_CONV_2D_PARAM_GROUP, input_tensor.GetDimension(kChannelIndex));
    conv_op.AddTensorParam(QNN_OP_CONV_2D_PARAM_STRIDE, stride_tensor);
    conv_op.AddTensorParam(QNN_OP_CONV_2D_PARAM_DILATION, dilation_tensor);
    conv_op.AddTensorParam(QNN_OP_CONV_2D_PARAM_PAD_AMOUNT, padding_tensor);
  } else {
    OpWrapper &conv_op = CreateOpWrapper(res, QNN_OP_DEPTH_WISE_CONV_2D);
    conv_op.AddInputTensor(input_tensor);
    conv_op.AddInputTensor(*reshaped_filter_tensor);
    if (bias_tensor) {
      conv_op.AddInputTensor(*bias_tensor);
    }
    conv_op.AddOutputTensor(outputs[kOutputIndex]);
    conv_op.AddTensorParam(QNN_OP_DEPTH_WISE_CONV_2D_PARAM_STRIDE,
                           stride_tensor);
    conv_op.AddTensorParam(QNN_OP_DEPTH_WISE_CONV_2D_PARAM_DILATION,
                           dilation_tensor);
    conv_op.AddTensorParam(QNN_OP_CONV_2D_PARAM_PAD_AMOUNT, padding_tensor);
  }
  return res;
}

} // namespace qnn
