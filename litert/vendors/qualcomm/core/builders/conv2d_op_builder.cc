// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/conv2d_op_builder.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <variant>
#include <vector>

#include "QnnOpDef.h" // from @qairt
#include "QnnTypes.h" // from @qairt
#include "litert/vendors/qualcomm/core/builders/cast_op_builder.h"
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

bool IsOneByOneConv2d(const TensorWrapper &filter_tensor) {
  return filter_tensor.GetDimension(kHeightIndex) == 1 &&
         filter_tensor.GetDimension(kWidthIndex) == 1;
}

TensorWrapper *DequantizePerChannelInt8StaticTensor(TensorPool &tensor_pool,
                                                    const TensorWrapper &src) {
  if (!src.IsTensorStatic() || !src.IsQuantI8() || !src.IsPerChannelQuant()) {
    QNN_LOG_ERROR("Conv2d filter must be static per-channel int8.");
    return nullptr;
  }

  const auto src_data = src.GetTensorData<std::int8_t>();
  if (!src_data.has_value()) {
    QNN_LOG_ERROR("Failed to get Conv2d filter tensor data.");
    return nullptr;
  }

  const auto &quant_params =
      std::get<AxisScaleOffsetQuantizeParamsWrapper>(src.GetQuantParams());
  const auto scales = quant_params.GetScales();
  const auto zero_points = quant_params.GetZeroPoints();
  const auto axis = quant_params.GetAxis();
  if (axis < 0 || static_cast<std::uint32_t>(axis) >= src.GetRank()) {
    QNN_LOG_ERROR("Invalid Conv2d filter quantization axis.");
    return nullptr;
  }

  const auto axis_index = static_cast<size_t>(axis);
  const auto &dims = src.GetDimensions();
  const auto channel_count = dims[axis_index];
  if (scales.size() != channel_count || zero_points.size() != channel_count) {
    QNN_LOG_ERROR("Conv2d filter quantization params do not match axis size.");
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

std::vector<OpWrapper>
BuildConv2dOpImp(TensorPool &tensor_pool, const TensorWrapper &input_tensor,
                 TensorWrapper &filter_tensor, TensorWrapper *bias_tensor,
                 const TensorWrapper &output_tensor,
                 const std::uint32_t stride_h, const std::uint32_t stride_w,
                 const std::uint32_t dilation_h, const std::uint32_t dilation_w,
                 const PaddingType padding_type, bool use_int64_bias_as_int32) {
  std::vector<OpWrapper> res;

  // transpose filter
  const std::vector<uint32_t> &filters_dims = filter_tensor.GetDimensions();
  auto &filter_quant_params = filter_tensor.GetQuantParams();
  std::vector<std::uint32_t> permute_dims{filters_dims[1], filters_dims[2],
                                          filters_dims[3], filters_dims[0]};
  if (std::holds_alternative<AxisScaleOffsetQuantizeParamsWrapper>(
          filter_quant_params)) {
    auto &axis_quant_params =
        std::get<AxisScaleOffsetQuantizeParamsWrapper>(filter_quant_params);
    const std::array<std::int32_t, 4> new_axis{3, 0, 1, 2};
    axis_quant_params.SetAxis(new_axis[axis_quant_params.GetAxis()]);
  }

  size_t filter_bytes = filter_tensor.GetTensorBytes();
  TensorWrapper *transposed_filter_tensor = nullptr;
  if (filter_tensor.IsTensorStatic() &&
      filter_tensor.GetDataType() ==
          Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_8) {
    auto filter_data = filter_tensor.GetTensorData<std::int8_t>();
    std::vector<int8_t> transpose_weight_int8;
    TransposeFromOHWIToHWIO(filter_data.value(), filters_dims,
                            transpose_weight_int8);
    transposed_filter_tensor = &(tensor_pool.CreateStaticTensor(
        filter_tensor.GetDataType(), filter_quant_params, permute_dims,
        filter_bytes, transpose_weight_int8.data()));
  } else if (filter_tensor.IsTensorStatic() &&
             filter_tensor.GetDataType() ==
                 Qnn_DataType_t::QNN_DATATYPE_UFIXED_POINT_8) {
    auto filter_data = filter_tensor.GetTensorData<std::uint8_t>();
    std::vector<uint8_t> transpose_weight_uint8;
    TransposeFromOHWIToHWIO(filter_data.value(), filters_dims,
                            transpose_weight_uint8);
    transposed_filter_tensor = &(tensor_pool.CreateStaticTensor(
        filter_tensor.GetDataType(), filter_quant_params, permute_dims,
        filter_bytes, transpose_weight_uint8.data()));
  } else if (filter_tensor.IsTensorStatic() && filter_tensor.IsF32()) {
    auto filter_data = filter_tensor.GetTensorData<float>();
    if (!filter_data.has_value()) {
      QNN_LOG_ERROR("Failed to get Conv2d FP32 filter tensor data.");
      return {};
    }
    std::vector<float> transpose_weight_float;
    TransposeFromOHWIToHWIO(filter_data.value(), filters_dims,
                            transpose_weight_float);
    transposed_filter_tensor = &(tensor_pool.CreateStaticTensor(
        filter_tensor.GetDataType(), filter_quant_params, permute_dims,
        sizeof(float) * transpose_weight_float.size(),
        transpose_weight_float.data()));
  } else {
    transposed_filter_tensor =
        &(tensor_pool.CloneNativeTensorFrom(filter_tensor, permute_dims));

    const std::vector<std::uint32_t> permute_shape{4};
    const std::array<std::uint32_t, 4> permute_data{kHeightIndex, kWidthIndex,
                                                    kChannelIndex, kBatchIndex};
    auto &permute_tensor = tensor_pool.CreateStaticTensor(
        QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, permute_shape,
        sizeof(decltype(permute_data)::value_type) * permute_data.size(),
        permute_data.data());

    OpWrapper &transpose_op = CreateOpWrapper(res, QNN_OP_TRANSPOSE);
    transpose_op.AddInputTensor(filter_tensor);
    transpose_op.AddOutputTensor(*transposed_filter_tensor);
    transpose_op.AddTensorParam(QNN_OP_TRANSPOSE_PARAM_PERM, permute_tensor);
  }

  // conv
  OpWrapper &conv_op = CreateOpWrapper(res, QNN_OP_CONV_2D);
  conv_op.AddInputTensor(input_tensor);
  conv_op.AddInputTensor(*transposed_filter_tensor);
  if (bias_tensor != nullptr) {
    // QNN only support per-tensor quant for bias,
    // and the scale and offset are both zero.
    bias_tensor->ConvertAxisScaleOffsetToScaleOffset();

    if (use_int64_bias_as_int32 && bias_tensor->IsTensorStatic() &&
        bias_tensor->GetDataType() == QNN_DATATYPE_INT_64) {
      auto *converted_bias_tensor =
          tensor_pool.ConvertStaticTensorFrom<std::int32_t>(*bias_tensor);
      if (converted_bias_tensor == nullptr) {
        return {};
      }
      conv_op.AddInputTensor(*converted_bias_tensor);
      QNN_LOG_WARNING("Convert bias tensor in conv2d op from int64 to int32.");
    } else {
      conv_op.AddInputTensor(*bias_tensor);
    }
  }

  conv_op.AddOutputTensor(output_tensor);

  // stride param
  const std::array<std::uint32_t, 2> stride_data{stride_h, stride_w};
  const std::vector<std::uint32_t> stride_shape{2};
  auto &stride_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, stride_shape,
      sizeof(decltype(stride_data)::value_type) * stride_data.size(),
      stride_data.data());
  conv_op.AddTensorParam(QNN_OP_CONV_2D_PARAM_STRIDE, stride_tensor);

  // dilation param
  const std::array<std::uint32_t, 2> dilation_data{dilation_h, dilation_w};
  const std::vector<std::uint32_t> dilation_shape{2};
  auto &dilation_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, dilation_shape,
      sizeof(decltype(dilation_data)::value_type) * dilation_data.size(),
      dilation_data.data());
  conv_op.AddTensorParam(QNN_OP_CONV_2D_PARAM_DILATION, dilation_tensor);

  // padding param
  const auto [padding_before_height, padding_after_height] =
      ComputePaddingBeforeAfter(input_tensor.GetDimension(kHeightIndex),
                                filter_tensor.GetDimension(kHeightIndex),
                                stride_h, dilation_h, padding_type);
  const auto [padding_before_width, padding_after_width] =
      ComputePaddingBeforeAfter(input_tensor.GetDimension(kWidthIndex),
                                filter_tensor.GetDimension(kWidthIndex),
                                stride_w, dilation_w, padding_type);
  const std::array<std::uint32_t, 4> padding_data = {
      padding_before_height, padding_after_height, padding_before_width,
      padding_after_width};
  const std::vector<std::uint32_t> padding_shape{2, 2};
  auto &padding_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, padding_shape,
      sizeof(decltype(padding_data)::value_type) * padding_data.size(),
      padding_data.data());
  conv_op.AddTensorParam(QNN_OP_CONV_2D_PARAM_PAD_AMOUNT, padding_tensor);

  // group param
  if ((input_tensor.GetDimension(kChannelIndex) %
       filter_tensor.GetDimension(kChannelIndex)) != 0) {
    QNN_LOG_WARNING(
        "The channels of the input tensor cannot be evenly divided by the "
        "channels of the filter tensor.");
  }
  if (const std::uint32_t groups = input_tensor.GetDimension(kChannelIndex) /
                                   filter_tensor.GetDimension(kChannelIndex);
      groups > 1) {
    conv_op.AddScalarParam<std::uint32_t>(QNN_OP_CONV_2D_PARAM_GROUP, groups);
  }

  return res;
}

std::vector<OpWrapper> BuildFp32Conv2dOp(
    TensorPool &tensor_pool, const std::vector<TensorWrapperRef> &inputs,
    const std::vector<TensorWrapperRef> &outputs, const std::uint32_t stride_h,
    const std::uint32_t stride_w, const std::uint32_t dilation_h,
    const std::uint32_t dilation_w, const PaddingType padding_type,
    bool use_int64_bias_as_int32) {
  TensorWrapper *filter_tensor =
      DequantizePerChannelInt8StaticTensor(tensor_pool, inputs[kFilterIndex]);
  if (filter_tensor == nullptr) {
    return {};
  }

  TensorWrapper *bias_tensor =
      kBiasIndex < inputs.size() ? &inputs[kBiasIndex].get() : nullptr;
  return BuildConv2dOpImp(tensor_pool, inputs[kInputIndex].get(),
                          *filter_tensor, bias_tensor, outputs[kOutputIndex],
                          stride_h, stride_w, dilation_h, dilation_w,
                          padding_type, use_int64_bias_as_int32);
}

std::vector<OpWrapper> BuildFp16Conv2dOp(
    TensorPool &tensor_pool, const std::vector<TensorWrapperRef> &inputs,
    const std::vector<TensorWrapperRef> &outputs, const std::uint32_t stride_h,
    const std::uint32_t stride_w, const std::uint32_t dilation_h,
    const std::uint32_t dilation_w, const PaddingType padding_type,
    bool use_int64_bias_as_int32) {
  std::vector<OpWrapper> res;

  constexpr std::int32_t kFilterOutputChannelAxis = 0;
  TensorWrapper &filter_tensor = inputs[kFilterIndex];
  if (filter_tensor.IsPerTensorQuant()) {
    const auto &filter_quant = std::get<ScaleOffsetQuantizeParamsWrapper>(
        filter_tensor.GetQuantParams());
    QuantizeParamsWrapperVariant per_channel_quant{
        std::in_place_type<AxisScaleOffsetQuantizeParamsWrapper>,
        kFilterOutputChannelAxis,
        filter_tensor.GetDimension(kFilterOutputChannelAxis),
        filter_quant.GetScale(), filter_quant.GetZeroPoint()};
    filter_tensor.SetQuantParams(per_channel_quant);
  } else if (!filter_tensor.IsPerChannelQuant()) {
    QNN_LOG_ERROR(
        "FP16 Conv2d with int8 weight only supports per-channel quantized "
        "weight.");
    return {};
  }

  const TensorWrapper &input_tensor = inputs[kInputIndex];
  TensorWrapper &input_f16 = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_FLOAT_16, QuantizeParamsWrapperVariant{},
      input_tensor.GetDimensions());
  res.emplace_back(CreateCastOp(input_tensor, input_f16));

  TensorWrapper *bias_f16 = nullptr;
  if (kBiasIndex < inputs.size()) {
    const TensorWrapper &bias_tensor = inputs[kBiasIndex];
    TensorWrapper &bias_f16_tensor = tensor_pool.CreateNativeTensor(
        QNN_DATATYPE_FLOAT_16, QuantizeParamsWrapperVariant{},
        bias_tensor.GetDimensions());
    res.emplace_back(CreateCastOp(bias_tensor, bias_f16_tensor));
    bias_f16 = &bias_f16_tensor;
  }

  const TensorWrapper &output_tensor = outputs[kOutputIndex];
  TensorWrapper &output_f16 = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_FLOAT_16, QuantizeParamsWrapperVariant{},
      output_tensor.GetDimensions());

  auto conv_ops = BuildConv2dOpImp(
      tensor_pool, input_f16, filter_tensor, bias_f16, output_f16, stride_h,
      stride_w, dilation_h, dilation_w, padding_type, use_int64_bias_as_int32);
  std::move(conv_ops.begin(), conv_ops.end(), std::back_inserter(res));

  res.emplace_back(CreateCastOp(output_f16, output_tensor));
  return res;
}

} // namespace

std::vector<OpWrapper>
BuildConv2dOp(TensorPool &tensor_pool,
              const std::vector<TensorWrapperRef> &inputs,
              const std::vector<TensorWrapperRef> &outputs,
              const std::uint32_t stride_h, const std::uint32_t stride_w,
              const std::uint32_t dilation_h, const std::uint32_t dilation_w,
              const PaddingType padding_type, bool use_int64_bias_as_int32) {
  if (inputs.size() < 2 || outputs.empty()) {
    return {};
  }

  const bool has_bias = kBiasIndex < inputs.size();
  if (inputs[kInputIndex].get().IsF32() &&
      inputs[kFilterIndex].get().IsTensorStatic() &&
      inputs[kFilterIndex].get().IsQuantI8() &&
      inputs[kFilterIndex].get().IsPerChannelQuant() &&
      !IsOneByOneConv2d(inputs[kFilterIndex].get()) &&
      outputs[kOutputIndex].get().IsF32() &&
      (!has_bias || inputs[kBiasIndex].get().IsF32())) {
    return BuildFp32Conv2dOp(tensor_pool, inputs, outputs, stride_h, stride_w,
                             dilation_h, dilation_w, padding_type,
                             use_int64_bias_as_int32);
  }

  if (inputs[kInputIndex].get().IsF32() &&
      inputs[kFilterIndex].get().IsQuantI8() &&
      outputs[kOutputIndex].get().IsF32() &&
      (!has_bias || inputs[kBiasIndex].get().IsF32())) {
    return BuildFp16Conv2dOp(tensor_pool, inputs, outputs, stride_h, stride_w,
                             dilation_h, dilation_w, padding_type,
                             use_int64_bias_as_int32);
  }

  TensorWrapper *bias_tensor = has_bias ? &inputs[kBiasIndex].get() : nullptr;
  return BuildConv2dOpImp(tensor_pool, inputs[kInputIndex].get(),
                          inputs[kFilterIndex].get(), bias_tensor,
                          outputs[kOutputIndex], stride_h, stride_w, dilation_h,
                          dilation_w, padding_type, use_int64_bias_as_int32);
}

} // namespace qnn
