// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include "QnnTypes.h" // from @qairt
#include "litert/vendors/qualcomm/core/builders/depthwise_conv2d_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"
#include <gtest/gtest.h>

namespace litert::qnn {
namespace {

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

TEST_P(QnnModelTest, QuantizedDepthwiseConv2dKernel4x4Stride4UsesConv2d) {
  static constexpr std::uint32_t kBatch = 1;
  static constexpr std::uint32_t kInputHeight = 4;
  static constexpr std::uint32_t kInputWidth = 4;
  static constexpr std::uint32_t kChannels = 2;
  static constexpr std::uint32_t kKernelSize = 4;
  static constexpr std::uint32_t kStride = 4;
  static constexpr std::uint32_t kDilationHeight = 1;
  static constexpr std::uint32_t kDilationWidth = 1;
  static constexpr std::uint32_t kOutputHeight = 1;
  static constexpr std::uint32_t kOutputWidth = 1;
  static constexpr std::array<std::int32_t, kChannels> kBiasData{10, 20};

  static constexpr float kScale = 1.0f;
  static constexpr std::int32_t kZeroPoint = 0;
  const ::qnn::ScaleOffsetQuantizeParamsWrapper quant{kScale, kZeroPoint};

  auto &input = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_SFIXED_POINT_8, quant,
      {kBatch, kInputHeight, kInputWidth, kChannels});

  auto &output = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_SFIXED_POINT_8, quant,
      {kBatch, kOutputHeight, kOutputWidth, kChannels});

  std::vector<std::int8_t> weight_data(kBatch * kKernelSize * kKernelSize *
                                       kChannels);
  for (std::uint32_t i = 0; i < weight_data.size(); i += kChannels) {
    weight_data[i] = 2;
    weight_data[i + 1] = 3;
  }
  auto &weight = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, quant,
      {kBatch, kKernelSize, kKernelSize, kChannels},
      weight_data.size() * sizeof(decltype(weight_data)::value_type),
      weight_data.data());

  auto &bias = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_32, quant, {kChannels},
      kBiasData.size() * sizeof(decltype(kBiasData)::value_type),
      kBiasData.data());

  auto ops = ::qnn::BuildDepthwiseConv2dOp(
      tensor_pool_, {input, weight, bias}, {output}, kStride, kStride,
      kDilationHeight, kDilationWidth, ::qnn::PaddingType::Same);

  ASSERT_EQ(ops.size(), 1u);
  ASSERT_TRUE(ops[0].IsOpCode(::qnn::QnnOpCode::kConv2d));

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "Execution requires an on-device Qualcomm HTP.";
#else
  auto input_idx = qnn_model_.AddInputTensor(input);
  auto output_idx = qnn_model_.AddOutputTensor(output);
  std::vector<std::int8_t> input_data(input.GetTensorNumElements());
  for (std::uint32_t i = 0; i < input_data.size(); i += kChannels) {
    input_data[i] = 1;
    input_data[i + 1] = 2;
  }
  ASSERT_TRUE(qnn_model_.SetInputData<std::int8_t>(input_idx, input_data));

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<std::int8_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), output.GetTensorNumElements());
  EXPECT_NEAR(::qnn::Dequantize(output_data.value()[0], kScale, kZeroPoint),
              42.0f, kScale);
  EXPECT_NEAR(::qnn::Dequantize(output_data.value()[1], kScale, kZeroPoint),
              116.0f, kScale);
#endif
}

TEST_P(QnnModelTest, QuantizedDepthwiseConv2dKernel3x3Stride3UsesDepthwise) {
  static constexpr std::uint32_t kBatch = 1;
  static constexpr std::uint32_t kInputHeight = 3;
  static constexpr std::uint32_t kInputWidth = 3;
  static constexpr std::uint32_t kChannels = 2;
  static constexpr std::uint32_t kKernelSize = 3;
  static constexpr std::uint32_t kStride = 3;
  static constexpr std::uint32_t kDilationHeight = 1;
  static constexpr std::uint32_t kDilationWidth = 1;
  static constexpr std::uint32_t kOutputHeight = 1;
  static constexpr std::uint32_t kOutputWidth = 1;
  static constexpr std::array<std::int32_t, kChannels> kBiasData{10, 20};

  static constexpr float kScale = 1.0f;
  static constexpr std::int32_t kZeroPoint = 0;
  const ::qnn::ScaleOffsetQuantizeParamsWrapper quant{kScale, kZeroPoint};

  auto &input = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_SFIXED_POINT_8, quant,
      {kBatch, kInputHeight, kInputWidth, kChannels});

  auto &output = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_SFIXED_POINT_8, quant,
      {kBatch, kOutputHeight, kOutputWidth, kChannels});

  std::vector<std::int8_t> weight_data(kBatch * kKernelSize * kKernelSize *
                                       kChannels);
  for (std::uint32_t i = 0; i < weight_data.size(); i += kChannels) {
    weight_data[i] = 2;
    weight_data[i + 1] = 3;
  }
  auto &weight = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, quant,
      {kBatch, kKernelSize, kKernelSize, kChannels},
      weight_data.size() * sizeof(decltype(weight_data)::value_type),
      weight_data.data());

  auto &bias = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_32, quant, {kChannels},
      kBiasData.size() * sizeof(decltype(kBiasData)::value_type),
      kBiasData.data());

  auto ops = ::qnn::BuildDepthwiseConv2dOp(
      tensor_pool_, {input, weight, bias}, {output}, kStride, kStride,
      kDilationHeight, kDilationWidth, ::qnn::PaddingType::Same);

  ASSERT_EQ(ops.size(), 1u);
  ASSERT_TRUE(ops[0].IsOpCode(::qnn::QnnOpCode::kDepthWiseConv2d));

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "Execution requires an on-device Qualcomm HTP.";
#else
  auto input_idx = qnn_model_.AddInputTensor(input);
  auto output_idx = qnn_model_.AddOutputTensor(output);
  std::vector<std::int8_t> input_data(input.GetTensorNumElements());
  for (std::uint32_t i = 0; i < input_data.size(); i += kChannels) {
    input_data[i] = 1;
    input_data[i + 1] = 2;
  }
  ASSERT_TRUE(qnn_model_.SetInputData<std::int8_t>(input_idx, input_data));

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<std::int8_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), output.GetTensorNumElements());
  EXPECT_NEAR(::qnn::Dequantize(output_data.value()[0], kScale, kZeroPoint),
              28.0f, kScale);
  EXPECT_NEAR(::qnn::Dequantize(output_data.value()[1], kScale, kZeroPoint),
              74.0f, kScale);
#endif
}

TEST_P(QnnModelTest, Fp32DepthwiseConv2dDequantizesPerChannelInt8Weight) {
  static constexpr std::int32_t kWeightAxis = 3;
  static constexpr std::uint32_t kBatch = 1;
  static constexpr std::uint32_t kInputHeight = 2;
  static constexpr std::uint32_t kInputWidth = 2;
  static constexpr std::uint32_t kChannels = 2;
  static constexpr std::uint32_t kKernelSize = 2;
  static constexpr std::uint32_t kStride = 1;
  static constexpr std::uint32_t kDilation = 1;
  static constexpr std::array<std::int8_t, 8> kWeightData{2, 3, 4, 5,
                                                          6, 7, 8, 9};
  static constexpr std::array<float, 2> kBiasData{0.5f, -0.5f};
  const std::vector<float> weight_scales{0.5f, 0.25f};
  const std::vector<std::int32_t> weight_zero_points{1, -1};
  ::qnn::QuantizeParamsWrapperVariant weight_quant{
      std::in_place_type<::qnn::AxisScaleOffsetQuantizeParamsWrapper>,
      kWeightAxis, weight_scales, weight_zero_points};

  auto &input = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_FLOAT_32, ::qnn::QuantizeParamsWrapperVariant{},
      {kBatch, kInputHeight, kInputWidth, kChannels});
  auto &output = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_FLOAT_32, ::qnn::QuantizeParamsWrapperVariant{},
      {kBatch, kInputHeight, kInputWidth, kChannels});
  auto &weight = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, weight_quant,
      {kBatch, kKernelSize, kKernelSize, kChannels},
      kWeightData.size() * sizeof(decltype(kWeightData)::value_type),
      kWeightData.data());
  auto &bias = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, ::qnn::QuantizeParamsWrapperVariant{}, {kChannels},
      kBiasData.size() * sizeof(decltype(kBiasData)::value_type),
      kBiasData.data());

  auto ops = ::qnn::BuildDepthwiseConv2dOp(
      tensor_pool_, {input, weight, bias}, {output}, kStride, kStride,
      kDilation, kDilation, ::qnn::PaddingType::Same);

  ASSERT_EQ(ops.size(), 1u);
  ASSERT_TRUE(ops[0].IsOpCode(::qnn::QnnOpCode::kDepthWiseConv2d));
  ASSERT_EQ(ops[0].GetInputCount(), 3u);
  EXPECT_TRUE(ops[0].GetInputTensor(0).IsF32());
  EXPECT_TRUE(ops[0].GetInputTensor(1).IsF32());
  EXPECT_FALSE(ops[0].GetInputTensor(1).IsQuant());
  EXPECT_TRUE(ops[0].GetInputTensor(2).IsF32());
  EXPECT_TRUE(ops[0].GetOutputTensor(0).IsF32());
  EXPECT_EQ(ops[0].GetInputTensor(1).GetDimensions(),
            (std::vector<std::uint32_t>{kKernelSize, kKernelSize, kBatch,
                                        kChannels}));

  const auto filter_data = ops[0].GetInputTensor(1).GetTensorData<float>();
  ASSERT_TRUE(filter_data.has_value());
  ASSERT_EQ(filter_data->size(), kWeightData.size());
  EXPECT_NEAR((*filter_data)[0], 0.5f, 1e-6f);
  EXPECT_NEAR((*filter_data)[1], 1.0f, 1e-6f);
  EXPECT_NEAR((*filter_data)[6], 3.5f, 1e-6f);
  EXPECT_NEAR((*filter_data)[7], 2.5f, 1e-6f);

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());
}

} // namespace
} // namespace litert::qnn
