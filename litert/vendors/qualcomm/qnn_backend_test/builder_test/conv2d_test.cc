// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include "QnnTypes.h" // from @qairt
#include "litert/vendors/qualcomm/core/builders/conv2d_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {
using testing::ElementsAre; // NOLINT
using testing::FloatNear;   // NOLINT

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

TEST_P(QnnModelTest, Conv2dFp32CastsAroundFp16Conv2d) {
  static constexpr float kWeightScale = 1.0f;
  static constexpr std::int32_t kWeightZeroPoint = 0;
  static constexpr std::array<std::int8_t, 4> kWeightData{1, 0, 0, 1};
  static constexpr std::array<float, 2> kBiasData{0.5f, -0.5f};
  static constexpr std::uint32_t kStride = 1;
  static constexpr std::uint32_t kDilation = 1;
  static constexpr bool kUseInt64BiasAsInt32 = false;
  const std::vector<std::uint32_t> kInOutDims{1, 1, 1, 2};
  const std::vector<std::uint32_t> kFilterDims{2, 1, 1, 2};
  const std::vector<std::uint32_t> kTransposedFilterDims{1, 1, 2, 2};
  const std::vector<std::uint32_t> kBiasDims{2};

  ::qnn::QuantizeParamsWrapperVariant weight_quant_param{
      std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, kWeightScale,
      kWeightZeroPoint};

  auto &input = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_FLOAT_32, {}, kInOutDims);
  auto &weight = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, weight_quant_param, kFilterDims,
      kWeightData.size() * sizeof(decltype(kWeightData)::value_type),
      kWeightData.data());
  auto &bias = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, {}, kBiasDims,
      kBiasData.size() * sizeof(decltype(kBiasData)::value_type),
      kBiasData.data());
  auto &output = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_FLOAT_32, {}, kInOutDims);

  auto ops = ::qnn::BuildConv2dOp(
      tensor_pool_, {input, weight, bias}, {output}, kStride, kStride,
      kDilation, kDilation, ::qnn::PaddingType::Valid, kUseInt64BiasAsInt32);

  ASSERT_EQ(ops.size(), 4u);
  EXPECT_TRUE(ops[0].IsOpCode(::qnn::QnnOpCode::kCast));
  EXPECT_TRUE(ops[0].GetInputTensor(0).IsF32());
  EXPECT_TRUE(ops[0].GetOutputTensor(0).IsF16());

  EXPECT_TRUE(ops[1].IsOpCode(::qnn::QnnOpCode::kCast));
  EXPECT_TRUE(ops[1].GetInputTensor(0).IsF32());
  EXPECT_TRUE(ops[1].GetOutputTensor(0).IsF16());

  ASSERT_TRUE(ops[2].IsOpCode(::qnn::QnnOpCode::kConv2d));
  ASSERT_EQ(ops[2].GetInputCount(), 3u);
  EXPECT_TRUE(ops[2].GetInputTensor(0).IsF16());
  EXPECT_TRUE(ops[2].GetInputTensor(1).IsQuantI8());
  EXPECT_TRUE(ops[2].GetInputTensor(1).IsPerChannelQuant());
  EXPECT_EQ(ops[2].GetInputTensor(1).GetDimensions(), kTransposedFilterDims);
  const auto &filter_quant =
      std::get<::qnn::AxisScaleOffsetQuantizeParamsWrapper>(
          ops[2].GetInputTensor(1).GetQuantParams());
  EXPECT_EQ(filter_quant.GetAxis(), 3);
  EXPECT_TRUE(ops[2].GetInputTensor(2).IsF16());
  EXPECT_TRUE(ops[2].GetOutputTensor(0).IsF16());
  EXPECT_EQ(ops[2].GetOutputTensor(0).GetDimensions(), kInOutDims);

  EXPECT_TRUE(ops[3].IsOpCode(::qnn::QnnOpCode::kCast));
  EXPECT_TRUE(ops[3].GetInputTensor(0).IsF16());
  EXPECT_TRUE(ops[3].GetOutputTensor(0).IsF32());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  //   if (!is_fp16_supported_) {
  //     GTEST_SKIP() << "The rest of this test applies only to HTP targets with
  //     "
  //                     "FP16 support.";
  //   }
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#else
  auto input_idx = qnn_model_.AddInputTensor(input);
  auto output_idx = qnn_model_.AddOutputTensor(output);

  qnn_model_.SetInputData<float>(input_idx, {1, 2});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 2);
  EXPECT_THAT(output_data.value(),
              ElementsAre(FloatNear(1.5f, 1e-3f), FloatNear(1.5f, 1e-3f)));
#endif
}

TEST_P(QnnModelTest, Conv2dFp32DequantizesPerChannelInt8WeightForNon1x1) {
  static constexpr std::int32_t kWeightAxis = 0;
  static constexpr std::uint32_t kOutputChannels = 2;
  static constexpr std::uint32_t kKernelSize = 2;
  static constexpr std::uint32_t kStride = 1;
  static constexpr std::uint32_t kDilation = 1;
  static constexpr bool kUseInt64BiasAsInt32 = false;
  static constexpr std::array<std::int8_t, 8> kWeightData{2, 4, 6, 8,
                                                          3, 5, 7, 9};
  static constexpr std::array<float, 2> kBiasData{0.5f, -0.5f};
  const std::vector<std::uint32_t> kInDims{1, 2, 2, 1};
  const std::vector<std::uint32_t> kOutDims{1, 2, 2, kOutputChannels};
  const std::vector<std::uint32_t> kFilterDims{kOutputChannels, kKernelSize,
                                               kKernelSize, 1};
  const std::vector<std::uint32_t> kTransposedFilterDims{
      kKernelSize, kKernelSize, 1, kOutputChannels};
  const std::vector<float> weight_scales{0.5f, 0.25f};
  const std::vector<std::int32_t> weight_zero_points{1, -1};
  ::qnn::QuantizeParamsWrapperVariant weight_quant{
      std::in_place_type<::qnn::AxisScaleOffsetQuantizeParamsWrapper>,
      kWeightAxis, weight_scales, weight_zero_points};

  auto &input = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_FLOAT_32, {}, kInDims);
  auto &weight = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, weight_quant, kFilterDims,
      kWeightData.size() * sizeof(decltype(kWeightData)::value_type),
      kWeightData.data());
  auto &bias = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, {}, {kOutputChannels},
      kBiasData.size() * sizeof(decltype(kBiasData)::value_type),
      kBiasData.data());
  auto &output = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_FLOAT_32, {}, kOutDims);

  auto ops = ::qnn::BuildConv2dOp(
      tensor_pool_, {input, weight, bias}, {output}, kStride, kStride,
      kDilation, kDilation, ::qnn::PaddingType::Same, kUseInt64BiasAsInt32);

  ASSERT_EQ(ops.size(), 1u);
  ASSERT_TRUE(ops[0].IsOpCode(::qnn::QnnOpCode::kConv2d));
  ASSERT_EQ(ops[0].GetInputCount(), 3u);
  EXPECT_TRUE(ops[0].GetInputTensor(0).IsF32());
  EXPECT_TRUE(ops[0].GetInputTensor(1).IsF32());
  EXPECT_FALSE(ops[0].GetInputTensor(1).IsQuant());
  EXPECT_TRUE(ops[0].GetInputTensor(2).IsF32());
  EXPECT_TRUE(ops[0].GetOutputTensor(0).IsF32());
  EXPECT_EQ(ops[0].GetInputTensor(1).GetDimensions(), kTransposedFilterDims);

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
