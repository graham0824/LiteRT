// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/spatial_transform_nd_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {
using testing::ElementsAre;

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

// Mirrors tflite/kernels/batch_to_space_nd_test.cc SimpleConstTest:
//   input shape {4, 2, 2, 1}, block_shape=[2, 2], crops=[0, 0, 0, 0]
//   output shape {1, 4, 4, 1}
TEST_P(QnnModelTest, BatchToSpaceNdSimple) {
  const std::vector<std::uint32_t> kInputDims{4, 2, 2, 1};
  const std::vector<std::uint32_t> kOutputDims{1, 4, 4, 1};
  static constexpr std::array<std::int32_t, 2> kBlockShapeData{2, 2};
  static constexpr std::array<std::int32_t, 4> kCropsData{0, 0, 0, 0};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_32, {}, kInputDims);
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_32, {}, kOutputDims);
  auto& block_shape_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {2},
      sizeof(std::int32_t) * kBlockShapeData.size(), kBlockShapeData.data());
  auto& crops_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {2, 2},
      sizeof(std::int32_t) * kCropsData.size(), kCropsData.data());

  auto ops = ::qnn::BuildBatchToSpaceNdOp(
      tensor_pool_, {input_tensor, block_shape_tensor, crops_tensor},
      {output_tensor});
  ASSERT_FALSE(ops.empty());
  EXPECT_TRUE(ops.front().IsOpCode(::qnn::QnnOpCode::kBatchToSpace));

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
  qnn_model_.SetInputData<float>(
      input_idx, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                  11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  EXPECT_THAT(output_data.value(),
              ElementsAre(1.0f, 5.0f, 2.0f, 6.0f, 9.0f, 13.0f, 10.0f, 14.0f,
                          3.0f, 7.0f, 4.0f, 8.0f, 11.0f, 15.0f, 12.0f, 16.0f));
}

// Mirrors tflite/kernels/space_to_batch_nd_test.cc SimpleConstTest:
//   input shape {1, 4, 4, 1}, block_shape=[2, 2], paddings=[0, 0, 0, 0]
//   output shape {4, 2, 2, 1}
TEST_P(QnnModelTest, SpaceToBatchNdSimple) {
  const std::vector<std::uint32_t> kInputDims{1, 4, 4, 1};
  const std::vector<std::uint32_t> kOutputDims{4, 2, 2, 1};
  static constexpr std::array<std::int32_t, 2> kBlockShapeData{2, 2};
  static constexpr std::array<std::int32_t, 4> kPaddingsData{0, 0, 0, 0};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_32, {}, kInputDims);
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_32, {}, kOutputDims);
  auto& block_shape_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {2},
      sizeof(std::int32_t) * kBlockShapeData.size(), kBlockShapeData.data());
  auto& paddings_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {2, 2},
      sizeof(std::int32_t) * kPaddingsData.size(), kPaddingsData.data());

  auto ops = ::qnn::BuildSpaceToBatchNdOp(
      tensor_pool_, {input_tensor, block_shape_tensor, paddings_tensor},
      {output_tensor});
  ASSERT_FALSE(ops.empty());
  EXPECT_TRUE(ops.front().IsOpCode(::qnn::QnnOpCode::kSpaceToBatch));

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
  qnn_model_.SetInputData<float>(
      input_idx, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                  11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  EXPECT_THAT(output_data.value(),
              ElementsAre(1.0f, 3.0f, 9.0f, 11.0f, 2.0f, 4.0f, 10.0f, 12.0f,
                          5.0f, 7.0f, 13.0f, 15.0f, 6.0f, 8.0f, 14.0f, 16.0f));
}

// The builder requires a static block_shape tensor; a non-static one is
// rejected with an empty op list.
TEST_P(QnnModelTest, BatchToSpaceNdRejectsNonStaticBlockShape) {
  const std::vector<std::uint32_t> kInputDims{4, 2, 2, 1};
  const std::vector<std::uint32_t> kOutputDims{1, 4, 4, 1};
  static constexpr std::array<std::int32_t, 4> kCropsData{0, 0, 0, 0};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_32, {}, kInputDims);
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_32, {}, kOutputDims);
  // block_shape as a non-static (input) tensor must be rejected.
  auto& block_shape_tensor = tensor_pool_.CreateInputTensorWithName(
      "block_shape", QNN_DATATYPE_INT_32, {}, {2});
  auto& crops_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {2, 2},
      sizeof(std::int32_t) * kCropsData.size(), kCropsData.data());

  auto ops = ::qnn::BuildBatchToSpaceNdOp(
      tensor_pool_, {input_tensor, block_shape_tensor, crops_tensor},
      {output_tensor});
  EXPECT_TRUE(ops.empty());
}

}  // namespace
}  // namespace litert::qnn
