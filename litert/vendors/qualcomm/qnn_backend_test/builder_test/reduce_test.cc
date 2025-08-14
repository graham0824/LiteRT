// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "QnnTypes.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/reduce_op_builder.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

using testing::FloatNear;
using testing::Pointwise;
namespace litert::qnn {
namespace {

TEST_F(QnnModelTest, ReduceAll) {
  SetUpQnnModel(::qnn::Options(), "SM8650");

  auto& input_0 = tensor_pool_.CreateInputTensorWithSuffix(
      QNN_DATATYPE_BOOL_8, {}, {1, 1, 4, 2}, "");
  const std::vector<std::int32_t> kAxis{3};
  auto& input_1 = tensor_pool_.CreateStaticTensorWithSuffix(
      QNN_DATATYPE_INT_32, {}, {1}, "",
      sizeof(decltype(kAxis)::value_type) * kAxis.size(), kAxis.data());
  auto& output_0 = tensor_pool_.CreateOutpuTensorWithSuffix(QNN_DATATYPE_BOOL_8,
                                                            {}, {1, 1, 4}, "");
  auto ops = ::qnn::BuildReduceAllOp(tensor_pool_, {input_0, input_1},
                                     {output_0}, false);
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));

  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

  #if !defined(__ANDROID__)
    GTEST_SKIP() << "The rest of this test is specific to Android devices
    with a "
                    "Qualcomm HTP";
  #endif

  auto input_idx = qnn_model_.AddInputTensor(input_0);
  auto output_idx = qnn_model_.AddOutputTensor(output_0);

  qnn_model_.SetInputData<bool>(
      input_idx, {false, false, false, true, true, false, true, true});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<bool>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 4);
  ASSERT_THAT(output_data.value(), ::testing::ElementsAreArray({false, false, false, true}));
}

TEST_F(QnnModelTest, ReduceAny) {
  SetUpQnnModel(::qnn::Options(), "SM8650");

  auto& input_0 = tensor_pool_.CreateInputTensorWithSuffix(
      QNN_DATATYPE_BOOL_8, {}, {1, 1, 4, 2}, "");
  const std::vector<std::int32_t> kAxis{3};
  auto& input_1 = tensor_pool_.CreateStaticTensorWithSuffix(
      QNN_DATATYPE_INT_32, {}, {1}, "",
      sizeof(decltype(kAxis)::value_type) * kAxis.size(), kAxis.data());
  auto& output_0 = tensor_pool_.CreateOutpuTensorWithSuffix(QNN_DATATYPE_BOOL_8,
                                                            {}, {1, 1, 4}, "");
  auto ops = ::qnn::BuildReduceAnyOp(tensor_pool_, {input_0, input_1},
                                     {output_0}, false);
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));

  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

  #if !defined(__ANDROID__)
    GTEST_SKIP() << "The rest of this test is specific to Android devices
    with a "
                    "Qualcomm HTP";
  #endif

  auto input_idx = qnn_model_.AddInputTensor(input_0);
  auto output_idx = qnn_model_.AddOutputTensor(output_0);

  qnn_model_.SetInputData<bool>(
      input_idx, {false, false, false, true, true, false, true, true});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<bool>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 4);
  ASSERT_THAT(output_data.value(), ::testing::ElementsAreArray({false, true, true, true}));
}

}  // namespace
}  // namespace litert::qnn
