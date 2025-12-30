// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0


#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {

namespace {

// TODO(Alen): The current test coverage is not exhaustive.
// Some corner cases may not be tested. Narrowed types may lead to unexpected
// behavior.

TEST(TensorPoolConvertStaticTensorTest, ConvertNonStaticTensor) {
  TensorWrapper native_tensor;
  ASSERT_TRUE(CreateNativeTensor(native_tensor, "", QNN_DATATYPE_FLOAT_32,
                                 QuantizeParamsWrapperVariant{}, {1, 2, 3}));

  TensorWrapper static_tensor;
  ASSERT_FALSE(
      ConvertStaticTensorFrom<float>(static_tensor, "", native_tensor));
}

TEST(TensorPoolConvertStaticTensorTest, ExceedRangeAndFailToConvert) {
  std::vector<std::int32_t> tensor_data{
      std::numeric_limits<std::int32_t>::min(),
      std::numeric_limits<std::int32_t>::max()};
  TensorWrapper static_tensor;
  ASSERT_TRUE(CreateStaticTensor(
      static_tensor, "", QNN_DATATYPE_INT_32, QuantizeParamsWrapperVariant{},
      {2}, sizeof(decltype(tensor_data)::value_type) * tensor_data.size(),
      tensor_data.data()));

  TensorWrapper converted_tensor;
  ASSERT_FALSE(ConvertStaticTensorFrom<std::int16_t>(converted_tensor, "",
                                                     static_tensor));
}

TEST(TensorPoolConvertStaticTensorTest, SameTypeConversionFloat32) {
  std::vector<float> tensor_data{0, 1, 2, 3, 4, 5};

  TensorWrapper static_tensor;
  ASSERT_TRUE(CreateStaticTensor(
      static_tensor, "", QNN_DATATYPE_FLOAT_32, QuantizeParamsWrapperVariant{},
      {1, 2, 3}, sizeof(decltype(tensor_data)::value_type) * tensor_data.size(),
      tensor_data.data()));

  TensorWrapper converted_tensor;
  ASSERT_TRUE(
      ConvertStaticTensorFrom<float>(converted_tensor, "", static_tensor));

  auto converted_data = converted_tensor.GetTensorData<float>();
  ASSERT_TRUE(converted_data.has_value());

  ASSERT_EQ(tensor_data.size(), converted_data->size());
  for (size_t i = 0; i < tensor_data.size(); ++i) {
    ASSERT_FLOAT_EQ(tensor_data[i], (*converted_data)[i]);
  }
}

TEST(TensorPoolConvertStaticTensorTest, SameTypeConversionInt32) {
  std::vector<std::int32_t> tensor_data{0, 1, 2, 3, 4, 5};

  TensorWrapper static_tensor;
  ASSERT_TRUE(CreateStaticTensor(
      static_tensor, "", QNN_DATATYPE_INT_32, QuantizeParamsWrapperVariant{},
      {1, 2, 3}, sizeof(decltype(tensor_data)::value_type) * tensor_data.size(),
      tensor_data.data()));

  TensorWrapper converted_tensor;
  ASSERT_TRUE(ConvertStaticTensorFrom<std::int32_t>(converted_tensor, "",
                                                    static_tensor));

  auto converted_data = converted_tensor.GetTensorData<std::int32_t>();
  ASSERT_TRUE(converted_data.has_value());

  ASSERT_EQ(tensor_data.size(), converted_data->size());
  for (size_t i = 0; i < tensor_data.size(); ++i) {
    ASSERT_EQ(tensor_data[i], (*converted_data)[i]);
  }
}

TEST(TensorPoolConvertStaticTensorTest, ExpandTypeConversionFloat32) {
  std::vector<float> tensor_data{0, 1, 2, 3, 4, 5};

  TensorWrapper static_tensor;
  ASSERT_TRUE(CreateStaticTensor(
      static_tensor, "", QNN_DATATYPE_FLOAT_32, QuantizeParamsWrapperVariant{},
      {1, 2, 3}, sizeof(decltype(tensor_data)::value_type) * tensor_data.size(),
      tensor_data.data()));

  TensorWrapper converted_tensor;
  ASSERT_TRUE(
      ConvertStaticTensorFrom<double>(converted_tensor, "", static_tensor));

  auto converted_data = converted_tensor.GetTensorData<double>();
  ASSERT_TRUE(converted_data.has_value());

  ASSERT_EQ(tensor_data.size(), converted_data->size());
  for (size_t i = 0; i < tensor_data.size(); ++i) {
    ASSERT_DOUBLE_EQ(tensor_data[i], (*converted_data)[i]);
  }
}

TEST(TensorPoolConvertStaticTensorTest, ExpandTypeConversionInt32) {
  std::vector<std::int32_t> tensor_data{0, 1, 2, 3, 4, 5};

  TensorWrapper static_tensor;
  ASSERT_TRUE(CreateStaticTensor(
      static_tensor, "", QNN_DATATYPE_INT_32, QuantizeParamsWrapperVariant{},
      {1, 2, 3}, sizeof(decltype(tensor_data)::value_type) * tensor_data.size(),
      tensor_data.data()));

  TensorWrapper converted_tensor;
  ASSERT_TRUE(ConvertStaticTensorFrom<std::int64_t>(converted_tensor, "",
                                                    static_tensor));

  auto converted_data = converted_tensor.GetTensorData<std::int64_t>();
  ASSERT_TRUE(converted_data.has_value());

  ASSERT_EQ(tensor_data.size(), converted_data->size());
  for (size_t i = 0; i < tensor_data.size(); ++i) {
    ASSERT_EQ(tensor_data[i], (*converted_data)[i]);
  }
}

TEST(TensorPoolConvertStaticTensorTest, NarrowTypeConversionFloat32) {
  std::vector<double> tensor_data{0, 1, 2, 3, 4, 5};
  TensorWrapper static_tensor;
  ASSERT_TRUE(CreateStaticTensor(
      static_tensor, "", QNN_DATATYPE_FLOAT_64, QuantizeParamsWrapperVariant{},
      {1, 2, 3}, sizeof(decltype(tensor_data)::value_type) * tensor_data.size(),
      tensor_data.data()));

  TensorWrapper converted_tensor;
  ASSERT_TRUE(
      ConvertStaticTensorFrom<float>(converted_tensor, "", static_tensor));

  auto converted_data = converted_tensor.GetTensorData<float>();
  ASSERT_TRUE(converted_data.has_value());

  ASSERT_EQ(tensor_data.size(), converted_data->size());
  for (size_t i = 0; i < tensor_data.size(); ++i) {
    ASSERT_DOUBLE_EQ(tensor_data[i], (*converted_data)[i]);
  }
}

TEST(TensorPoolConvertStaticTensorTest, NarrowTypeConversionInt32) {
  std::vector<std::int64_t> tensor_data{0, 1, 2, 3, 4, 5};

  TensorWrapper static_tensor;
  ASSERT_TRUE(CreateStaticTensor(
      static_tensor, "", QNN_DATATYPE_INT_64, QuantizeParamsWrapperVariant{},
      {1, 2, 3}, sizeof(decltype(tensor_data)::value_type) * tensor_data.size(),
      tensor_data.data()));

  TensorWrapper converted_tensor;
  ASSERT_TRUE(ConvertStaticTensorFrom<std::int32_t>(converted_tensor, "",
                                                    static_tensor));

  auto converted_data = converted_tensor.GetTensorData<std::int32_t>();
  ASSERT_TRUE(converted_data.has_value());

  ASSERT_EQ(tensor_data.size(), converted_data->size());
  for (size_t i = 0; i < tensor_data.size(); ++i) {
    ASSERT_EQ(tensor_data[i], (*converted_data)[i]);
  }
}

TEST(TensorPoolConvertStaticTensorTest, CreateStatictensorByValueFloat) {
  std::vector<float> golden_data = {6.0f, 6.0f, 6.0f};

  TensorWrapper static_tensor;
  ASSERT_TRUE(CreateStaticTensorWithValue(
      static_tensor, "", QNN_DATATYPE_FLOAT_32, {}, {1, 1, 3}, 6));

  const auto tensor_data = static_tensor.GetTensorData<float>();

  EXPECT_TRUE(tensor_data.has_value());
  EXPECT_EQ(tensor_data, golden_data);
}

TEST(TensorPoolConvertStaticTensorTest, CreateStatictensorByValueInt8) {
  ScaleOffsetQuantizeParamsWrapper q_param(2, -5);  // offset = 5

  std::vector<float> golden_data = {2.0, 2.0, 2.0};

  TensorWrapper static_tensor;
  ASSERT_TRUE(CreateStaticTensorWithValue(
      static_tensor, "", QNN_DATATYPE_SFIXED_POINT_8, q_param, {1, 1, 3}, 2));

  const auto& q_param_ref = static_tensor.GetQuantParams();
  const float scale =
      std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetScale();
  const std::int32_t zero_point =
      std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetZeroPoint();

  const auto tensor_data = static_tensor.GetTensorData<std::int8_t>();

  EXPECT_TRUE(tensor_data.has_value());

  // Dequantize each element from the tensor data.
  for (int i = 0; i < golden_data.size(); i++) {
    EXPECT_NEAR(Dequantize((*tensor_data)[i], scale, zero_point),
                golden_data[i], 1e-7);
  }
}

TEST(TensorPoolConvertStaticTensorTest, CreateStatictensorByValueUInt8) {
  ScaleOffsetQuantizeParamsWrapper q_param(2, 5);  // offset = -5

  std::vector<float> golden_data = {2.0, 2.0, 2.0};

  TensorWrapper static_tensor;
  ASSERT_TRUE(CreateStaticTensorWithValue(
      static_tensor, "", QNN_DATATYPE_UFIXED_POINT_8, q_param, {1, 1, 3}, 2));

  const auto& q_param_ref = static_tensor.GetQuantParams();
  const float scale =
      std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetScale();
  const std::int32_t zero_point =
      std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetZeroPoint();

  const auto tensor_data = static_tensor.GetTensorData<std::uint8_t>();

  EXPECT_TRUE(tensor_data.has_value());

  // Dequantize each element from the tensor data.
  for (int i = 0; i < golden_data.size(); i++) {
    EXPECT_NEAR(Dequantize((*tensor_data)[i], scale, zero_point),
                golden_data[i], 1e-7);
  }
}

TEST(TensorPoolConvertStaticTensorTest, CreateStatictensorByValueInt16) {
  ScaleOffsetQuantizeParamsWrapper q_param(2, -5);  // offset = 5

  std::vector<float> golden_data = {2.0, 2.0, 2.0};

  TensorWrapper static_tensor;
  ASSERT_TRUE(CreateStaticTensorWithValue(
      static_tensor, "", QNN_DATATYPE_SFIXED_POINT_16, q_param, {1, 1, 3}, 2));

  const auto& q_param_ref = static_tensor.GetQuantParams();
  const float scale =
      std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetScale();
  const std::int32_t zero_point =
      std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetZeroPoint();

  const auto tensor_data = static_tensor.GetTensorData<std::int16_t>();

  EXPECT_TRUE(tensor_data.has_value());

  // Dequantize each element from the tensor data.
  for (int i = 0; i < golden_data.size(); i++) {
    EXPECT_NEAR(Dequantize((*tensor_data)[i], scale, zero_point),
                golden_data[i], 1e-7);
  }
}

TEST(TensorPoolConvertStaticTensorTest, CreateStatictensorByValueUInt16) {
  ScaleOffsetQuantizeParamsWrapper q_param(2, 5);  // offset = -5

  std::vector<float> golden_data = {2.0, 2.0, 2.0};

  TensorWrapper static_tensor;
  ASSERT_TRUE(CreateStaticTensorWithValue(
      static_tensor, "", QNN_DATATYPE_UFIXED_POINT_16, q_param, {1, 1, 3}, 2));

  const auto& q_param_ref = static_tensor.GetQuantParams();
  const float scale =
      std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetScale();
  const std::int32_t zero_point =
      std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetZeroPoint();

  const auto tensor_data = static_tensor.GetTensorData<std::uint16_t>();

  EXPECT_TRUE(tensor_data.has_value());

  // Dequantize each element from the tensor data.
  for (int i = 0; i < golden_data.size(); i++) {
    EXPECT_NEAR(Dequantize((*tensor_data)[i], scale, zero_point),
                golden_data[i], 1e-7);
  }
}

TEST(TensorPoolConvertStaticTensorTest, CreateStatictensorByValueSFixInt32) {
  ScaleOffsetQuantizeParamsWrapper q_param(2, -5);  // offset = 5

  std::vector<float> golden_data = {2.0, 2.0, 2.0};

  TensorWrapper static_tensor;
  ASSERT_TRUE(CreateStaticTensorWithValue(static_tensor, "",
                                          QNN_DATATYPE_SFIXED_POINT_32, q_param,
                                          {1, 1, 3}, 2.0));

  const auto& q_param_ref = static_tensor.GetQuantParams();
  const float scale =
      std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetScale();
  const std::int32_t zero_point =
      std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetZeroPoint();

  const auto tensor_data = static_tensor.GetTensorData<std::int32_t>();

  EXPECT_TRUE(tensor_data.has_value());

  // Dequantize each element from the tensor data.
  for (int i = 0; i < golden_data.size(); i++) {
    EXPECT_NEAR(Dequantize((*tensor_data)[i], scale, zero_point),
                golden_data[i], 1e-7);
  }
}

TEST(TensorPoolConvertStaticTensorTest, CreateStatictensorByValueUFixInt32) {
  ScaleOffsetQuantizeParamsWrapper q_param(2, 5);  // offset = -5

  std::vector<float> golden_data = {2, 2, 2};

  TensorWrapper static_tensor;
  ASSERT_TRUE(CreateStaticTensorWithValue(static_tensor, "",
                                          QNN_DATATYPE_UFIXED_POINT_32, q_param,
                                          {1, 1, 3}, 2.0));

  const auto& q_param_ref = static_tensor.GetQuantParams();
  const float scale =
      std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetScale();
  const std::int32_t zero_point =
      std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetZeroPoint();

  const auto tensor_data = static_tensor.GetTensorData<std::uint32_t>();

  EXPECT_TRUE(tensor_data.has_value());

  // Dequantize each element from the tensor data.
  for (int i = 0; i < golden_data.size(); i++) {
    EXPECT_NEAR(Dequantize((*tensor_data)[i], scale, zero_point),
                golden_data[i], 1e-7);
  }
}

TEST(TensorPoolConvertStaticTensorTest, CreateStatictensorByValueInt32) {
  std::vector<std::int32_t> golden_data = {2, 2, 2};

  TensorWrapper static_tensor;
  ASSERT_TRUE(CreateStaticTensorWithValue(
      static_tensor, "", QNN_DATATYPE_INT_32, {}, {1, 1, 3}, 2.0));

  const auto tensor_data = static_tensor.GetTensorData<std::int32_t>();

  EXPECT_TRUE(tensor_data.has_value());

  EXPECT_EQ(tensor_data, golden_data);
}

TEST(TensorPoolConvertStaticTensorTest, CreateStatictensorByValueUInt32) {
  std::vector<std::uint32_t> golden_data = {2, 2, 2};

  TensorWrapper static_tensor;
  ASSERT_TRUE(CreateStaticTensorWithValue(
      static_tensor, "", QNN_DATATYPE_UINT_32, {}, {1, 1, 3}, 2.0));

  const auto tensor_data = static_tensor.GetTensorData<std::uint32_t>();

  EXPECT_TRUE(tensor_data.has_value());

  EXPECT_EQ(tensor_data, golden_data);
}

}  // namespace

}  // namespace qnn
