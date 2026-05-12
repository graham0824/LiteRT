// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/mha_to_sha.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/builders/cast_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/matmul_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/builders/pack_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/slice_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/softmax_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/split_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/transpose_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/unpack_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {
namespace {

// Returns a boolean indicating whether the output tensor of op1 at out_index is
// connected to the input tensor of op2 at in_index.
#define IS_CONNECTED(op1, out_index, op2, in_index)     \
  (ops[start_index + op1].GetOutputTensor(out_index) == \
   ops[start_index + op2].GetInputTensor(in_index))

constexpr size_t kMulIndex = 0;
constexpr size_t kTransposePrefillIndex = 1;
constexpr size_t kReshapePrefillIndex = 2;
constexpr size_t kMatMulK1Index = 1;
constexpr size_t kMatMulK2Index = 2;
constexpr size_t kConcatIndex = 3;
constexpr size_t kReshape0Index = 4;
constexpr size_t kAddIndex = 5;
constexpr size_t kReshape1Index = 6;
constexpr size_t kSoftmaxIndex = 7;
constexpr size_t kSlice1Index = 8;
constexpr size_t kSlice2Index = 9;
constexpr size_t kMatMulV1Index = 10;
constexpr size_t kMatMulV2Index = 11;
constexpr size_t kAdd2Index = 12;
constexpr size_t kReshape2Index = 13;
constexpr size_t kTranspose2Index = 14;
constexpr size_t kReshape3Index = 15;

// QNN Slice Param ranges in the form (begin, end, stride) for each axis. To
// set the 3rd axis "end" value, we need to access ranges[3 * 2 + 2 - 1 = 7].
constexpr size_t kSlice3rdAxisEndIndex = 7;

const TensorWrapper& BuildSingleSHA(
    std::vector<OpWrapper>& new_ops, TensorPool& tensor_pool,
    const TensorWrapper& sha_input, const TensorWrapper& mask, size_t num_heads,
    const OpWrapper& mul, const OpWrapper& matmul_k1,
    const OpWrapper& matmul_k2, const OpWrapper& concat, const OpWrapper& add_1,
    const OpWrapper& softmax, const OpWrapper& slice_1,
    const OpWrapper& slice_2, const OpWrapper& matmul_v1,
    const OpWrapper& matmul_v2, const OpWrapper& add_2) {
  // Mul
  const auto& mul_output = tensor_pool.CloneNativeTensorFrom(
      mul.GetOutputTensor(0), sha_input.GetDimensions());
  new_ops.emplace_back(
      CreateElementWiseMulOp(sha_input, mul.GetInputTensor(1), mul_output));

  // MatMul 1
  auto matmul_k1_output_dims = matmul_k1.GetOutputTensor(0).GetDimensions();
  matmul_k1_output_dims[2] /= num_heads;
  const auto& matmul_k1_output = tensor_pool.CloneNativeTensorFrom(
      matmul_k1.GetOutputTensor(0), matmul_k1_output_dims);
  const std::array<ConstTensorWrapperRef, 2> matmul_k1_inputs = {
      mul_output, matmul_k1.GetInputTensor(1)};
  const std::array<ConstTensorWrapperRef, 1> matmul_k1_outputs = {
      matmul_k1_output};
  new_ops.emplace_back(
      CreateOpWithSameParams(matmul_k1, matmul_k1_inputs, matmul_k1_outputs));

  // MatMul 2
  auto matmul_k2_output_dims = matmul_k2.GetOutputTensor(0).GetDimensions();
  matmul_k2_output_dims[2] /= num_heads;
  const auto& matmul_k2_output = tensor_pool.CloneNativeTensorFrom(
      matmul_k2.GetOutputTensor(0), matmul_k2_output_dims);
  const std::array<ConstTensorWrapperRef, 2> matmul_k2_inputs = {
      mul_output, matmul_k2.GetInputTensor(1)};
  const std::array<ConstTensorWrapperRef, 1> matmul_k2_outputs = {
      matmul_k2_output};
  new_ops.emplace_back(
      CreateOpWithSameParams(matmul_k2, matmul_k2_inputs, matmul_k2_outputs));

  // Concat
  auto concat_output_dims = matmul_k1_output_dims;
  concat_output_dims[3] += matmul_k2_output_dims[3];
  const auto& concat_output = tensor_pool.CloneNativeTensorFrom(
      concat.GetOutputTensor(0), concat_output_dims);
  const std::array<ConstTensorWrapperRef, 2> concat_inputs = {matmul_k1_output,
                                                              matmul_k2_output};
  const std::array<ConstTensorWrapperRef, 1> concat_outputs = {concat_output};
  new_ops.emplace_back(
      CreateOpWithSameParams(concat, concat_inputs, concat_outputs));

  // Add
  const auto& add_1_output = tensor_pool.CloneNativeTensorFrom(
      add_1.GetOutputTensor(0), concat_output.GetDimensions());
  new_ops.emplace_back(
      CreateElementWiseAddOp(concat_output, mask, add_1_output));
  // Softmax
  const auto& softmax_output = tensor_pool.CloneNativeTensorFrom(
      softmax.GetOutputTensor(0), add_1_output.GetDimensions());
  const std::array<ConstTensorWrapperRef, 1> softmax_inputs = {add_1_output};
  const std::array<ConstTensorWrapperRef, 1> softmax_outputs = {softmax_output};
  new_ops.emplace_back(
      CreateOpWithSameParams(softmax, softmax_inputs, softmax_outputs));

  // Slice 1
  auto slice_1_ranges = slice_1.GetTensorParam(0).GetTensor();
  auto slice_1_rangs_data = slice_1_ranges.GetTensorData<int32_t>();
  std::vector<int32_t> sha_slice_1_ranges_data(
      slice_1_rangs_data.value().begin(), slice_1_rangs_data.value().end());
  sha_slice_1_ranges_data[kSlice3rdAxisEndIndex] /= num_heads;
  const auto& sha_slice_1_ranges = tensor_pool.CreateStaticTensor(
      slice_1_ranges.GetDataType(), slice_1_ranges.GetQuantParams(),
      slice_1_ranges.GetDimensions(), slice_1_ranges.GetTensorBytes(),
      sha_slice_1_ranges_data.data());
  auto slice_1_output_dims = slice_1.GetOutputTensor(0).GetDimensions();
  slice_1_output_dims[2] /= num_heads;
  const auto& slice_1_output = tensor_pool.CloneNativeTensorFrom(
      slice_1.GetOutputTensor(0), slice_1_output_dims);
  new_ops.emplace_back(
      CreateSliceOp(softmax_output, slice_1_output, sha_slice_1_ranges));

  // Slice 2
  auto slice_2_ranges = slice_2.GetTensorParam(0).GetTensor();
  auto slice_2_ranges_data = slice_2_ranges.GetTensorData<int32_t>();
  std::vector<int32_t> sha_slice_2_ranges_data(
      slice_2_ranges_data.value().begin(), slice_2_ranges_data.value().end());
  sha_slice_2_ranges_data[kSlice3rdAxisEndIndex] /= num_heads;
  const auto& sha_slice_2_ranges = tensor_pool.CreateStaticTensor(
      slice_2_ranges.GetDataType(), slice_2_ranges.GetQuantParams(),
      slice_2_ranges.GetDimensions(), slice_2_ranges.GetTensorBytes(),
      sha_slice_2_ranges_data.data());
  auto slice_2_output_dims = slice_2.GetOutputTensor(0).GetDimensions();
  slice_2_output_dims[2] /= num_heads;
  const auto& slice_2_output = tensor_pool.CloneNativeTensorFrom(
      slice_2.GetOutputTensor(0), slice_2_output_dims);
  new_ops.emplace_back(
      CreateSliceOp(softmax_output, slice_2_output, sha_slice_2_ranges));

  // MatMul 1
  std::vector<uint32_t> matmul_v1_output_dims =
      matmul_v1.GetOutputTensor(0).GetDimensions();
  matmul_v1_output_dims[2] = matmul_v1_output_dims[2] / num_heads;
  const auto& matmul_v1_output = tensor_pool.CloneNativeTensorFrom(
      matmul_v1.GetOutputTensor(0), matmul_v1_output_dims);
  const std::array<ConstTensorWrapperRef, 2> matmul_v1_inputs = {
      slice_1_output, matmul_v1.GetInputTensor(1)};
  const std::array<ConstTensorWrapperRef, 1> matmul_v1_outputs = {
      matmul_v1_output};
  new_ops.emplace_back(
      CreateOpWithSameParams(matmul_v1, matmul_v1_inputs, matmul_v1_outputs));

  // MatMul 2
  std::vector<uint32_t> matmul_v2_output_dims =
      matmul_v2.GetOutputTensor(0).GetDimensions();
  matmul_v2_output_dims[2] = matmul_v2_output_dims[2] / num_heads;
  const auto& matmul_v2_output = tensor_pool.CloneNativeTensorFrom(
      matmul_v2.GetOutputTensor(0), matmul_v2_output_dims);
  const std::array<ConstTensorWrapperRef, 2> matmul_v2_inputs = {
      slice_2_output, matmul_v2.GetInputTensor(1)};
  const std::array<ConstTensorWrapperRef, 1> matmul_v2_outputs = {
      matmul_v2_output};
  new_ops.emplace_back(
      CreateOpWithSameParams(matmul_v2, matmul_v2_inputs, matmul_v2_outputs));

  // Add 2
  const auto& add_2_output = tensor_pool.CloneNativeTensorFrom(
      add_2.GetOutputTensor(0), matmul_v1_output.GetDimensions());
  new_ops.emplace_back(
      CreateElementWiseAddOp(matmul_v1_output, matmul_v2_output, add_2_output));
  return add_2_output;
}

void CloneNamespace(const OpWrapper& source, OpWrapper& destination,
                    absl::string_view group_name = {}) {
  absl::string_view start_op_name = source.GetName();
  size_t pos = start_op_name.rfind('/');
  if (pos == absl::string_view::npos) {
    return;
  }
  if (group_name.empty())
    destination.AddPrefixToName(
        absl::StrCat(start_op_name.substr(0, pos), "/"));
  else
    destination.AddPrefixToName(
        absl::StrCat(start_op_name.substr(0, pos), "/", group_name, "/"));
}

void CloneNamespace(const OpWrapper& source, std::vector<OpWrapper>& ops,
                    absl::string_view group_name = "") {
  for (auto& op : ops) {
    CloneNamespace(source, op, group_name);
  }
}

}  // namespace
std::vector<ConstTensorWrapperRef> UnpackTensor(TensorPool& tensor_pool,
                                                std::vector<OpWrapper>& new_ops,
                                                const TensorWrapper& input,
                                                size_t unpack_dims) {
  auto input_dims = input.GetDimensions();
  auto num_unpack = input_dims[unpack_dims];
  input_dims.erase(input_dims.begin() + unpack_dims);
  std::vector<ConstTensorWrapperRef> outputs;
  outputs.reserve(num_unpack);
  for (size_t i = 0; i < num_unpack; ++i) {
    outputs.emplace_back(tensor_pool.CloneNativeTensorFrom(input, input_dims));
  }
  new_ops.emplace_back(
      CreateUnpackOp(input, outputs, static_cast<std::uint32_t>(unpack_dims)));
  return outputs;
}

std::vector<ConstTensorWrapperRef> SplitTensor(TensorPool& tensor_pool,
                                               std::vector<OpWrapper>& new_ops,
                                               const TensorWrapper& input,
                                               size_t axis, size_t tile_size) {
  auto input_dims = input.GetDimensions();
  size_t same_size_cnt = input_dims[axis] / tile_size;
  size_t num_outputs = same_size_cnt + (input_dims[axis] % tile_size != 0);
  QNN_LOG_DEBUG("[SplitTensor] num_outputs: %d", num_outputs);  
  if (num_outputs == 1) return {input};

  std::vector<ConstTensorWrapperRef> outputs;
  outputs.reserve(num_outputs);
  // Create Regular tiles based on tile_size.
  input_dims[axis] = tile_size;
  for (size_t i = 0; i < same_size_cnt; ++i) {
    QNN_LOG_DEBUG(" tile_size(%d) %d", i, input_dims[axis]);
    outputs.emplace_back(tensor_pool.CloneNativeTensorFrom(input, input_dims));
  }
  // Create the last tile smaller than tile_size if needed.
  if (num_outputs > same_size_cnt) {
    input_dims[axis] = input.GetDimension(axis) % tile_size;
    QNN_LOG_DEBUG(" tile_size(%d) %d", num_outputs - 1, input_dims[axis]);
    outputs.emplace_back(tensor_pool.CloneNativeTensorFrom(input, input_dims));
  }

  std::vector<std::uint32_t> split_indice;
  split_indice.reserve(num_outputs - 1);
  for (std::uint32_t i = 1; i < num_outputs; i++) {
    split_indice.emplace_back(i * tile_size);
  }
  const auto& split_indice_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, {},
      {static_cast<std::uint32_t>(split_indice.size())},
      sizeof(split_indice[0]) * split_indice.size(), split_indice.data());
  new_ops.emplace_back(
      CreateSplitOp(input, outputs, axis, split_indice_tensor));
  return outputs;
}

TensorWrapper& BuildSingleSHAByUnpackAxis1(
    std::vector<OpWrapper>& new_ops, TensorPool& tensor_pool,
    const uint32_t num_attn_per_kv_heads, const TensorWrapper& input,
    const TensorWrapper& k_cache, const TensorWrapper& k_slice,
    const TensorWrapper& v_cache, const TensorWrapper& v_slice,
    const OpWrapper* scale_mul, const OpWrapper& q_kcache_matmul,
    const OpWrapper& q_kslice_matmul, const OpWrapper& qk_concat,
    const OpWrapper& mask_add, const OpWrapper& softmax,
    const OpWrapper& qk_vcache_slice, const OpWrapper& qk_vslice_slice,
    const OpWrapper& qk_vcache_matmul, const OpWrapper& qk_vslice_matmul,
    const OpWrapper& qkv_add) {
  const TensorWrapper* matmul_input = &input;
  // Scale Mul -> change matmul_input.
  if (scale_mul) {
    auto mul_output_dims = scale_mul->GetOutputTensor(0).GetDimensions();
    mul_output_dims[1] = 1;
    matmul_input = &tensor_pool.CloneNativeTensorFrom(
        scale_mul->GetOutputTensor(0), mul_output_dims);
    new_ops.emplace_back(CreateElementWiseMulOp(
        input, scale_mul->GetInputTensor(1), *matmul_input));
  }

  // Q KCache Matmul
  auto q_kcache_matmul_output_dims =
      q_kcache_matmul.GetOutputTensor(0).GetDimensions();
  q_kcache_matmul_output_dims[1] = 1u;
  q_kcache_matmul_output_dims[2] /= num_attn_per_kv_heads;
  auto& q_kcache_matmul_output = tensor_pool.CloneNativeTensorFrom(
      q_kcache_matmul.GetOutputTensor(0), q_kcache_matmul_output_dims);
  const std::array<ConstTensorWrapperRef, 2> q_kcache_matmul_inputs = {
      *matmul_input, k_cache};
  const std::array<ConstTensorWrapperRef, 1> q_kcache_matmul_outputs = {
      q_kcache_matmul_output};
  new_ops.emplace_back(CreateOpWithSameParams(
      q_kcache_matmul, q_kcache_matmul_inputs, q_kcache_matmul_outputs));

  // Q KSlice Matmul
  auto q_kslice_matmul_output_dims =
      q_kslice_matmul.GetOutputTensor(0).GetDimensions();
  q_kslice_matmul_output_dims[1] = 1u;
  q_kslice_matmul_output_dims[2] /= num_attn_per_kv_heads;
  auto& q_kslice_matmul_output = tensor_pool.CloneNativeTensorFrom(
      q_kslice_matmul.GetOutputTensor(0), q_kslice_matmul_output_dims);
  const std::array<ConstTensorWrapperRef, 2> q_kslice_matmul_inputs = {
      *matmul_input, k_slice};
  const std::array<ConstTensorWrapperRef, 1> q_kslice_matmul_outputs = {
      q_kslice_matmul_output};
  new_ops.emplace_back(CreateOpWithSameParams(
      q_kslice_matmul, q_kslice_matmul_inputs, q_kslice_matmul_outputs));

  // QK Concat
  std::uint32_t adjusted_axis = 3u;
  auto concat_output_dims = qk_concat.GetOutputTensor(0).GetDimensions();
  concat_output_dims[1] = 1;
  concat_output_dims[2] /= num_attn_per_kv_heads;
  auto& concat_output = tensor_pool.CloneNativeTensorFrom(
      qk_concat.GetOutputTensor(0), concat_output_dims);
  new_ops.emplace_back(
      CreateConcatenationOp({q_kcache_matmul_output, q_kslice_matmul_output},
                            concat_output, adjusted_axis));

  // Mask Add
  const auto& mask_add_out = mask_add.GetOutputTensor(0);
  auto mask_add_output_dims = mask_add_out.GetDimensions();
  mask_add_output_dims[1] = 1;
  mask_add_output_dims[2] /= num_attn_per_kv_heads;
  auto& mask_add_output = tensor_pool.CloneNativeTensorFrom(
      mask_add.GetOutputTensor(0), mask_add_output_dims);
  new_ops.emplace_back(CreateElementWiseAddOp(
      concat_output, mask_add.GetInputTensor(1), mask_add_output));
  // Softmax
  auto softmax_output_dims = softmax.GetOutputTensor(0).GetDimensions();
  softmax_output_dims[1] = 1;
  softmax_output_dims[2] /= num_attn_per_kv_heads;
  auto& softmax_output = tensor_pool.CloneNativeTensorFrom(
      softmax.GetOutputTensor(0), softmax_output_dims);
  const std::array<ConstTensorWrapperRef, 1> softmax_inputs = {mask_add_output};
  const std::array<ConstTensorWrapperRef, 1> softmax_outputs = {softmax_output};
  new_ops.emplace_back(
      CreateOpWithSameParams(softmax, softmax_inputs, softmax_outputs));

  // QK VCache Slice
  auto qk_vcache_slice_param = qk_vcache_slice.GetTensorParam(0).GetTensor();
  auto qk_vcache_slice_param_data =
      qk_vcache_slice_param.GetTensorData<int32_t>();
  std::vector<int32_t> qk_vcache_slice_ranges(
      qk_vcache_slice_param_data.value().begin(),
      qk_vcache_slice_param_data.value().end());
  qk_vcache_slice_ranges[4] = 1;
  qk_vcache_slice_ranges[7] /= num_attn_per_kv_heads;
  std::vector<uint32_t> qk_vcache_slice_param_dims = {
      static_cast<uint32_t>(qk_vcache_slice_ranges.size() / 3), 3};
  auto& qk_vcache_slice_param_tensor = tensor_pool.CreateStaticTensor(
      qk_vcache_slice_param.GetDataType(),
      qk_vcache_slice_param.GetQuantParams(), qk_vcache_slice_param_dims,
      sizeof(qk_vcache_slice_ranges[0]) * qk_vcache_slice_ranges.size(),
      qk_vcache_slice_ranges.data());
  auto qk_vcache_slice_output_dims =
      qk_vcache_slice.GetOutputTensor(0).GetDimensions();
  qk_vcache_slice_output_dims[1] = 1;
  qk_vcache_slice_output_dims[2] /= num_attn_per_kv_heads;
  auto& qk_vcache_slice_output = tensor_pool.CloneNativeTensorFrom(
      qk_vcache_slice.GetOutputTensor(0), qk_vcache_slice_output_dims);
  new_ops.emplace_back(CreateSliceOp(softmax_output, qk_vcache_slice_output,
                                     qk_vcache_slice_param_tensor));

  // QK VSlice Slice
  auto qk_vslice_slice_param = qk_vslice_slice.GetTensorParam(0).GetTensor();
  auto qk_vslice_slice_param_data =
      qk_vslice_slice_param.GetTensorData<int32_t>();
  std::vector<int32_t> qk_vslice_slice_ranges(
      qk_vslice_slice_param_data.value().begin(),
      qk_vslice_slice_param_data.value().end());
  qk_vslice_slice_ranges[4] = 1;
  qk_vslice_slice_ranges[7] /= num_attn_per_kv_heads;
  std::vector<uint32_t> qk_vslice_slice_param_dims = {
      static_cast<uint32_t>(qk_vslice_slice_ranges.size() / 3), 3};
  auto& qk_vslice_slice_param_tensor = tensor_pool.CreateStaticTensor(
      qk_vslice_slice_param.GetDataType(),
      qk_vslice_slice_param.GetQuantParams(), qk_vslice_slice_param_dims,
      sizeof(qk_vslice_slice_ranges[0]) * qk_vslice_slice_ranges.size(),
      qk_vslice_slice_ranges.data());
  auto qk_vslice_slice_output_dims =
      qk_vslice_slice.GetOutputTensor(0).GetDimensions();
  qk_vslice_slice_output_dims[1] = 1;
  qk_vslice_slice_output_dims[2] /= num_attn_per_kv_heads;
  auto& qk_vslice_slice_output = tensor_pool.CloneNativeTensorFrom(
      qk_vslice_slice.GetOutputTensor(0), qk_vslice_slice_output_dims);
  new_ops.emplace_back(CreateSliceOp(softmax_output, qk_vslice_slice_output,
                                     qk_vslice_slice_param_tensor));

  // QK VCache Matmul
  auto qk_vcache_matmul_output_dims =
      qk_vcache_matmul.GetOutputTensor(0).GetDimensions();
  qk_vcache_matmul_output_dims[1] = 1;
  qk_vcache_matmul_output_dims[2] /= num_attn_per_kv_heads;
  auto& qk_vcache_matmul_output = tensor_pool.CloneNativeTensorFrom(
      qk_vcache_matmul.GetOutputTensor(0), qk_vcache_matmul_output_dims);
  const std::array<ConstTensorWrapperRef, 2> qk_vcache_matmul_inputs = {
      qk_vcache_slice_output, v_cache};
  const std::array<ConstTensorWrapperRef, 1> qk_vcache_matmul_outputs = {
      qk_vcache_matmul_output};
  new_ops.emplace_back(CreateOpWithSameParams(
      qk_vcache_matmul, qk_vcache_matmul_inputs, qk_vcache_matmul_outputs));

  // QK VSlice Matmul
  auto qk_vslice_matmul_output_dims =
      qk_vslice_matmul.GetOutputTensor(0).GetDimensions();
  qk_vslice_matmul_output_dims[1] = 1;
  qk_vslice_matmul_output_dims[2] /= num_attn_per_kv_heads;
  auto& qk_vslice_matmul_output = tensor_pool.CloneNativeTensorFrom(
      qk_vslice_matmul.GetOutputTensor(0), qk_vslice_matmul_output_dims);
  const std::array<ConstTensorWrapperRef, 2> qk_vslice_matmul_inputs = {
      qk_vslice_slice_output, v_slice};
  const std::array<ConstTensorWrapperRef, 1> qk_vslice_matmul_outputs = {
      qk_vslice_matmul_output};
  new_ops.emplace_back(CreateOpWithSameParams(
      qk_vslice_matmul, qk_vslice_matmul_inputs, qk_vslice_matmul_outputs));

  // QKV Add
  auto qkv_add_output_dims = qkv_add.GetOutputTensor(0).GetDimensions();
  qkv_add_output_dims[1] = 1;
  qkv_add_output_dims[2] /= num_attn_per_kv_heads;
  auto& qkv_add_output = tensor_pool.CloneNativeTensorFrom(
      qkv_add.GetOutputTensor(0), qkv_add_output_dims);
  new_ops.emplace_back(CreateElementWiseAddOp(
      qk_vcache_matmul_output, qk_vslice_matmul_output, qkv_add_output));

  return qkv_add_output;
}

size_t OptimizeGqaPrefill(std::function<bool(OpWrapper&)> validate_op_config,
                          std::vector<OpWrapper>& ops, size_t start_index,
                          TensorPool& tensor_pool, size_t pattern_size) {
  QNN_LOG_INFO("[G2G] OptimizeGqaPrefill");
  const auto is_connected =
      [&ops, &start_index](int32_t output_op_index, size_t output_tensor_index,
                           int32_t input_op_index,
                           size_t input_tensor_index) -> bool {
    // Input/output op index might be negative.
    int32_t out_op_idx = static_cast<int32_t>(start_index) + output_op_index;
    int32_t in_op_idx = static_cast<int32_t>(start_index) + input_op_index;
    return out_op_idx >= 0 && in_op_idx >= 0 &&
           ops[out_op_idx].GetOutputTensor(output_tensor_index) ==
               ops[in_op_idx].GetInputTensor(input_tensor_index);
  };
  bool has_scale_mul =
      is_connected(-1, 0, 0, 0) && IsElementWiseMultiply(ops[start_index - 1]);
  QNN_LOG_INFO("[G2G] GQA Optimization (Prefill): has_scale_mul %d",
               has_scale_mul);
  // Adjust indices based on scale_mul.
  start_index -= has_scale_mul;
  pattern_size += has_scale_mul;
  const size_t kQScaleReshapeIdx = 0 + has_scale_mul;
  const size_t kQKCacheMatmulIdx = 1 + has_scale_mul;
  const size_t kQKSliceMatmulIdx = 2 + has_scale_mul;
  const size_t kQKConcatIdx = 3 + has_scale_mul;
  const size_t kMaskConcatIdx = 4 + has_scale_mul;
  const size_t kMaskAddIdx = 5 + has_scale_mul;
  const size_t kSoftmaxIdx = 6 + has_scale_mul;
  const size_t kQKVCacheSliceIdx = 7 + has_scale_mul;
  const size_t kQKVSliceSliceIdx = 8 + has_scale_mul;
  const size_t kQKVCacheMatmulIdx = 9 + has_scale_mul;
  const size_t kQKVSliceMatmulIdx = 10 + has_scale_mul;
  const size_t kQKVAddIdx = 11 + has_scale_mul;
  const size_t kQKVReshapeIdx = 12 + has_scale_mul;
  const size_t kQKVTransposeIdx = 13 + has_scale_mul;
  const size_t kOProjReshapeIdx = 14 + has_scale_mul;

  if (!(is_connected(kQScaleReshapeIdx, 0, kQKCacheMatmulIdx, 0) &&
        is_connected(kQScaleReshapeIdx, 0, kQKSliceMatmulIdx, 0) &&
        is_connected(kQKCacheMatmulIdx, 0, kQKConcatIdx, 0) &&
        is_connected(kQKSliceMatmulIdx, 0, kQKConcatIdx, 1) &&
        is_connected(kQKConcatIdx, 0, kMaskAddIdx, 0) &&
        is_connected(kMaskAddIdx, 0, kSoftmaxIdx, 0) &&
        is_connected(kSoftmaxIdx, 0, kQKVCacheSliceIdx, 0) &&
        is_connected(kSoftmaxIdx, 0, kQKVSliceSliceIdx, 0) &&
        is_connected(kQKVCacheSliceIdx, 0, kQKVCacheMatmulIdx, 0) &&
        is_connected(kQKVSliceSliceIdx, 0, kQKVSliceMatmulIdx, 0) &&
        is_connected(kQKVCacheMatmulIdx, 0, kQKVAddIdx, 0) &&
        is_connected(kQKVSliceMatmulIdx, 0, kQKVAddIdx, 1) &&
        is_connected(kQKVAddIdx, 0, kQKVReshapeIdx, 0) &&
        is_connected(kQKVReshapeIdx, 0, kQKVTransposeIdx, 0) &&
        is_connected(kQKVTransposeIdx, 0, kOProjReshapeIdx, 0) &&
        IsElementWiseAdd(ops[start_index + kMaskAddIdx]) &&
        IsElementWiseAdd(ops[start_index + kQKVAddIdx]))) {
    return 1;
  }

  const size_t num_q_heads = ops[start_index].GetInputTensor(0).GetDimension(1);
  const size_t num_kv_heads =
      ops[start_index + kQKSliceMatmulIdx].GetInputTensor(1).GetDimension(1);
  // Strict check:
  // - FastVLM with q head = 14, kv head = 2.
  // - Kanana with q head = 16, kv head = 8.
  // - TinyTiny with q head = 4, kv head = 2 & 1.
  QNN_LOG_INFO(
      "[G2G] GQA Optimization (Prefill):\n  # Q Heads: %d\n  # K Heads: %d",
      num_q_heads, num_kv_heads);
  if (!((num_q_heads == 14 && num_kv_heads == 2) ||
        (num_q_heads == 16 && num_kv_heads == 8) ||
        (num_q_heads == 4 && num_kv_heads == 2) ||
        (num_q_heads == 4 && num_kv_heads == 1))) {
    return 1;
  }
  QNN_LOG_INFO("[G2G] GQA Optimization (Prefill): Start");
  std::vector<OpWrapper> new_ops;

  // QKV Unpack
  const auto& k_cache = ops[start_index + kQKCacheMatmulIdx].GetInputTensor(1);
  auto k_cache_unpack_outputs = SplitTensor(tensor_pool, new_ops, k_cache);
  constexpr size_t kUnpackAxis = 1;
  auto k_slice_unpack_outputs = SplitTensor(
      tensor_pool, new_ops,
      ops[start_index + kQKSliceMatmulIdx].GetInputTensor(1), kUnpackAxis);
  auto q_inputs =
      SplitTensor(tensor_pool, new_ops, ops[start_index].GetInputTensor(0));

  const auto& v_cache = ops[start_index + kQKVCacheMatmulIdx].GetInputTensor(1);
  auto v_cache_unpack_outputs = SplitTensor(tensor_pool, new_ops, v_cache);

  const auto& v_slice = ops[start_index + kQKVSliceMatmulIdx].GetInputTensor(1);
  auto v_slice_unpack_outputs = SplitTensor(tensor_pool, new_ops, v_slice);

  auto group_size = num_q_heads / num_kv_heads;
  // Remove unnessary concat mask.
  auto add_op = CreateElementWiseAddOp(
      ops[start_index + kMaskAddIdx].GetInputTensor(0),
      ops[start_index + kMaskConcatIdx].GetInputTensor(0),
      ops[start_index + kMaskAddIdx].GetOutputTensor(0));
  const auto& mask_add_out = add_op.GetOutputTensor(0);
  auto mask_add_output_dims = mask_add_out.GetDimensions();
  // Build num_head SHAs
  std::vector<ConstTensorWrapperRef> sha_outputs;
  sha_outputs.reserve(num_q_heads);
  for (size_t i = 0; i < num_kv_heads; ++i) {
    for (size_t j = 0; j < group_size; ++j) {
      auto& sha_output = BuildSingleSHAByUnpackAxis1(
          new_ops, tensor_pool, group_size, q_inputs[i * group_size + j],
          k_cache_unpack_outputs[i], k_slice_unpack_outputs[i],
          v_cache_unpack_outputs[i], v_slice_unpack_outputs[i],
          has_scale_mul ? &ops[start_index] : nullptr,
          ops[start_index + kQKCacheMatmulIdx],
          ops[start_index + kQKSliceMatmulIdx], ops[start_index + kQKConcatIdx],
          add_op, ops[start_index + kSoftmaxIdx],
          ops[start_index + kQKVCacheSliceIdx],
          ops[start_index + kQKVSliceSliceIdx],
          ops[start_index + kQKVCacheMatmulIdx],
          ops[start_index + kQKVSliceMatmulIdx], ops[start_index + kQKVAddIdx]);
      sha_outputs.emplace_back(sha_output);
    }
  }
  const auto& qkv_reshape = ops[start_index + pattern_size - 1];
  // Concat SHA outputs by the last dimension.
  const auto concat_axis = sha_outputs[0].get().GetRank() - 1;
  auto concat_sha_dims = sha_outputs[0].get().GetDimensions();
  concat_sha_dims[concat_axis] = 0;
  for (const auto& sha_output : sha_outputs) {
    concat_sha_dims[concat_axis] += sha_output.get().GetDimension(concat_axis);
  }
  const auto& concat_sha_output = tensor_pool.CloneNativeTensorFrom(
      qkv_reshape.GetInputTensor(0), concat_sha_dims);
  new_ops.emplace_back(
      CreateConcatenationOp(sha_outputs, concat_sha_output, concat_axis));
  new_ops.emplace_back(
      CreateReshapeOp(concat_sha_output, qkv_reshape.GetOutputTensor(0)));

  // Clone namespace.
  CloneNamespace(ops[start_index + kQScaleReshapeIdx], new_ops);
  // Validate new graph.
  const bool is_valid =
      std::all_of(new_ops.begin(), new_ops.end(),
                  [validate_op_config](OpWrapper& op_wrapper) -> bool {
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

size_t OptimizeGqaDecode(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  QNN_LOG_INFO("Unified GQA DECODE");
  const auto is_connected =
      [&ops, &start_index](int32_t output_op_index, size_t output_tensor_index,
                           int32_t input_op_index,
                           size_t input_tensor_index) -> bool {
    // Input/output op index might be negative.
    int32_t out_op_idx = static_cast<int32_t>(start_index) + output_op_index;
    int32_t in_op_idx = static_cast<int32_t>(start_index) + input_op_index;
    return out_op_idx >= 0 && in_op_idx >= 0 &&
           ops[out_op_idx].GetOutputTensor(output_tensor_index) ==
               ops[in_op_idx].GetInputTensor(input_tensor_index);
  };
  // Case 1 (offset: 2): mul -> reshape -> gqa
  // Case 2 (offset: 1): rehsapae -> gqa
  // Case 3 (offset: 0): other -> gqa
  size_t offset = 0;
  if (is_connected(-1, 0, 0, 0) && is_connected(-1, 0, 1, 0) &&
      ops[start_index - 1].IsOpCode(QnnOpCode::kReshape)) {
    offset++;
    start_index--;
    if (is_connected(-1, 0, 0, 0) &&
        IsElementWiseMultiply(ops[start_index - 1])) {
      offset++;
      start_index--;
    }
  }
  pattern_size += offset;
  if (offset == 2) {
    QNN_LOG_INFO(">> mul -> reshape -> GQA");
  } else if (offset == 1) {
    QNN_LOG_INFO(">> reshape -> GQA @ %d (is_reshape %d)", start_index,
                 ops[start_index].IsOpCode(QnnOpCode::kReshape));
  } else if (offset == 0) {
    QNN_LOG_INFO(">> -> GQA");
  } else {
    QNN_LOG_INFO(">> ERROR: Unknown GQA");
    return 1;
  }

  const size_t kQKCacheMatmulIdx = 0 + offset;
  const size_t kQKSliceMatmulIdx = 1 + offset;
  const size_t kQKConcatIdx = 2 + offset;
  const size_t kMaskAddIdx = 3 + offset;
  const size_t kSoftmaxIdx = 4 + offset;
  const size_t kQKVCacheSliceIdx = 5 + offset;
  const size_t kQKVSliceSliceIdx = 6 + offset;
  const size_t kQKVCacheMatmulIdx = 7 + offset;
  const size_t kQKVSliceMatmulIdx = 8 + offset;
  const size_t kQKVAddIdx = 9 + offset;
  const size_t kQKVReshapeIdx = 10 + offset;

  if (!(ops[start_index + kQKCacheMatmulIdx].GetInputTensor(0) ==
            ops[start_index + kQKSliceMatmulIdx].GetInputTensor(0) &&
        is_connected(kQKCacheMatmulIdx, 0, kQKConcatIdx, 0) &&
        is_connected(kQKSliceMatmulIdx, 0, kQKConcatIdx, 1) &&
        is_connected(kQKConcatIdx, 0, kMaskAddIdx, 0) &&
        is_connected(kMaskAddIdx, 0, kSoftmaxIdx, 0) &&
        is_connected(kSoftmaxIdx, 0, kQKVCacheSliceIdx, 0) &&
        is_connected(kSoftmaxIdx, 0, kQKVSliceSliceIdx, 0) &&
        is_connected(kQKVCacheSliceIdx, 0, kQKVCacheMatmulIdx, 0) &&
        is_connected(kQKVSliceSliceIdx, 0, kQKVSliceMatmulIdx, 0) &&
        is_connected(kQKVCacheMatmulIdx, 0, kQKVAddIdx, 0) &&
        is_connected(kQKVSliceMatmulIdx, 0, kQKVAddIdx, 1) &&
        is_connected(kQKVAddIdx, 0, kQKVReshapeIdx, 0))) {
    QNN_LOG_WARNING(
        "[G2G] Failed to check connectivity when doing MHA-SHA transformation "
        "for GQA decode.");
    return 1;
  }

  constexpr size_t kSupportedRank = 4;
  constexpr size_t kUnpackAxis = 1;
  size_t split_index = 1;
  size_t num_q_heads =
      ops[start_index].GetInputTensor(0).GetDimension(split_index);
  if (num_q_heads == 1) {
    split_index = 2;
    num_q_heads = ops[start_index].GetInputTensor(0).GetDimension(split_index);
  }
  const size_t num_kv_heads =
      ops[start_index + kQKSliceMatmulIdx].GetInputTensor(1).GetDimension(1);
  // Strict check:
  // - FastVLM with q head = 14, kv head = 2.
  // - Kanana with q head = 16, kv head = 8.
  // - TinyTiny with q head = 4, kv head = 2.
  //   - Disable q head = 4 and kv head = 1.
  QNN_LOG_INFO(
      "[G2G] GQA Optimization (Decode):\n  # Q Heads: %d\n  # K Heads: %d",
      num_q_heads, num_kv_heads);
  if (!((num_q_heads == 14 && num_kv_heads == 2) ||
        (num_q_heads == 16 && num_kv_heads == 8) ||
        (num_q_heads == 4 && num_kv_heads == 2))) {
    return 1;
  }
  QNN_LOG_INFO("[G2G] GQA Optimization (Decode): Start");

  std::vector<OpWrapper> new_ops;
  auto q_inputs = SplitTensor(tensor_pool, new_ops,
                              ops[start_index].GetInputTensor(0), split_index);
  auto k_cache_outputs = SplitTensor(
      tensor_pool, new_ops,
      ops[start_index + kQKCacheMatmulIdx].GetInputTensor(1), kUnpackAxis);
  auto k_slice_outputs = SplitTensor(
      tensor_pool, new_ops,
      ops[start_index + kQKSliceMatmulIdx].GetInputTensor(1), kUnpackAxis);
  auto v_cache_outputs = SplitTensor(
      tensor_pool, new_ops,
      ops[start_index + kQKVCacheMatmulIdx].GetInputTensor(1), kUnpackAxis);
  auto v_slice_outputs = SplitTensor(
      tensor_pool, new_ops,
      ops[start_index + kQKVSliceMatmulIdx].GetInputTensor(1), kUnpackAxis);

  // Build SHA
  const auto group_size = num_q_heads / num_kv_heads;
  std::vector<ConstTensorWrapperRef> sha_outputs;
  for (size_t i = 0; i < num_kv_heads; ++i) {
    for (size_t j = 0; j < group_size; ++j) {
      const auto& sha_output = BuildSingleSHAByUnpackAxis1(
          new_ops, tensor_pool, group_size, q_inputs[i * group_size + j],
          k_cache_outputs[i], k_slice_outputs[i], v_cache_outputs[i],
          v_slice_outputs[i],
          IsElementWiseMultiply(ops[start_index]) ? &ops[start_index] : nullptr,
          ops[start_index + kQKCacheMatmulIdx],
          ops[start_index + kQKSliceMatmulIdx], ops[start_index + kQKConcatIdx],
          ops[start_index + kMaskAddIdx], ops[start_index + kSoftmaxIdx],
          ops[start_index + kQKVCacheSliceIdx],
          ops[start_index + kQKVSliceSliceIdx],
          ops[start_index + kQKVCacheMatmulIdx],
          ops[start_index + kQKVSliceMatmulIdx], ops[start_index + kQKVAddIdx]);
      sha_outputs.emplace_back(sha_output);
    }
  }

  // Concat SHA outputs by the last dimension.
  const auto concat_axis = sha_outputs[0].get().GetRank() - 1;
  auto concat_sha_dims = sha_outputs[0].get().GetDimensions();
  concat_sha_dims[concat_axis] = 0;
  for (const auto& sha_output : sha_outputs) {
    concat_sha_dims[concat_axis] += sha_output.get().GetDimension(concat_axis);
  }
  const auto& concat_sha_output = tensor_pool.CloneNativeTensorFrom(
      ops[start_index + kQKVReshapeIdx].GetInputTensor(0), concat_sha_dims);
  new_ops.emplace_back(
      CreateConcatenationOp(sha_outputs, concat_sha_output, concat_axis));
  new_ops.emplace_back(CreateReshapeOp(
      concat_sha_output, ops[start_index + kQKVReshapeIdx].GetOutputTensor(0)));
  // Clone namespace.
  CloneNamespace(ops[start_index], new_ops);
  // Validate new graph.
  const bool is_valid =
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
    QNN_LOG_INFO("new_ops size: %d", new_ops.size());
    QNN_LOG_INFO("%s", new_ops[0].GetName().data());
    QNN_LOG_INFO("%s %d", ops[start_index].GetName().data(), start_index);
    for (size_t i = 37; i< 140; ++i) {
        QNN_LOG_INFO("%d", ops[i].GetOpCode());
    }
    ops.insert(ops.begin() + start_index + pattern_size,
               std::make_move_iterator(new_ops.begin()),
               std::make_move_iterator(new_ops.end()));
    ops.erase(ops.begin() + start_index,
              ops.begin() + start_index + pattern_size);
    QNN_LOG_INFO("[G2G] GQA optimization (decode) done.");
    return step_size;
  }
  QNN_LOG_WARNING(
      "[G2G] Validation failed. Rolling back to the original graph.");
  return 1;
}

size_t OptimizeMHAAttn(std::function<bool(OpWrapper&)> validate_op_config,
                       std::vector<OpWrapper>& ops, size_t attn_start_index,
                       TensorPool& tensor_pool, size_t pattern_size) {
  // attn (attention mask)
  constexpr size_t kAttnSelect = 7;
  constexpr size_t kAttnNotEqual = 6;
  constexpr size_t kAttnReshape = 5;
  // attn (QK)
  constexpr int32_t kAttnMulQ = -4;
  constexpr int32_t kAttnMulK = -3;
  constexpr int32_t kAttnTransposeQ = -2;
  constexpr int32_t kAttnTransposeK = -1;
  // attn (Softmax & V)
  constexpr int32_t kAttnSoftmax = 1;
  constexpr int32_t kAttnTransposeIn = 2;
  constexpr int32_t kAttnMatMul = 3;
  constexpr int32_t kAttnTransposeOut = 4;

  // Connection check: Reshape -> NotEqual -> Select
  size_t start_index = attn_start_index;
  if (!(IS_CONNECTED(kAttnReshape, 0, kAttnNotEqual, 0)) &&
      (IS_CONNECTED(kAttnNotEqual, 0, kAttnSelect, 0))) {
    return 1;
  }
  // attn_not_equal_op is copied from ops since ops will be modified.
  const auto& attn_not_equal_op = ops[attn_start_index + kAttnNotEqual];
  const auto& not_equal_out = attn_not_equal_op.GetOutputTensor(0);
  // Count the operations that have NotEqual as their source op.
  const auto& reshape_in =
      ops[attn_start_index + kAttnReshape].GetInputTensor(0);
  size_t num_out =
      std::count_if(ops.begin(), ops.end(), [&](const OpWrapper& op) {
        return op.IsOpCode(QnnOpCode::kElementWiseSelect) &&
               op.GetInputTensor(0) == not_equal_out;
      });
  if (num_out == 0) {
    return 1;
  }

  QNN_LOG_INFO("[G2G] MHA optimization (Attn)");
  // Handle masking.
  std::vector<OpWrapper> new_ops;
  auto not_equal_out_dims = not_equal_out.GetDimensions();
  not_equal_out_dims.erase(not_equal_out_dims.begin() + 1);
  const auto& select_mask =
      tensor_pool.CloneNativeTensorFrom(not_equal_out, not_equal_out_dims);
  // Change NotEqual to Equal -> Cast -> Mul.
  const auto& zero_tensor = attn_not_equal_op.GetInputTensor(1);
  new_ops.emplace_back(
      CreateElementWiseEqualOp(reshape_in, zero_tensor, select_mask));
  const auto& select_out =
      ops[attn_start_index + kAttnSelect].GetOutputTensor(0);
  auto select_out_dims = select_out.GetDimensions();
  select_out_dims.erase(select_out_dims.begin() + 1);

  const auto& mul_in =
      tensor_pool.CloneNativeTensorFrom(select_out, select_out_dims);
  new_ops.emplace_back(CreateCastOp(select_mask, mul_in));

  const auto& select_const =
      ops[attn_start_index + kAttnSelect].GetInputTensor(2);
  // TODO(jiunkaiy): Remove this magic number (-65472) after HTP resolves
  // accuracy issues.
  float mul_const_value =
      std::max(select_const.GetTensorData<float>().value()[0], -65472.f);
  const auto& mul_const = tensor_pool.CreateStaticTensor(
      select_const.GetDataType(), select_const.GetQuantParams(),
      select_const.GetDimensions(), select_const.GetTensorBytes(),
      &mul_const_value);
  const auto& add_in =
      tensor_pool.CloneNativeTensorFrom(select_out, select_out_dims);
  new_ops.emplace_back(CreateElementWiseMulOp(mul_in, mul_const, add_in));

  // Create SHAs based on Select index.
  size_t select_index = 0;
  for (size_t output_index = 0; output_index < num_out; ++output_index) {
    // Identify Select index.
    auto it_select = std::find_if(
        ops.begin() + select_index + 1, ops.end(), [&](const OpWrapper& op) {
          return op.IsOpCode(QnnOpCode::kElementWiseSelect) &&
                 op.GetInputTensor(0) == not_equal_out;
        });
    if (it_select == ops.end()) {
      QNN_LOG_ERROR("Could not find Select op with the given input tensor");
      break;
    }
    select_index = std::distance(ops.begin(), it_select);

    // Connection check based on Select index.
    start_index = select_index;
    if (!(IS_CONNECTED(0, 0, kAttnSoftmax, 0) &&
          IS_CONNECTED(kAttnSoftmax, 0, kAttnMatMul, 1) &&
          IS_CONNECTED(kAttnTransposeIn, 0, kAttnMatMul, 0) &&
          IS_CONNECTED(kAttnMatMul, 0, kAttnTransposeOut, 0))) {
      QNN_LOG_ERROR("[G2G] Connection check failed.");
      return 1;
    }
    // Identify MatMul's index.
    auto it_matmul =
        std::find_if(ops.begin(), ops.end(), [&](const OpWrapper& op) {
          return op.IsOpCode(QnnOpCode::kMatMul) &&
                 op.GetOutputTensor(0) == ops[select_index].GetInputTensor(1);
        });
    if (it_matmul == ops.end()) {
      QNN_LOG_ERROR("Could not find MatMul op with the given output tensor");
      break;
    }
    size_t matmul_qk_index = std::distance(ops.begin(), it_matmul);

    // Connection check based on Matmul index.
    start_index = matmul_qk_index;
    if (!(IS_CONNECTED(kAttnMulQ, 0, kAttnTransposeQ, 0) &&
          IS_CONNECTED(kAttnMulK, 0, kAttnTransposeK, 0) &&
          IS_CONNECTED(kAttnTransposeQ, 0, 0, 0) &&
          IS_CONNECTED(kAttnTransposeK, 0, 0, 1) &&
          IsElementWiseMultiply(ops[start_index + kAttnMulQ]) &&
          IsElementWiseMultiply(ops[start_index + kAttnMulK]))) {
      QNN_LOG_ERROR("[G2G] Connection check failed.");
      return 1;
    }
    // QKV Unpack
    const auto& mul_q_in = ops[matmul_qk_index + kAttnMulQ].GetInputTensor(0);
    auto q_unpack_dims = mul_q_in.GetDimensions();
    uint32_t num_heads = q_unpack_dims[2];
    const auto& mul_k_in = ops[matmul_qk_index + kAttnMulK].GetInputTensor(0);
    auto k_unpack_dims = mul_k_in.GetDimensions();
    const auto& transpose_v_in =
        ops[select_index + kAttnTransposeIn].GetInputTensor(0);
    auto transpose_v_perm =
        ops[select_index + kAttnTransposeIn].GetTensorParam(0).GetTensor();
    std::vector<uint32_t> perm_data = {0, 2, 1};
    const auto& perm_tensor = tensor_pool.CreateStaticTensor(
        QNN_DATATYPE_UINT_32, transpose_v_perm.GetQuantParams(), {3},
        perm_data.size() * sizeof(perm_data[0]), perm_data.data());
    auto v_unpack_dims = transpose_v_in.GetDimensions();
    const auto& mha_out =
        ops[select_index + kAttnTransposeOut].GetOutputTensor(0);
    auto mha_out_dims = mha_out.GetDimensions();
    if (!(num_heads == k_unpack_dims[2] && num_heads == v_unpack_dims[2] &&
          num_heads == mha_out_dims[2])) {
      QNN_LOG_ERROR("[G2G] Num heads mismatches.");
      return 1;
    }
    q_unpack_dims.erase(q_unpack_dims.begin() + 2);
    k_unpack_dims.erase(k_unpack_dims.begin() + 2);
    v_unpack_dims.erase(v_unpack_dims.begin() + 2);
    mha_out_dims.erase(mha_out_dims.begin() + 2);
    // Prepare inputs and outputs for num_heads SHAs.
    std::vector<ConstTensorWrapperRef> q_sha_inputs;
    std::vector<ConstTensorWrapperRef> k_sha_inputs;
    std::vector<ConstTensorWrapperRef> v_sha_inputs;
    std::vector<ConstTensorWrapperRef> sha_outputs;
    q_sha_inputs.reserve(num_heads);
    k_sha_inputs.reserve(num_heads);
    v_sha_inputs.reserve(num_heads);
    sha_outputs.reserve(num_heads);

    for (int i = 0; i < num_heads; ++i) {
      const auto& q_unpack =
          tensor_pool.CloneNativeTensorFrom(mul_q_in, q_unpack_dims);
      q_sha_inputs.emplace_back(q_unpack);

      const auto& k_unpack =
          tensor_pool.CloneNativeTensorFrom(mul_k_in, k_unpack_dims);
      k_sha_inputs.emplace_back(k_unpack);

      const auto& v_unpack =
          tensor_pool.CloneNativeTensorFrom(transpose_v_in, v_unpack_dims);
      v_sha_inputs.emplace_back(v_unpack);

      const auto& sha_out =
          tensor_pool.CloneNativeTensorFrom(mha_out, mha_out_dims);
      sha_outputs.emplace_back(sha_out);
    }
    new_ops.emplace_back(CreateUnpackOp(mul_q_in, q_sha_inputs, 2));
    new_ops.emplace_back(CreateUnpackOp(mul_k_in, k_sha_inputs, 2));
    new_ops.emplace_back(CreateUnpackOp(transpose_v_in, v_sha_inputs, 2));

    for (int i = 0; i < num_heads; ++i) {
      const auto& q_matmul_in =
          tensor_pool.CloneNativeTensorFrom(q_sha_inputs[i]);
      new_ops.emplace_back(CreateElementWiseMulOp(
          q_sha_inputs[i], ops[matmul_qk_index + kAttnMulQ].GetInputTensor(1),
          q_matmul_in));

      const auto& k_transpose_in =
          tensor_pool.CloneNativeTensorFrom(k_sha_inputs[i]);
      new_ops.emplace_back(CreateElementWiseMulOp(
          k_sha_inputs[i], ops[matmul_qk_index + kAttnMulK].GetInputTensor(1),
          k_transpose_in));

      const auto& k_matmul_in = tensor_pool.CloneNativeTensorFrom(
          k_transpose_in,
          {k_unpack_dims[0], k_unpack_dims[2], k_unpack_dims[1]});
      new_ops.emplace_back(
          CreateTransposeOp(k_transpose_in, k_matmul_in, perm_tensor));
      // MatMul
      const auto& matmul_qk_out = ops[matmul_qk_index].GetOutputTensor(0);
      const auto& select_in = tensor_pool.CloneNativeTensorFrom(
          matmul_qk_out,
          {q_matmul_in.GetDimension(0), q_matmul_in.GetDimension(1),
           k_matmul_in.GetDimension(2)});
      const std::array<ConstTensorWrapperRef, 2> matmul_qk_inputs = {
          q_matmul_in, k_matmul_in};
      const std::array<ConstTensorWrapperRef, 1> matmul_qk_outputs = {
          select_in};
      new_ops.emplace_back(CreateOpWithSameParams(
          ops[matmul_qk_index], matmul_qk_inputs, matmul_qk_outputs));

      // Change Select to Add.
      const auto& softmax_in =
          tensor_pool.CloneNativeTensorFrom(select_out, select_out_dims);
      new_ops.emplace_back(
          CreateElementWiseAddOp(select_in, add_in, softmax_in));

      // Softmax
      const auto& qk_softmax =
          tensor_pool.CloneNativeTensorFrom(softmax_in, select_out_dims);
      const std::array<ConstTensorWrapperRef, 1> softmax_inputs = {softmax_in};
      const std::array<ConstTensorWrapperRef, 1> softmax_outputs = {qk_softmax};
      new_ops.emplace_back(CreateOpWithSameParams(
          ops[select_index + kAttnSoftmax], softmax_inputs, softmax_outputs));

      // MatMul
      const std::array<ConstTensorWrapperRef, 2> matmul_out_inputs = {
          qk_softmax, v_sha_inputs[i]};
      const std::array<ConstTensorWrapperRef, 1> matmul_out_outputs = {
          sha_outputs[i]};
      new_ops.emplace_back(CreateOpWithSameParams(
          ops[matmul_qk_index], matmul_out_inputs, matmul_out_outputs));
    }
    // Pack
    new_ops.emplace_back(CreatePackOp(sha_outputs, mha_out, 2));

    const bool is_valid =
        std::all_of(new_ops.begin(), new_ops.end(),
                    [validate_op_config](OpWrapper& op_wrapper) -> bool {
                      return validate_op_config(op_wrapper);
                    });
    if (is_valid) {
      // Adjust the name to avoid a name collision in the Qnn JSON dump.
      for (size_t i = 0; i < new_ops.size(); ++i) {
        new_ops[i].AddSuffixToName(absl::StrCat("_qcg2g_", i));
      }
      // Replace the matched pattern with a newly generated subgraph.
      ops.insert(ops.begin() + select_index + kAttnTransposeOut + 1,
                 std::make_move_iterator(new_ops.begin()),
                 std::make_move_iterator(new_ops.end()));
      // Erase original pattern backwards.
      ops.erase(ops.begin() + select_index,
                ops.begin() + select_index + kAttnTransposeOut + 1);
      if (output_index == 0) {
        ops.erase(ops.begin() + attn_start_index + kAttnNotEqual);
        ops.erase(ops.begin() + attn_start_index + kAttnReshape);
      }
      ops.erase(ops.begin() + matmul_qk_index + kAttnMulQ,
                ops.begin() + matmul_qk_index + 1);
    } else {
      QNN_LOG_ERROR(
          "[G2G] Validation failed. Rolling back to the original graph.");
      return 1;
    }
    new_ops.clear();
  }
  return 1;
}

size_t SimplifyMaskingAdd(std::function<bool(OpWrapper&)> validate_op_config,
                          std::vector<OpWrapper>& ops, size_t start_index,
                          TensorPool& tensor_pool, size_t pattern_size) {
  constexpr size_t kMaskingPreReshapeIndex = 0;
  constexpr size_t kMaskingAddIndex = 1;
  constexpr size_t kMaskingPostReshapeIndex = 2;
  if (!(IS_CONNECTED(kMaskingPreReshapeIndex, 0, kMaskingAddIndex, 0)) &&
      (IS_CONNECTED(kMaskingAddIndex, 0, kMaskingPostReshapeIndex, 0))) {
    return 1;
  }
  auto& add_input =
      ops[start_index + kMaskingPreReshapeIndex].GetInputTensor(0);
  auto mask = &ops[start_index + kMaskingAddIndex].GetInputTensor(1);
  QNN_LOG_INFO("[G2G] Simplify masking");
  std::vector<OpWrapper> new_ops;
  for (size_t index = 0; index < mask->GetRank(); ++index) {
    size_t mask_dim = mask->GetDimension(index);
    size_t input_dim = add_input.GetDimension(index);
    if (!(mask_dim == input_dim || mask_dim == 1 || input_dim == 1)) {
      std::vector<qnn::ConstTensorWrapperRef> inputs;
      size_t broadcast_size = input_dim / mask_dim;
      inputs.reserve(broadcast_size);
      for (size_t i = 0; i < broadcast_size; ++i) {
        inputs.emplace_back(*mask);
      }
      auto new_dims = mask->GetDimensions();
      new_dims[index] = input_dim;
      mask = &tensor_pool.CloneNativeTensorFrom(*mask, new_dims);
      new_ops.emplace_back(CreateConcatenationOp(inputs, *mask, index));
      QNN_LOG_INFO("[G2G] Simplify masking w/ Add @ %d", index);
      break;
    }
  }
  new_ops.emplace_back(CreateElementWiseAddOp(
      add_input, *mask,
      ops[start_index + kMaskingPostReshapeIndex].GetOutputTensor(0)));
  CloneNamespace(ops[start_index], new_ops);
  const bool is_valid =
      std::all_of(new_ops.begin(), new_ops.end(),
                  [validate_op_config](OpWrapper& op_wrapper) -> bool {
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
  } else {
    QNN_LOG_WARNING(
        "[G2G] Validation failed. Rolling back to the original graph.");
  }
  return 1;
}

size_t DuplicateOrRemoveConcate(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  constexpr size_t kMaskingConcatIndex = 0;
  constexpr size_t kMaskingAddIndex = 1;
  const auto& concat_op = ops[start_index + kMaskingConcatIndex];
  const auto& concat_op_name = concat_op.GetName();
  const auto& mask = concat_op.GetInputTensor(0);
  const auto& concat_output = concat_op.GetOutputTensor(0);
  const auto& add_op = ops[start_index + kMaskingAddIndex];
  if (!(IS_CONNECTED(kMaskingConcatIndex, 0, kMaskingAddIndex, 1) &&
        IsElementWiseAdd(add_op))) {
    return 1;
  }
  // Check if this concat is only for broadcast.
  size_t num_elements = concat_output.GetTensorNumElements();
  size_t input_cnt = 0;
  while (num_elements > 0) {
    auto& concat_input = concat_op.GetInputTensor(input_cnt);
    if (concat_input != mask) {
      return 1;
    }
    num_elements -= concat_input.GetTensorNumElements();
    input_cnt++;
  }

  // Find all add indices.
  std::vector<size_t> indices;
  for (size_t i = ops.size() - 1; i-- > 0;) {
    if (IsElementWiseAdd(ops[i]) && concat_output == ops[i].GetInputTensor(1)) {
      indices.push_back(i);
    }
  }
  if (indices.size() <= 1) {
    return 1;
  }

  bool can_remove_concat = true;
  for (size_t index = 0; index < mask.GetRank(); ++index) {
    size_t mask_dim = mask.GetDimension(index);
    size_t input_dim = add_op.GetInputTensor(0).GetDimension(index);
    if (!(mask_dim == input_dim || mask_dim == 1 || input_dim == 1)) {
      can_remove_concat = false;
      break;
    }
  }

  QNN_LOG_INFO("[G2G] %s concat", can_remove_concat ? "Remove" : "Duplicate");
  if (can_remove_concat) {
    for (size_t i : indices) {
      // Add
      auto add = CreateElementWiseAddOp(ops[i].GetInputTensor(0), mask,
                                        ops[i].GetOutputTensor(0));
      CloneNamespace(ops[i], add);
      ops[i] = std::move(add);
      ops[i].AddSuffixToName(absl::StrCat("_qcg2g_", i));
      validate_op_config(ops[i]);
    }
  } else {
    // Add concat ops.
    const std::vector<qnn::ConstTensorWrapperRef> concat_inputs(
        input_cnt, concat_op.GetInputTensor(0));
    for (size_t i : indices) {
      const auto& duplicated_concat_output =
          tensor_pool.CloneNativeTensorFrom(concat_output);
      // Add
      auto add = CreateElementWiseAddOp(ops[i].GetInputTensor(0),
                                        duplicated_concat_output,
                                        ops[i].GetOutputTensor(0));
      CloneNamespace(ops[i], add);
      ops[i] = std::move(add);
      ops[i].AddSuffixToName(absl::StrCat("_qcg2g_", i));
      validate_op_config(ops[i]);

      // Concat
      auto concat = CreateOpWithSameParams(concat_op, concat_inputs,
                                           {duplicated_concat_output});
      CloneNamespace(ops[i], concat);
      ops.insert(ops.begin() + i, concat);
      validate_op_config(ops[i]);
      ops[i].AddSuffixToName(absl::StrCat("_qcg2g_", i));
    }
  }
  // TODO(jiunkaiy): Can be more efficient, but keep this saftest way for now.
  ops.erase(std::remove_if(ops.begin(), ops.end(),
                           [&ops, &concat_op_name](const auto& op) {
                             return op.GetName() == concat_op_name;
                           }),
            ops.end());
  return pattern_size;
}

size_t FuseConcatReshape(std::function<bool(OpWrapper&)> validate_op_config,
                         std::vector<OpWrapper>& ops, size_t start_index,
                         TensorPool& tensor_pool, size_t pattern_size) {
  constexpr size_t kFusedConcatIndex = 0;
  constexpr size_t kFusedReshapeIndex = 1;
  auto& concat = ops[start_index + kFusedConcatIndex];
  auto& reshape = ops[start_index + kFusedReshapeIndex];

  // Connection check
  if (!IS_CONNECTED(kFusedConcatIndex, 0, kFusedReshapeIndex, 0)) {
    return 1;
  }
  // Check the indcies with different dim.
  const auto& reshape_input_dims = reshape.GetInputTensor(0).GetDimensions();
  const auto& reshape_output_dims = reshape.GetOutputTensor(0).GetDimensions();
  std::vector<size_t> diff_indices;
  diff_indices.reserve(
      std::max(reshape_output_dims.size(), reshape_output_dims.size()));
  for (size_t i = 0; i < reshape_input_dims.size(); ++i) {
    if (reshape_input_dims[i] != reshape_output_dims[i]) {
      diff_indices.emplace_back(i);
    }
  }
  constexpr size_t kDiffIndices = 2;
  if (diff_indices.size() != kDiffIndices ||
      reshape_output_dims[diff_indices[0]] != 1) {
    return 1;
  }

  // Change concat axis form diff_indices[0] to diff_indices[1], and remove
  // reshape. Example: 2, 128 -> 1, 256 => concat at the 2nd axis instead of the
  // 1st.
  QNN_LOG_INFO("[G2G] convert-reshape fusion");
  std::vector<ConstTensorWrapperRef> concat_inputs;
  concat_inputs.reserve(concat.GetInputTensorSize());
  for (size_t i = 0; i < concat.GetInputTensorSize(); ++i) {
    concat_inputs.emplace_back(concat.GetInputTensor(i));
  }
  auto new_concat = CreateConcatenationOp(
      concat_inputs, {reshape.GetOutputTensor(0)}, diff_indices[1]);
  new_concat.AddSuffixToName(absl::StrCat("_qcg2g_0"));
  CloneNamespace(concat, new_concat);
  if (validate_op_config(new_concat)) {
    ops.erase(ops.begin() + start_index,
              ops.begin() + start_index + pattern_size);
    ops.emplace(ops.begin() + start_index, std::move(new_concat));
  } else {
    QNN_LOG_WARNING(
        "[G2G] Validation failed. Rolling back to the original graph.");
  }
  return 1;
}

size_t TileMatMul(std::function<bool(OpWrapper&)> validate_op_config,
                  std::vector<OpWrapper>& ops, size_t start_index,
                  TensorPool& tensor_pool, size_t pattern_size) {
  constexpr size_t kLargeMatMulIndex = 0;
  auto& matmul = ops[start_index + kLargeMatMulIndex];
  auto& input = matmul.GetInputTensor(0);
  auto& input_dims = input.GetDimensions();
  auto& large_cache = matmul.GetInputTensor(1);
  auto& large_cache_dims = large_cache.GetDimensions();
  auto& output = matmul.GetOutputTensor(0);
  // Size check
  if (large_cache_dims.size() != 4) {
    return 1;
  }
  std::vector<OpWrapper> new_ops;
  // Orignal graph:
  // In[0]: [..., M, K]
  // In[1]: [..., K, N]
  // Out[0]: [..., M, N]
  QNN_LOG_INFO("(M, K, N) = (%d, %d, %d)", input_dims[2], input_dims[3], large_cache_dims[3]);
  using GemmDim = std::tuple<size_t, size_t, size_t>;
  GemmDim mkn(input_dims[2], input_dims[3], large_cache_dims[3]);
  size_t tiling_size = 0;
//   if (mkn == GemmDim(128, 256, 4095)) {
//     tiling_size = 32;
//   } else if (mkn == GemmDim(128, 4095, 256)) {
//     tiling_size = 128;
//   } else if (mkn == GemmDim(128, 4095, 512)) {
//     tiling_size = 64;
//   } else if (mkn == GemmDim(128, 512, 4095)) {
//     tiling_size = 32;
//   } else if (mkn == GemmDim(1, 256, 4095)) {
//     tiling_size = 0;
//   } else if (mkn == GemmDim(1, 4095, 256)) {
//     tiling_size = 64;
//   } else if (mkn == GemmDim(1, 4095, 512)) {
//     tiling_size = 64;
//   } else if (mkn == GemmDim(1, 512, 4095)) {
//     tiling_size = 32;
//   }
  if (mkn == GemmDim(128, 4095, 512)) {
    tiling_size = 256;
  }
  if (mkn == GemmDim(128, 4095, 256)) {
    tiling_size = 256;
  }
  if (tiling_size == 0) {
    QNN_LOG_WARNING("tiling size = 0, skipping G2G");
  }
  if (large_cache_dims[3] == 4095) {
    QNN_LOG_INFO(
        "Found MatMul (Tile axis: 3):\n>> Name: %s\n  In[0] Dims: [%d, %d, "
        "%d, %d]\n  In[1] Dims: [%d, %d, %d, %d]",
        matmul.GetName().data(), input_dims[0], input_dims[1], input_dims[2],
        input_dims[3], large_cache_dims[0], large_cache_dims[1],
        large_cache_dims[2], large_cache_dims[3]);
    static constexpr size_t kSplitMatMulConcatTileSize = 32;
    size_t tsize = (tiling_size) ? tiling_size : kSplitMatMulConcatTileSize;
    // After G2G:
    // Concat(
    //   ([..., M, K] * [..., K, Nt]),
    //   ([..., M, K] * [..., K, Nt]),
    //   ...,)
    // = [..., M, N]
    auto k = SplitTensor(tensor_pool, new_ops, matmul.GetInputTensor(1), 3, tsize);
    std::vector<ConstTensorWrapperRef> matmul_outputs;
    for (size_t i = 0; i < k.size(); ++i) {
      auto output_dims = output.GetDimensions();
      output_dims[3] = k[i].get().GetDimension(3);
      auto& matmul_output = matmul_outputs.emplace_back(
          tensor_pool.CloneNativeTensorFrom(output, output_dims));
      new_ops.emplace_back(CreateOpWithSameParams(
          matmul, {matmul.GetInputTensor(0), k[i]}, {matmul_output}));
    }
    new_ops.emplace_back(CreateConcatenationOp(matmul_outputs, output, 3));
  } else if (input_dims[3] == 4095 && large_cache_dims[2] == 4095) {
    QNN_LOG_INFO(
        "Found MatMul (Tile axis: 2):\n>> Name: %s\n  In[0] Dims: [%d, %d, %d, "
        "%d]\n  In[1] Dims: [%d, %d, %d, %d]",
        matmul.GetName().data(), input_dims[0], input_dims[1], input_dims[2],
        input_dims[3], large_cache_dims[0], large_cache_dims[1],
        large_cache_dims[2], large_cache_dims[3]);
    // After G2G:
    // ([..., M, Kt] * [..., Kt, N])
    // + ([..., M, Kt] * [..., Kt, N])
    // + ...
    // = [..., M, N]
    static constexpr size_t kSplitMatMulAddTileSize = 128;
    size_t tsize = (tiling_size) ? tiling_size : kSplitMatMulAddTileSize;
    auto inputs =
        SplitTensor(tensor_pool, new_ops, matmul.GetInputTensor(0), 3, tsize);
    auto v =
        SplitTensor(tensor_pool, new_ops, matmul.GetInputTensor(1), 2, tsize);
    // Create one MatMul per (inputs[i], v[i]) split pair.
    // Each produces [..., M, N] — same shape as the full output.
    std::vector<ConstTensorWrapperRef> matmul_outputs;
    matmul_outputs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto& mm_output = matmul_outputs.emplace_back(
          tensor_pool.CloneNativeTensorFrom(output, output.GetDimensions()));
      new_ops.emplace_back(
          CreateOpWithSameParams(matmul, {inputs[i], v[i]}, {mm_output}));
    }
    // Define QNN_MATMUL_ADDER_TREE to use a balanced binary adder tree
    // (lower graph depth, better parallelism on NPU).
    // Leave undefined to use a sequential left-to-right reduction chain
    // (simpler graph, same arithmetic result).
// #define QNN_MATMUL_ADDER_TREE
    if (matmul_outputs.size() == 1) {
      // Only one tile (K <= kSplitMatMulAddTileSize): redo the sole matmul
      // directly into `output` instead of the intermediate tensor.
      new_ops.pop_back();
      new_ops.emplace_back(
          CreateOpWithSameParams(matmul, {inputs[0], v[0]}, {output}));
    } else {
#ifdef QNN_MATMUL_ADDER_TREE
      // Binary adder tree: pair up results level by level until one remains.
      // Depth = ceil(log2(N)); odd element at each level is promoted as-is.
      std::vector<ConstTensorWrapperRef> add_level = matmul_outputs;
      while (add_level.size() > 2) {
        std::vector<ConstTensorWrapperRef> next_level;
        next_level.reserve((add_level.size() + 1) / 2);
        for (size_t i = 0; i + 1 < add_level.size(); i += 2) {
          auto& add_output = next_level.emplace_back(
              tensor_pool.CloneNativeTensorFrom(output, output.GetDimensions()));
          new_ops.emplace_back(CreateElementWiseAddOp(
              add_level[i], add_level[i + 1], add_output));
        }
        if (add_level.size() % 2 == 1) {
          next_level.push_back(add_level.back());
        }
        add_level = std::move(next_level);
      }
      // Final pair — wire directly to the original `output` tensor.
      new_ops.emplace_back(
          CreateElementWiseAddOp(add_level[0], add_level[1], output));
#else
      // Sequential chain: accumulate left-to-right.
      // acc = matmul_outputs[0] + matmul_outputs[1] + ... + matmul_outputs[N-1]
      ConstTensorWrapperRef acc = matmul_outputs[0];
      for (size_t i = 1; i + 1 < matmul_outputs.size(); ++i) {
        auto& add_output =
            tensor_pool.CloneNativeTensorFrom(output, output.GetDimensions());
        new_ops.emplace_back(
            CreateElementWiseAddOp(acc, matmul_outputs[i], add_output));
        acc = add_output;
      }
      // Final add — wire directly to the original `output` tensor.
      new_ops.emplace_back(CreateElementWiseAddOp(
          acc, matmul_outputs.back(), output));
#endif  // QNN_MATMUL_ADDER_TREE
    }
  } else {
    return 1;
  }
  static size_t matmul_tiling_cnt = 0;
  CloneNamespace(matmul, new_ops,
                 absl::StrCat("matmul_tiling_", matmul_tiling_cnt++));
  // Validate new graph.
  const bool is_valid =
      std::all_of(new_ops.begin(), new_ops.end(),
                  [validate_op_config](::qnn::OpWrapper& op_wrapper) -> bool {
                    return validate_op_config(op_wrapper);
                  });
  if (is_valid) {
    // Adjust the name to avoid a name collision in the Qnn JSON dump.
    for (size_t i = 0; i < new_ops.size(); ++i) {
      new_ops[i].AddSuffixToName(
          absl::StrCat("_qcg2g_", i * matmul_tiling_cnt));
    }
    // Replace the matched pattern with a newly generated subgraph.
    size_t step_size = new_ops.size();
    ops.insert(ops.begin() + start_index + pattern_size,
               std::make_move_iterator(new_ops.begin()),
               std::make_move_iterator(new_ops.end()));
    ops.erase(ops.begin() + start_index,
              ops.begin() + start_index + pattern_size);
    QNN_LOG_INFO("[G2G] Matmul Tiling done.");
    return step_size;
  }
  QNN_LOG_WARNING(
      "[G2G] Validation failed. Rolling back to the original graph.");
  return 1;
}

size_t RevertMulTranspose(std::function<bool(OpWrapper&)> validate_op_config,
                          std::vector<OpWrapper>& ops, size_t start_index,
                          TensorPool& tensor_pool, size_t pattern_size) {
  static constexpr size_t kRevertedMulIndex = 0;
  static constexpr size_t kRevertedTransposeIndex = 1;
  const auto& mul = ops[start_index + kRevertedMulIndex];
  const auto& transpose = ops[start_index + kRevertedTransposeIndex];

  if (!IsElementWiseMultiply(mul)) return 1;
  if (!IS_CONNECTED(kRevertedMulIndex, 0, kRevertedTransposeIndex, 0)) return 1;
  const auto& const_val =
      ops[start_index + kRevertedMulIndex].GetInputTensor(1);
  if (!const_val.IsTensorStatic()) return 1;
  const auto& const_val_dims = const_val.GetDimensions();
  if (!(const_val_dims.size() == 1 && const_val_dims[0] == 1)) return 1;
  QNN_LOG_INFO("[G2G] Revert Mul Transpose");
  const auto& transpose_out = tensor_pool.CloneNativeTensorFrom(
      mul.GetInputTensor(0), transpose.GetOutputTensor(0).GetDimensions());
  std::vector<OpWrapper> new_ops =
      MakeVector(CreateOpWithSameParams(transpose, {mul.GetInputTensor(0)},
                                        {transpose_out}),
                 CreateElementWiseMulOp(transpose_out, const_val,
                                        transpose.GetOutputTensor(0)));
  // Validate new graph.
  const bool is_valid =
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
    QNN_LOG_INFO("[G2G] Revert Mul Transpose done.");
    return step_size;
  }
  QNN_LOG_WARNING(
      "[G2G] Validation failed. Rolling back to the original graph.");
  return 1;
}
}  // namespace qnn
