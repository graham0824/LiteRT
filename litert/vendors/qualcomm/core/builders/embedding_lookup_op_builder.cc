// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/embedding_lookup_op_builder.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/builders/custom_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/builders/quantize_op_builder.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {
namespace {
constexpr int kTableIdx = 1;
constexpr int kIndicesIdx = 0;
constexpr int kOutputIdx = 0;
constexpr std::int32_t kGatherDefaultAxis = 0;
// QNN op type name defined in the custom embedding op package XML.
constexpr char kEmbeddingCustomOpType[] = "EMBEDDING";
// Tile size (in codes) used by the HVX kernel's self-packed weight format.
constexpr uint32_t kTileSize = 512;

// Converts V×D int2 codes (stored as int8, natural order) to the HVX kernel's
// self-packed uint8 format: D/4 bytes per row, 512-code tile permutation.
//
// Tile permutation: stored_pos_in_tile = (d_in_tile % 64) * 8 + (d_in_tile / 64)
// Crumb encoding:   crumb = ((code + 2) ^ 2) & 0x3
// Byte packing:     little-endian crumbs, 4 per byte
std::vector<uint8_t> PackWeightToTilePermuted(
    absl::Span<const int8_t> codes, uint32_t V, uint32_t D) {
  QNN_LOG_INFO("(V ,D) = (%u, %u)", V, D);
  QNN_LOG_INFO("Codes size = %zu", codes.size());
  for (size_t i=0; i< 10; ++i) {
    QNN_LOG_INFO(">> %d", codes[i]);
  }
  // const uint32_t bytes_per_row = D / 4;
  std::vector<uint8_t> packed(codes.size()/4, 0);
  static constexpr size_t kNumCodesPerInt8 = 4;
  for (size_t i = 0; i < packed.size(); ++i) {
    for (size_t j = 0; j < kNumCodesPerInt8; ++j) {
      // tile[ (q%8)·64 + q/8 ]
      size_t index = i * kNumCodesPerInt8 + j;
    }
    packed[i] = 
  }
  for (auto& c: packed) {

  }

  // for (uint32_t v = 0; v < V; ++v) {
  //   uint8_t* row = packed.data() + v * bytes_per_row;
  //   const int8_t* src = codes.data() + v * D;
  //   for (uint32_t d = 0; d < D; ++d) {
  //     const uint8_t crumb =
  //         static_cast<uint8_t>((static_cast<int32_t>(src[d]) + 2) ^ 2) & 0x3u;
  //     const uint32_t d_in_tile = d % kTileSize;
  //     const uint32_t tile_base = (d / kTileSize) * kTileSize;
  //     const uint32_t stored_pos =
  //         tile_base + (d_in_tile % 64u) * 8u + (d_in_tile / 64u);
  //     row[stored_pos / 4u] |= crumb << ((stored_pos % 4u) * 2u);
  //   }
  // }
  return packed;
}
}  // namespace

OpWrapper CreateGatherOp(const TensorWrapper& table,
                         const TensorWrapper& indices,
                         const TensorWrapper& output, std::int32_t axis) {
  OpWrapper op(GetUniqueOpName(QNN_OP_GATHER), QNN_OP_GATHER,
               QnnOpCode::kGather);
  op.AddInputTensor(table);
  op.AddInputTensor(indices);
  op.AddOutputTensor(output);
  op.AddScalarParam<std::int32_t>(QNN_OP_GATHER_PARAM_AXIS, axis);
  return op;
}

std::vector<OpWrapper> BuildEmbeddingLookupOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  const TensorWrapper* table_tensor = &(inputs[kTableIdx].get());
  const TensorWrapper& indices_tensor = inputs[kIndicesIdx];
  const TensorWrapper& output_tensor = outputs[kOutputIdx];

  // Case: QInt8 table with QInt16 output
  if (table_tensor->IsQuantI8() && output_tensor.IsQuantI16()) {
    QNN_LOG_WARNING(
        "The data type of embedding lookup table is int8, but output data type "
        "is int16. Int8 table will be cast to int16.");
    std::vector<std::int16_t> int16_data;
    size_t data_len = table_tensor->GetTensorNumElements();
    auto int8_data = table_tensor->GetTensorData<std::int8_t>();
    if (!int8_data.has_value()) {
      QNN_LOG_ERROR("Embedding lookup get int8 table failed.");
      return {};
    }
    int16_data.reserve(data_len);
    for (size_t i = 0; i < data_len; ++i) {
      int16_data.emplace_back(static_cast<std::int16_t>((*int8_data)[i]));
    }

    table_tensor = &tensor_pool.CreateStaticTensor(
        output_tensor.GetDataType(), table_tensor->GetQuantParams(),
        table_tensor->GetDimensions(),
        sizeof(decltype(int16_data)::value_type) * int16_data.size(),
        reinterpret_cast<void*>(int16_data.data()));
  }

  const auto& table_quant_params = table_tensor->GetQuantParams();
  if (table_quant_params == output_tensor.GetQuantParams()) {
    return MakeVector(CreateGatherOp(*table_tensor, indices_tensor,
                                     output_tensor, kGatherDefaultAxis));
  }
  QNN_LOG_WARNING(
      "Add a Convert op after the Gather op since the table's quant params do "
      "not match the output's.");
  const auto& gather_output =
      tensor_pool.CloneNativeTensorFrom(output_tensor, table_quant_params);
  return MakeVector(CreateGatherOp(*table_tensor, indices_tensor, gather_output,
                                   kGatherDefaultAxis),
                    CreateConvertOp(gather_output, output_tensor));
}

std::vector<OpWrapper> BuildEmbeddingLookupFpa2wOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs,
    const CustomOpPackage& custom_op_package) {
  if (custom_op_package.name.empty()) {
    QNN_LOG_ERROR(
        "BuildEmbeddingLookupFpa2wOp: custom_op_package.name is empty. "
        "Set it via QualcommOptions::SetCustomOpPackage().");
    return {};
  }

  const TensorWrapper& indices_tensor = inputs[kIndicesIdx];
  const TensorWrapper& weight_tensor = inputs[kTableIdx];
  const TensorWrapper& output_tensor = outputs[kOutputIdx];

  const auto* bw_params = std::get_if<BwAxisScaleOffsetQuantizeParamsWrapper>(
      &weight_tensor.GetQuantParams());
  if (bw_params == nullptr) {
    QNN_LOG_ERROR(
        "BuildEmbeddingLookupFpa2wOp: weight tensor lacks "
        "BwAxisScaleOffsetQuantizeParams.");
    return {};
  }

  const auto& dims = weight_tensor.GetDimensions();
  if (dims.size() < 2) {
    QNN_LOG_ERROR(
        "BuildEmbeddingLookupFpa2wOp: weight tensor must be at least 2-D.");
    return {};
  }
  const uint32_t V = dims[0];
  const uint32_t D = dims[1];
  if (D % 4 != 0) {
    QNN_LOG_ERROR(
        "BuildEmbeddingLookupFpa2wOp: embedding dimension D=%u is not "
        "divisible by 4.",
        D);
    return {};
  }

  // Extract int8 codes (already unpacked from int2 by TensorWrapper on load).
  const auto int8_data = weight_tensor.GetTensorData<int8_t>();
  if (!int8_data.has_value()) {
    QNN_LOG_ERROR(
        "BuildEmbeddingLookupFpa2wOp: failed to get int8 weight data.");
    return {};
  }

  // Pack weight to HVX self-packed tile-permuted uint8 format.
  std::vector<uint8_t> packed = PackWeightToTilePermuted(*int8_data, V, D);

  // Extract per-channel scales.
  std::vector<float> scales = bw_params->GetScales();
  if (scales.size() != V) {
    QNN_LOG_ERROR(
        "BuildEmbeddingLookupFpa2wOp: expected %u per-channel scales, got %zu.",
        V, scales.size());
    return {};
  }

  // Create static tensor for packed weight: shape (V, D/4), dtype uint8.
  const std::vector<uint32_t> packed_dims = {V, D / 4u};
  const TensorWrapper& packed_weight_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_8, UndefinedQuantizeParamsWrapper{}, packed_dims,
      static_cast<uint32_t>(packed.size() * sizeof(uint8_t)), packed.data());

  // Create static tensor for scales: shape (V,), dtype float32.
  const std::vector<uint32_t> scales_dims = {V};
  const TensorWrapper& scales_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, UndefinedQuantizeParamsWrapper{}, scales_dims,
      static_cast<uint32_t>(scales.size() * sizeof(float)), scales.data());

  return MakeVector(
      CreateCustomOp(custom_op_package.name.c_str(), kEmbeddingCustomOpType,
                     {indices_tensor, packed_weight_tensor, scales_tensor},
                     {output_tensor}));
}

}  // namespace qnn
