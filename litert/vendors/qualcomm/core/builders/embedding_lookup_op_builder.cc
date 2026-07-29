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
// The two HVX tiles the tile-generic kernel (4_hvx_tile_generic) decodes: full
// 512-code tiles, then an OPTIONAL single 256-code tail. A row is laid out as
// n512 = D/512 full tiles followed by one 256-tail iff D % 512 == 256, so D need
// only be a multiple of 256.
constexpr uint32_t kTile512 = 512;
constexpr uint32_t kTile256 = 256;

// Logical in-row index L -> physical crumb position within the packed row, for a
// row laid out as n512 full 512-tiles then an optional 256-tail. This is the
// INVERSE (scatter form) of the kernel's tile_order_{512,256}() gather: within a
// W-wide tile (W = tile/8: 64 for the 512 tile, 32 for the 256 tail),
//     phys = tile_base + (l % W) * 8 + l / W.
// For D % 512 == 0 this is byte-identical to the previous global-512 mapping
// (ind%64)*8 + ind/64, so the existing D=multiple-of-512 path is unchanged.
uint32_t LogicalToPhysicalInRow(uint32_t L, uint32_t D) {
  const uint32_t n512 = D / kTile512;
  const uint32_t tail_base = n512 * kTile512;
  if (L < tail_base) {                              // inside a full 512-tile
    const uint32_t tile = L / kTile512;
    const uint32_t l = L % kTile512;
    return tile * kTile512 + (l % 64u) * 8u + l / 64u;   // W = 64
  }
  const uint32_t l = L - tail_base;                 // inside the 256-tail, 0..255
  return tail_base + (l % 32u) * 8u + l / 32u;      // W = 32
}

// Converts V×D int2 codes (stored as int8, natural/logical order) to the HVX
// kernel's self-packed uint8 format: D/4 bytes per row, per-tile permutation.
// Packs PER ROW (not over a global V*D flatten) so a 256-tail that ends mid-row
// is handled correctly. D must be a multiple of 256.
std::vector<std::uint8_t> PackWeightToTilePermuted(
    absl::Span<const int8_t> codes, uint32_t V, uint32_t D) {
  QNN_LOG_INFO("(V ,D) = (%u, %u)", V, D);
  QNN_LOG_INFO("Codes size = %zu", codes.size());
  if (D % kTile256 != 0) {
    QNN_LOG_ERROR("D=%u must be a multiple of %u (n*512 + optional 256-tail)", D,
                  kTile256);
    return {};
  }
  const uint32_t bytes_per_row = D / 4u;
  std::vector<std::uint8_t> packed(static_cast<size_t>(V) * bytes_per_row, 0);
  for (uint32_t v = 0; v < V; ++v) {
    const size_t code_base = static_cast<size_t>(v) * D;
    const size_t byte_base = static_cast<size_t>(v) * bytes_per_row;
    for (uint32_t L = 0; L < D; ++L) {
      const uint32_t q = LogicalToPhysicalInRow(L, D);    // physical crumb pos
      const size_t packed_index = byte_base + q / 4u;
      const uint32_t num_left_shift = 2u * (q % 4u);
      packed[packed_index] &= ~(0b11u << num_left_shift);
      packed[packed_index] |= ((codes[code_base + L] & 0b11) << num_left_shift);
    }
  }
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
// #define DQ_FC
std::vector<OpWrapper> BuildEmbeddingLookupFpa2wOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs,
    const CustomOpPackage& custom_op_package) {
#ifndef DQ_FC
  // The custom-HVX-op path needs a configured op package; the DQ_FC reference
  // path (below) emits a plain Gather and does not.
  if (custom_op_package.name.empty()) {
    QNN_LOG_ERROR(
        "BuildEmbeddingLookupFpa2wOp: custom_op_package.name is empty. "
        "Set it via QualcommOptions::SetCustomOpPackage().");
    return {};
  }
#endif  // !DQ_FC

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

  // Extract int8 codes (already unpacked from int2 by TensorWrapper on load).
  const auto int8_data = weight_tensor.GetTensorData<int8_t>();
  if (!int8_data.has_value()) {
    QNN_LOG_ERROR(
        "BuildEmbeddingLookupFpa2wOp: failed to get int8 weight data.");
    return {};
  }

  // Extract per-channel scales (one per row / channel on axis 0).
  std::vector<float> scales = bw_params->GetScales();
  if (scales.size() != V) {
    QNN_LOG_ERROR(
        "BuildEmbeddingLookupFpa2wOp: expected %u per-channel scales, got %zu.",
        V, scales.size());
    return {};
  }

#ifdef DQ_FC
  // Debug / accuracy-reference path: dequantize the per-channel int2 weight into
  // a plain fp32 embedding table at compile time and emit a standard Gather.
  // Selected by building with -DDQ_FC; needs no HVX custom op package.
  const std::vector<int32_t> zero_points = bw_params->GetZeroPoints();
  if (zero_points.size() != V) {
    QNN_LOG_ERROR(
        "BuildEmbeddingLookupFpa2wOp: expected %u per-channel zero points, got "
        "%zu.",
        V, zero_points.size());
    return {};
  }

  // Dequantized weight is defined as: S * (q - Z), applied per channel (row).
  std::vector<float> dequant(static_cast<size_t>(V) * D);
  for (uint32_t v = 0; v < V; ++v) {
    for (uint32_t d = 0; d < D; ++d) {
      const size_t idx = static_cast<size_t>(v) * D + d;
      dequant[idx] = Dequantize((*int8_data)[idx], scales[v], zero_points[v]);
    }
  }

  // Create static fp32 embedding table: shape (V, D).
  const std::vector<uint32_t> table_dims = {V, D};
  const TensorWrapper& dequant_table_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, UndefinedQuantizeParamsWrapper{}, table_dims,
      static_cast<uint32_t>(dequant.size() * sizeof(float)), dequant.data());

  // Output is already fp32, so no Convert op is needed after the Gather.
  return MakeVector(CreateGatherOp(dequant_table_tensor, indices_tensor,
                                   output_tensor, kGatherDefaultAxis));
#else   // DQ_FC
  if (D % kTile256 != 0) {
    QNN_LOG_ERROR(
        "BuildEmbeddingLookupFpa2wOp: embedding dimension D=%u is not "
        "a multiple of %u (n*512 full tiles + optional one 256-tail).",
        D, kTile256);
    return {};
  }

  // Pack weight to HVX self-packed tile-permuted uint8 format.
  std::vector<uint8_t> packed = PackWeightToTilePermuted(*int8_data, V, D);
  if (packed.empty()) {
    QNN_LOG_ERROR("Pack weight failure...");
    return {};
  }
  char packed0_bits[9];
  for (int b = 0; b < 8; ++b) {
    packed0_bits[b] = (packed[0] & (1 << (7 - b))) ? '1' : '0';
  }
  packed0_bits[8] = '\0';
  QNN_LOG_INFO("packed[0] = 0b%s", packed0_bits);

  // Create static tensor for packed weight: shape (V, D/4), dtype uint8.
  const std::vector<uint32_t> packed_dims = {V, D / 4u};
  QNN_LOG_INFO("checked packed: %d", packed.size() == V* D / 4u);
  const TensorWrapper& packed_weight_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_8, UndefinedQuantizeParamsWrapper{}, packed_dims,
      static_cast<uint32_t>(packed.size() * sizeof(uint8_t)), packed.data());

  // Create static tensor for scales: shape (V,), dtype float32.
  const std::vector<uint32_t> scales_dims = {V};
  QNN_LOG_INFO("checked scale: %d", scales.size() == V);
  const TensorWrapper& scales_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, UndefinedQuantizeParamsWrapper{}, scales_dims,
      static_cast<uint32_t>(scales.size() * sizeof(float)), scales.data());
  QNN_LOG_INFO("scales[0]: %f", scales[0]);
  return MakeVector(
      CreateCustomOp(custom_op_package.name.c_str(), kEmbeddingCustomOpType,
                     {indices_tensor, packed_weight_tensor, scales_tensor},
                     {output_tensor}));
#endif  // DQ_FC
}

}  // namespace qnn
