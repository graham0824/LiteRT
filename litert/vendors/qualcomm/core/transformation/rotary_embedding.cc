// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/rotary_embedding.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <variant>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/builders/quantize_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/builders/slice_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/split_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace qnn {

namespace {

OpWrapper CreateRotaryEmbeddingOp(const TensorWrapper& x_b1sd,
                                  const TensorWrapper& cos_half,
                                  const TensorWrapper& sin_half,
                                  const TensorWrapper& output,
                                  std::uint32_t head_size) {
  OpWrapper op(GetUniqueOpName(QNN_OP_ROTARY_EMBEDDING),
               QNN_OP_ROTARY_EMBEDDING, QnnOpCode::kRotaryEmbedding);
  op.AddInputTensor(x_b1sd);
  op.AddInputTensor(cos_half);
  op.AddInputTensor(sin_half);
  op.AddOutputTensor(output);
  op.AddScalarParam<bool>(QNN_OP_ROTARY_EMBEDDING_PARAM_INTERLEAVED, false);
  op.AddScalarParam<std::uint32_t>(
      QNN_OP_ROTARY_EMBEDDING_PARAM_ROTARY_EMBEDDING_DIM, head_size);
  return op;
}

// Reads StridedSlice ranges param into a vector, or returns nullopt.
std::optional<std::vector<std::int32_t>> ReadSliceRanges(
    const OpWrapper& slice) {
  auto span = slice.GetTensorParam(0).GetTensor().GetTensorData<std::int32_t>();
  if (!span) return std::nullopt;
  return std::vector<std::int32_t>(span->begin(), span->end());
}

// Creates a UFIXED8 counterpart for a SFIXED8 tensor and emits the Convert.
// SFIXED8 zero_point z maps to UFIXED8 zero_point z+128 with the same scale.
// Returns the UFIXED8 output tensor; pushes the Convert op into new_ops.
const TensorWrapper& ToUFixed8(
    const TensorWrapper& src, std::vector<OpWrapper>& new_ops,
    TensorPool& tensor_pool) {
  const auto* so =
      std::get_if<ScaleOffsetQuantizeParamsWrapper>(&src.GetQuantParams());
  if (!so) {
    QNN_LOG_ERROR("[G2G] RoPE+T: ToUFixed8 called on non-scale-offset tensor.");
    return src;
  }
  QuantizeParamsWrapperVariant ufixed_quant;
  ufixed_quant.emplace<ScaleOffsetQuantizeParamsWrapper>(
      so->GetScale(), so->GetOffset() + 128);
  auto& dst = tensor_pool.CreateNativeTensor(QNN_DATATYPE_UFIXED_POINT_8,
                                             ufixed_quant, src.GetDimensions());
  new_ops.emplace_back(CreateConvertOp(src, dst));
  return dst;
}

// Returns true if the Transpose op has perm [0,2,1,3].
bool IsPerm0213(const OpWrapper& transpose) {
  const auto perm_param = transpose.GetTensorParam(0).GetTensor();
  const auto perm_data = perm_param.GetTensorData<uint32_t>();
  return perm_data.has_value() && perm_data->size() == 4 &&
         (*perm_data)[0] == 0 && (*perm_data)[1] == 2 &&
         (*perm_data)[2] == 1 && (*perm_data)[3] == 3;
}

}  // namespace

size_t FuseRotaryEmbeddingWithTranspose(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  // Pattern indices (9 ops):
  //   0: Convert      x: F16/F32 → UFIXED8
  //   1: StridedSlice x1 = x[..., 0:D/2]
  //   2: StridedSlice x2 = x[..., D/2:D]
  //   3: Concat([x2, x1], axis=-1)
  //   4: ElementWiseBinary MUL(x, cos)
  //   5: ElementWiseBinary MUL(concat, sin)
  //   6: ElementWiseBinary ADD
  //   7: Convert      output quantize
  //   8: Transpose perm=[0,2,1,3]: [B,S,H,D] → [B,H,S,D]
  constexpr size_t kConvertIdx    = 0;
  constexpr size_t kSlice1Idx     = 1;
  constexpr size_t kSlice2Idx     = 2;
  constexpr size_t kConcatIdx     = 3;
  constexpr size_t kMulCosIdx     = 4;
  constexpr size_t kMulSinIdx     = 5;
  constexpr size_t kAddIdx        = 6;
  constexpr size_t kConvertOutIdx = 7;
  constexpr size_t kTransposeIdx  = 8;

  const auto& convert      = ops[start_index + kConvertIdx];
  const auto& slice1       = ops[start_index + kSlice1Idx];
  const auto& slice2       = ops[start_index + kSlice2Idx];
  const auto& concat       = ops[start_index + kConcatIdx];
  const auto& mul_cos      = ops[start_index + kMulCosIdx];
  const auto& mul_sin      = ops[start_index + kMulSinIdx];
  const auto& add          = ops[start_index + kAddIdx];
  const auto& convert_out  = ops[start_index + kConvertOutIdx];
  const auto& transpose    = ops[start_index + kTransposeIdx];

  // x is the Convert output: [B, S, H, D]
  const auto& x = convert.GetOutputTensor(0);
  const auto x_dims = x.GetDimensions();
  if (x_dims.size() != 4) return 1;
  const std::uint32_t B = x_dims[0], S = x_dims[1];
  const std::uint32_t H = x_dims[2], D = x_dims[3];
  const std::int32_t half = static_cast<std::int32_t>(D / 2);

  // Detect whether inputs are SFIXED8 (needs Convert→UFIXED8 before
  // RotaryEmbedding and back after) or already UFIXED8/UFIXED16 (no Cast needed).
  const bool needs_cast = x.IsQuantI8();  // SFIXED_POINT_8 = 0x0308

  // Both slices must consume x.
  if (slice1.GetInputTensor(0) != x || slice2.GetInputTensor(0) != x) {
    QNN_LOG_WARNING("[G2G] RoPE+T: slices don't consume Convert output.");
    return 1;
  }

  // Validate slice ranges: slice1=D[0:D/2], slice2=D[D/2:D].
  const auto r1 = ReadSliceRanges(slice1);
  const auto r2 = ReadSliceRanges(slice2);
  if (!r1 || !r2 || r1->size() < 12 || r2->size() < 12) {
    QNN_LOG_WARNING("[G2G] RoPE+T: couldn't read slice ranges.");
    return 1;
  }
  // ranges: [B0,Be,Bs, S0,Se,Ss, H0,He,Hs, D0,De,Ds] (indices 9,10 are D_start,D_end)
  if (!((*r1)[9] == 0 && (*r1)[10] == half) ||
      !((*r2)[9] == half && (*r2)[10] == static_cast<std::int32_t>(D))) {
    QNN_LOG_WARNING("[G2G] RoPE+T: unexpected slice ranges.");
    return 1;
  }

  // Concat must join [x2, x1] along last axis, doubling it.
  const auto& co_dims = concat.GetOutputTensor(0).GetDimensions();
  const auto& ci_dims = concat.GetInputTensor(0).GetDimensions();
  if (co_dims.empty() || ci_dims.empty() || co_dims.back() != 2 * ci_dims.back()) {
    QNN_LOG_WARNING("[G2G] RoPE+T: Concat shape mismatch.");
    return 1;
  }
  if (concat.GetInputTensor(0) != slice2.GetOutputTensor(0) ||
      concat.GetInputTensor(1) != slice1.GetOutputTensor(0)) {
    QNN_LOG_WARNING("[G2G] RoPE+T: Concat inputs are not [x2, x1].");
    return 1;
  }

  // Identify cos and sin.
  const bool cos_in0 = (mul_cos.GetInputTensor(0) == x);
  if (!cos_in0 && mul_cos.GetInputTensor(1) != x) {
    QNN_LOG_WARNING("[G2G] RoPE+T: MulCos doesn't use x.");
    return 1;
  }
  const auto& cos_tensor = cos_in0 ? mul_cos.GetInputTensor(1)
                                   : mul_cos.GetInputTensor(0);

  const auto& concat_out = concat.GetOutputTensor(0);
  const bool sin_in0 = (mul_sin.GetInputTensor(0) == concat_out);
  if (!sin_in0 && mul_sin.GetInputTensor(1) != concat_out) {
    QNN_LOG_WARNING("[G2G] RoPE+T: MulSin doesn't use concat output.");
    return 1;
  }
  const auto& sin_tensor = sin_in0 ? mul_sin.GetInputTensor(1)
                                   : mul_sin.GetInputTensor(0);

  // Add must consume both Mul outputs.
  const bool add_ok =
      ((add.GetInputTensor(0) == mul_cos.GetOutputTensor(0) &&
        add.GetInputTensor(1) == mul_sin.GetOutputTensor(0)) ||
       (add.GetInputTensor(0) == mul_sin.GetOutputTensor(0) &&
        add.GetInputTensor(1) == mul_cos.GetOutputTensor(0)));
  if (!add_ok) {
    QNN_LOG_WARNING("[G2G] RoPE+T: Add doesn't consume both Muls.");
    return 1;
  }

  // convert_out must consume Add output; Transpose must consume convert_out.
  if (convert_out.GetInputTensor(0) != add.GetOutputTensor(0) ||
      transpose.GetInputTensor(0) != convert_out.GetOutputTensor(0)) {
    QNN_LOG_WARNING("[G2G] RoPE+T: Add→Convert→Transpose chain broken.");
    return 1;
  }

  // Transpose must be perm=[0,2,1,3]: [B,S,H,D]→[B,H,S,D].
  if (!IsPerm0213(transpose)) {
    QNN_LOG_WARNING("[G2G] RoPE+T: Transpose perm is not [0,2,1,3].");
    return 1;
  }

  // cos/sin are [B, S, 1, D]; use a single StridedSlice with shrink_axes=4
  // (bit 2 squeezes the H=1 dim) while also slicing D, giving [B,S,D/2]
  // directly — no Reshape needed.
  const auto cos_dims = cos_tensor.GetDimensions();
  const auto sin_dims = sin_tensor.GetDimensions();
  if (cos_dims.size() != 4 || sin_dims.size() != 4) {
    QNN_LOG_WARNING("[G2G] RoPE+T: cos/sin must be rank 4.");
    return 1;
  }

  std::vector<OpWrapper> new_ops;
  const std::uint32_t seq_len = cos_dims[1];

  // --- Step 1: StridedSlice cos/sin [B,S,1,D] → [B,S,D/2] with shrink_axes=4.
  // ranges: [B:0..1, S:0..seq_len, H:0..1 (shrunk), D:0..half or half..D]
  // shrink_axes bit mask 0b0100 = 4 squeezes dim 2.
  const std::vector<std::int32_t> cos_ranges = {
      0, 1,    1,
      0, static_cast<std::int32_t>(seq_len), 1,
      0, 1,    1,   // dim 2 is squeezed by shrink_axes
      0, half, 1,
  };
  const std::vector<std::int32_t> sin_ranges = {
      0, 1,    1,
      0, static_cast<std::int32_t>(seq_len), 1,
      0, 1,    1,   // dim 2 is squeezed by shrink_axes
      half, static_cast<std::int32_t>(D), 1,
  };
  const std::vector<std::uint32_t> ranges_dims_4d = {4, 3};
  auto& cos_slice_ranges = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, ranges_dims_4d,
      sizeof(std::int32_t) * cos_ranges.size(), cos_ranges.data());
  auto& sin_slice_ranges = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, ranges_dims_4d,
      sizeof(std::int32_t) * sin_ranges.size(), sin_ranges.data());

  const std::vector<std::uint32_t> cs_half = {B, seq_len, static_cast<std::uint32_t>(half)};
  auto& cos_half_t = tensor_pool.CloneNativeTensorFrom(cos_tensor, cs_half);
  auto& sin_half_t = tensor_pool.CloneNativeTensorFrom(sin_tensor, cs_half);

  constexpr std::uint32_t kShrinkAxesBit2 = 4;  // 0b0100: squeeze dim 2
  auto make_shrink_slice = [&](const TensorWrapper& src,
                               const TensorWrapper& ranges,
                               const TensorWrapper& dst) {
    OpWrapper op(GetUniqueOpName(QNN_OP_STRIDED_SLICE), QNN_OP_STRIDED_SLICE,
                 QnnOpCode::kStridedSlice);
    op.AddInputTensor(src);
    op.AddOutputTensor(dst);
    op.AddTensorParam(QNN_OP_STRIDED_SLICE_PARAM_RANGES, ranges);
    op.AddScalarParam<std::uint32_t>(QNN_OP_STRIDED_SLICE_PARAM_SHRINK_AXES,
                                     kShrinkAxesBit2);
    return op;
  };
  new_ops.emplace_back(make_shrink_slice(cos_tensor, cos_slice_ranges, cos_half_t));
  new_ops.emplace_back(make_shrink_slice(sin_tensor, sin_slice_ranges, sin_half_t));

  // --- Step 1b: If SFIXED8, cast cos/sin and x to UFIXED8 for RotaryEmbedding. ---
  // After RotaryEmbedding, cast output back to SFIXED8 for the Concat.
  const TensorWrapper* cos_rope = &cos_half_t;
  const TensorWrapper* sin_rope = &sin_half_t;
  if (needs_cast) {
    cos_rope = &ToUFixed8(cos_half_t, new_ops, tensor_pool);
    sin_rope = &ToUFixed8(sin_half_t, new_ops, tensor_pool);
  }

  // --- Step 3: Split x [B,S,H,D] along axis=2 → H × [B,S,1,D]. ---
  // Split preserves rank so each output is [B,S,1,D].  No Reshape needed before
  // RotaryEmbedding: [B,S,1,D] and [B,1,S,D] have identical row-major layout
  // (element [b,s,0,d] and [b,0,s,d] share offset b*S*D + s*D + d), so we
  // just emit a Reshape to relabel the dims as [B,1,S,D] for QNN.
  const std::vector<std::uint32_t> bs1d = {B, S, 1, D};
  const std::vector<std::uint32_t> b1sd = {B, 1, S, D};

  std::vector<ConstTensorWrapperRef> split_outs;
  split_outs.reserve(H);
  for (std::uint32_t h = 0; h < H; ++h) {
    split_outs.emplace_back(tensor_pool.CloneNativeTensorFrom(x, bs1d));
  }
  // Build split_index tensor: cut points {1, 2, ..., H-1}.
  std::vector<std::uint32_t> split_indices;
  split_indices.reserve(H - 1);
  for (std::uint32_t h = 1; h < H; ++h) split_indices.emplace_back(h);
  auto& split_index_t = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, {}, {H - 1},
      sizeof(std::uint32_t) * split_indices.size(), split_indices.data());
  new_ops.emplace_back(CreateSplitOp(x, split_outs, 2, split_index_t));

  // --- Step 4: For each head, [optional SFIXED→UFIXED cast], Reshape
  //             [B,S,1,D]→[B,1,S,D], RotaryEmbedding, [optional UFIXED→SFIXED]. ---
  std::vector<ConstTensorWrapperRef> concat_inputs;
  concat_inputs.reserve(H);
  for (std::uint32_t h = 0; h < H; ++h) {
    // Reshape [B,S,1,D] → [B,1,S,D]  (same memory layout, relabels dims)
    auto& head_reshaped = tensor_pool.CloneNativeTensorFrom(x, b1sd);
    new_ops.emplace_back(CreateReshapeOp(split_outs[h].get(), head_reshaped));

    // If SFIXED8: cast the head input to UFIXED8 before RotaryEmbedding.
    const TensorWrapper* rope_in = &head_reshaped;
    if (needs_cast) {
      rope_in = &ToUFixed8(head_reshaped, new_ops, tensor_pool);
    }

    // RotaryEmbedding: inputs are UFIXED8 (or original dtype if no cast needed).
    auto& rope_out_u = tensor_pool.CloneNativeTensorFrom(*rope_in, b1sd);
    new_ops.emplace_back(
        CreateRotaryEmbeddingOp(*rope_in, *cos_rope, *sin_rope, rope_out_u, D));

    // If SFIXED8: cast RotaryEmbedding output back to SFIXED8 for Concat.
    if (needs_cast) {
      // Reuse the quant params of the original x (same scale/offset as SFIXED8 head).
      auto& rope_out_s = tensor_pool.CloneNativeTensorFrom(x, b1sd);
      new_ops.emplace_back(CreateConvertOp(rope_out_u, rope_out_s));
      concat_inputs.emplace_back(rope_out_s);
    } else {
      concat_inputs.emplace_back(rope_out_u);
    }
  }

  // --- Step 5: Concat(axis=1) all H × [B,1,S,D] → [B,H,S,D]. ---
  const std::vector<std::uint32_t> bhsd = {B, H, S, D};
  auto& rope_concat_out = tensor_pool.CloneNativeTensorFrom(x, bhsd);
  new_ops.emplace_back(CreateConcatenationOp(concat_inputs, rope_concat_out, 1));

  // --- Step 6: Single output Convert → transpose's output tensor. ---
  // (The pattern's convert_out carries the correct quant params for downstream ops.)
  new_ops.emplace_back(
      CreateOpWithSameParams(convert_out, {rope_concat_out},
                             {transpose.GetOutputTensor(0)}));

  // --- Validate all new ops including RotaryEmbedding. ---
  bool is_valid = true;
  for (size_t i = 0; i < new_ops.size(); ++i) {
    if (!validate_op_config(new_ops[i])) {
      const auto cfg = new_ops[i].GetOpConfig();
      QNN_LOG_WARNING(
          "[G2G] RoPE+T: new_ops[%zu] type=%s failed validation.",
          i, cfg.v1.typeName ? cfg.v1.typeName : "?");
      is_valid = false;
    }
  }
  if (!is_valid) {
    QNN_LOG_WARNING("[G2G] RoPE+T fusion validation failed, skipping.");
    return 1;
  }

  for (size_t i = 0; i < new_ops.size(); ++i) {
    new_ops[i].AddSuffixToName(absl::StrCat("_qcg2g_rope_", i));
  }

  ops.insert(ops.begin() + start_index + pattern_size,
             std::make_move_iterator(new_ops.begin()),
             std::make_move_iterator(new_ops.end()));
  // Erase the 8-op pattern entirely. The Convert at index 0 produces x which
  // is consumed by the new Unpack — but x is actually the Convert's OUTPUT,
  // which is an intermediate tensor. After erasing Convert, x has no producer.
  // Solution: keep the Convert (erase indices 1..7 only).
  ops.erase(ops.begin() + start_index + kSlice1Idx,
            ops.begin() + start_index + pattern_size);

  QNN_LOG_INFO("[G2G] RoPE+Transpose → Split + %u × [Reshape+RotaryEmbedding] + Concat + Convert fusion success.", H);
  return new_ops.size();
}

}  // namespace qnn
