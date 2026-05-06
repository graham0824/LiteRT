// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/compiler/qnn_transformations.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_builder.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_matchers.h"
#include "litert/cc/internal/litert_op_options.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/core/model/model.h"

namespace {

// Formats a Tensor's shape as "[d0, d1, ...]" for logging.
std::string FormatTensorShape(const litert::Tensor& t) {
  auto ranked_type = t.RankedTensorType();
  if (!ranked_type) return "<unranked>";
  auto dims = ranked_type->Layout().Dimensions();
  std::string s = "[";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i > 0) s += ", ";
    s += std::to_string(dims[i]);
  }
  s += "]";
  return s;
}

// Comprehensive diagnostic description of a tensor. Used by Step 0 only.
std::string DescribeTensor(const litert::Tensor& t) {
  if (t.Get() == nullptr) return "<null>";
  std::string s;
  auto type_id = t.TypeId();
  s += "tid=";
  s += (type_id == kLiteRtRankedTensorType ? "ranked"
        : type_id == kLiteRtUnrankedTensorType ? "unranked"
                                                : "other");
  s += " shape=" + FormatTensorShape(t);
  if (type_id == kLiteRtRankedTensorType) {
    auto rt = t.RankedTensorType();
    if (rt) {
      s += " etype=" + std::to_string(static_cast<int>(rt->ElementType()));
    }
  }
  auto qt = t.QTypeId();
  s += " qtype=";
  if (qt == kLiteRtQuantizationNone) {
    s += "none";
  } else if (qt == kLiteRtQuantizationPerTensor) {
    auto q = t.PerTensorQuantization();
    s += "per_tensor(scale=" + std::to_string(q.scale) +
         " zp=" + std::to_string(q.zero_point) + ")";
  } else if (qt == kLiteRtQuantizationPerChannel) {
    auto q = t.PerChannelQuantization();
    s += "per_channel(axis=" + std::to_string(q.quantized_dimension) +
         " n=" + std::to_string(q.num_channels);
    if (q.num_channels > 0 && q.scales != nullptr) {
      s += " scales[0..2]=";
      for (size_t i = 0; i < std::min<size_t>(q.num_channels, 3); ++i) {
        if (i > 0) s += ",";
        s += std::to_string(q.scales[i]);
      }
    }
    s += ")";
  } else {
    s += "?(" + std::to_string(static_cast<int>(qt)) + ")";
  }
  s += " const=" + std::string(t.IsConstant() ? "Y" : "N");
  s += " weights=" + std::string(t.HasWeights() ? "Y" : "N");
  s += " users=" + std::to_string(t.Uses().size());
  return s;
}

// Logs the int32 contents of a 1-D constant tensor (for Slice begin/size).
std::string DescribeInt32Const(const litert::Tensor& t) {
  if (t.Get() == nullptr) return "<null>";
  std::string s = "[";
  auto data_res = t.WeightsData<int32_t>();
  if (data_res) {
    auto data = *data_res;
    for (size_t i = 0; i < data.size(); ++i) {
      if (i > 0) s += ", ";
      s += std::to_string(data[i]);
    }
  } else {
    s += "<no-data>";
  }
  s += "]";
  return s;
}

// Formats the output[0] shape of `op` as "[d0, d1, ...]" for logging.
std::string FormatOutputShape(const litert::Op& op) {
  if (op.Outputs().empty()) return "<no-output>";
  return FormatTensorShape(op.Outputs()[0]);
}

// Returns the tensor's dim at `axis`, or -1 if the tensor is not ranked or
// the axis is out of bounds.
int32_t DimAt(const litert::Tensor& t, size_t axis) {
  auto ranked_type = t.RankedTensorType();
  if (!ranked_type) return -1;
  auto dims = ranked_type->Layout().Dimensions();
  if (axis >= dims.size()) return -1;
  return dims[axis];
}

// Returns the ranked dims of `t` as a vector, or empty if unranked.
std::vector<int32_t> DimsOf(const litert::Tensor& t) {
  auto ranked_type = t.RankedTensorType();
  if (!ranked_type) return {};
  auto dims = ranked_type->Layout().Dimensions();
  return std::vector<int32_t>(dims.begin(), dims.end());
}

// Builds a constant int32 1D tensor with the given values.
litert::Tensor BuildInt32Const1D(litert::Builder& builder,
                                 absl::Span<const int32_t> values) {
  std::vector<int32_t> shape{static_cast<int32_t>(values.size())};
  litert::RankedTensorType type(
      litert::GetElementType<int32_t>(),
      litert::Layout(
          litert::BuildLayout(shape.data(), shape.data() + shape.size())));
  auto spec = litert::RankedTensorSpecBuilder(type).Build();
  auto tensor = builder.BuildTensor(spec);
  if (!tensor) return litert::Tensor(nullptr);
  auto w = builder.BuildWeights<int32_t>(values, *tensor);
  (void)w;
  return *tensor;
}

// Builds a new tensor that shares `src`'s element type and quantization
// with a new shape.
//
// Quantization cloning rules:
//  - none              : no q-params on new tensor.
//  - per-tensor        : copy scale/zero_point verbatim.
//  - per-channel       : copy scales/zero_points verbatim by default. If the
//                        caller passes `slice_index`, interpret it as "src is
//                        being split along its quantized_dimension into N
//                        slices and this tensor is slice #i" — copy just the
//                        i-th scale/zero_point, adjust `num_channels` to 1.
// If `slice_index` is set but `src` isn't per-channel-quantized, the index is
// ignored.
litert::Tensor CloneTensorWithShape(
    litert::Builder& builder, const litert::Tensor& src,
    absl::Span<const int32_t> new_dims,
    std::optional<int32_t> slice_index = std::nullopt) {
  auto src_ranked = src.RankedTensorType();
  if (!src_ranked) return litert::Tensor(nullptr);
  litert::RankedTensorType new_type(
      src_ranked->ElementType(),
      litert::Layout(litert::BuildLayout(new_dims.data(),
                                         new_dims.data() + new_dims.size())));
  litert::RankedTensorSpecBuilder spec_builder(new_type);

  const auto qtype = src.QTypeId();
  if (qtype == kLiteRtQuantizationPerTensor) {
    spec_builder = std::move(spec_builder)
                       .WithPerTensorQuantization(src.PerTensorQuantization());
  } else if (qtype == kLiteRtQuantizationPerChannel) {
    auto src_pc = src.PerChannelQuantization();
    if (slice_index.has_value() && src_pc.scales != nullptr &&
        static_cast<uint64_t>(*slice_index) < src_pc.num_channels) {
      // Select a single channel.
      LiteRtQuantizationPerChannel single = src_pc;
      single.num_channels = 1;
      // Note: we point into the caller's scales/zero_points array at offset
      // slice_index. This pointer must outlive the tensor, which it does for
      // the duration of this transformation because src_pc.scales/zero_points
      // are owned by the matched subgraph (not by us).
      single.scales = src_pc.scales + *slice_index;
      if (src_pc.zero_points != nullptr) {
        single.zero_points = src_pc.zero_points + *slice_index;
      }
      spec_builder =
          std::move(spec_builder).WithPerChannelQuantization(single);
    } else {
      spec_builder =
          std::move(spec_builder).WithPerChannelQuantization(src_pc);
    }
  }
  auto tensor = builder.BuildTensor(std::move(spec_builder).Build());
  if (!tensor) return litert::Tensor(nullptr);
  return *tensor;
}

// Clone op options from `src_op` onto `dst_op`. Supports the op types used
// in the SHA rewrite. Returns true on success (or if no relevant options).
bool CloneOpOptions(litert::Builder& builder, const litert::Op& src_op,
                    litert::Op& dst_op) {
  switch (src_op.Code()) {
    case kLiteRtOpCodeTflMul: {
      litert::MulOptions opts;
      if (opts.InitFromOp(src_op.Get()) != kLiteRtStatusOk) return false;
      return builder.SetOpOptions(dst_op, std::move(opts)).HasValue();
    }
    case kLiteRtOpCodeTflBatchMatmul: {
      litert::BatchMatmulOptions opts;
      if (opts.InitFromOp(src_op.Get()) != kLiteRtStatusOk) return false;
      return builder.SetOpOptions(dst_op, std::move(opts)).HasValue();
    }
    case kLiteRtOpCodeTflConcatenation: {
      litert::ConcatenationOptions opts;
      if (opts.InitFromOp(src_op.Get()) != kLiteRtStatusOk) return false;
      return builder.SetOpOptions(dst_op, std::move(opts)).HasValue();
    }
    case kLiteRtOpCodeTflAdd: {
      litert::AddOptions opts;
      if (opts.InitFromOp(src_op.Get()) != kLiteRtStatusOk) return false;
      return builder.SetOpOptions(dst_op, std::move(opts)).HasValue();
    }
    case kLiteRtOpCodeTflSoftmax: {
      litert::SoftmaxOptions opts;
      if (opts.InitFromOp(src_op.Get()) != kLiteRtStatusOk) return false;
      return builder.SetOpOptions(dst_op, std::move(opts)).HasValue();
    }
    default:
      return true;  // Slice / Reshape have no options to clone.
  }
}

// Build a per-head single-head-attention subgraph. Rank is preserved
// throughout (axis-1 = 1 on every intermediate tensor). Returns the final
// per-head QKV-add output tensor. On failure returns a null Tensor.
litert::Tensor BuildSingleSHA(
    litert::Builder& builder, int32_t num_attn_per_kv_heads,
    const litert::Tensor& scale_mul_slice, const litert::Tensor& k_cache_slice,
    const litert::Tensor& k_slice_slice, const litert::Tensor& v_cache_slice,
    const litert::Tensor& v_slice_slice, const litert::Op& mul_op,
    const litert::Op& mm1_op, const litert::Op& mm2_op,
    const litert::Op& concat_op, const litert::Op& bias_add_op,
    const litert::Op& softmax_op, const litert::Op& slice_left_op,
    const litert::Op& slice_right_op, const litert::Op& mm3_op,
    const litert::Op& mm4_op, const litert::Op& final_add_op) {
  // Three shape-transform modes observed in the matched KV-swapped pattern:
  //
  //  - `attn`  : original axis-1 = num_attn_heads. Per-head: axis-1 -> 1,
  //              axis-2 unchanged (already per-head seq_q). Used for Mul.
  //  - `kv`    : original axis-1 = num_kv_heads, axis-2 = packed seq
  //              (= num_attn_per_kv_heads * seq_q). Per-head: axis-1 -> 1,
  //              axis-2 /= num_attn_per_kv_heads. Used for the MM*/Concat/
  //              Softmax/Slice/final-Add chain.
  //  - `packed`: original axis-0 = num_kv_heads, axis-1 = num_attn_per_kv.
  //              Per-head: axis-0 -> 1, axis-1 -> 1, axis-2/3 unchanged.
  //              Used for the post-R2 bias-Add output.
  auto dims_attn = [&](const litert::Op& src) {
    auto dims = DimsOf(src.Outputs()[0]);
    if (dims.size() >= 2) dims[1] = 1;
    return dims;
  };
  auto dims_kv = [&](const litert::Op& src) {
    auto dims = DimsOf(src.Outputs()[0]);
    if (dims.size() >= 2) dims[1] = 1;
    if (dims.size() >= 3 && num_attn_per_kv_heads > 0) {
      dims[2] /= num_attn_per_kv_heads;
    }
    return dims;
  };
  auto dims_packed = [&](const litert::Op& src) {
    auto dims = DimsOf(src.Outputs()[0]);
    if (dims.size() >= 1) dims[0] = 1;
    if (dims.size() >= 2) dims[1] = 1;
    return dims;
  };

  litert::Tensor null_tensor(nullptr);

  // 1. Mul (Q-scale). Per-head shape: [1, 1, seq_q, head_dim].
  auto mul_out_dims = dims_attn(mul_op);
  auto mul_out = CloneTensorWithShape(builder, mul_op.Outputs()[0],
                                      absl::MakeConstSpan(mul_out_dims));
  if (mul_out.Get() == nullptr) return null_tensor;
  auto mul_scale = mul_op.Inputs()[1];
  std::vector<litert::Tensor> mul_inputs{
      litert::Tensor(scale_mul_slice.Get()), litert::Tensor(mul_scale.Get())};
  std::vector<litert::Tensor> mul_outputs{mul_out};
  auto new_mul = builder.BuildOp(kLiteRtOpCodeTflMul, mul_inputs, mul_outputs);
  CloneOpOptions(builder, mul_op, new_mul);

  // 2. MM1 (Q * K_cache).
  auto mm1_out_dims = dims_kv(mm1_op);
  auto mm1_out = CloneTensorWithShape(builder, mm1_op.Outputs()[0],
                                      absl::MakeConstSpan(mm1_out_dims));
  if (mm1_out.Get() == nullptr) return null_tensor;
  std::vector<litert::Tensor> mm1_inputs{mul_out,
                                         litert::Tensor(k_cache_slice.Get())};
  std::vector<litert::Tensor> mm1_outputs{mm1_out};
  auto new_mm1 = builder.BuildOp(kLiteRtOpCodeTflBatchMatmul, mm1_inputs,
                                 mm1_outputs);
  CloneOpOptions(builder, mm1_op, new_mm1);

  // 3. MM2 (Q * K_slice).
  auto mm2_out_dims = dims_kv(mm2_op);
  auto mm2_out = CloneTensorWithShape(builder, mm2_op.Outputs()[0],
                                      absl::MakeConstSpan(mm2_out_dims));
  if (mm2_out.Get() == nullptr) return null_tensor;
  std::vector<litert::Tensor> mm2_inputs{mul_out,
                                         litert::Tensor(k_slice_slice.Get())};
  std::vector<litert::Tensor> mm2_outputs{mm2_out};
  auto new_mm2 = builder.BuildOp(kLiteRtOpCodeTflBatchMatmul, mm2_inputs,
                                 mm2_outputs);
  CloneOpOptions(builder, mm2_op, new_mm2);

  // 4. Concat (axis inherited from original).
  auto concat_out_dims = dims_kv(concat_op);
  auto concat_out = CloneTensorWithShape(builder, concat_op.Outputs()[0],
                                         absl::MakeConstSpan(concat_out_dims));
  if (concat_out.Get() == nullptr) return null_tensor;
  std::vector<litert::Tensor> concat_inputs{mm1_out, mm2_out};
  std::vector<litert::Tensor> concat_outputs{concat_out};
  auto new_concat = builder.BuildOp(kLiteRtOpCodeTflConcatenation,
                                    concat_inputs, concat_outputs);
  CloneOpOptions(builder, concat_op, new_concat);

  // 5. Bias/Mask Add (post-R2 layout: [num_kv, num_attn_per_kv, seq, K_seq]).
  // Per-head axis-0 and axis-1 both collapse to 1; axis-2 is already per-head.
  auto bias_add_out_dims = dims_packed(bias_add_op);
  auto bias_add_out = CloneTensorWithShape(
      builder, bias_add_op.Outputs()[0],
      absl::MakeConstSpan(bias_add_out_dims));
  if (bias_add_out.Get() == nullptr) return null_tensor;
  auto bias_tensor = bias_add_op.Inputs()[1];
  std::vector<litert::Tensor> bias_add_inputs{concat_out,
                                              litert::Tensor(bias_tensor.Get())};
  std::vector<litert::Tensor> bias_add_outputs{bias_add_out};
  auto new_bias_add = builder.BuildOp(kLiteRtOpCodeTflAdd, bias_add_inputs,
                                      bias_add_outputs);
  CloneOpOptions(builder, bias_add_op, new_bias_add);

  // 6. Softmax (skipping R3 since per-head dims match the bias-add output).
  auto softmax_out = CloneTensorWithShape(
      builder, softmax_op.Outputs()[0],
      absl::MakeConstSpan(bias_add_out_dims));
  if (softmax_out.Get() == nullptr) return null_tensor;
  std::vector<litert::Tensor> softmax_inputs{bias_add_out};
  std::vector<litert::Tensor> softmax_outputs{softmax_out};
  auto new_softmax = builder.BuildOp(kLiteRtOpCodeTflSoftmax, softmax_inputs,
                                     softmax_outputs);
  CloneOpOptions(builder, softmax_op, new_softmax);

  // Helper to build a per-head Slice that replicates `src_slice_op` on
  // `softmax_out`, with output shape `out_dims`. The original slices in the
  // KV-swapped pattern operate along axis 3 (K-seq), so `begin[3]` is kept
  // verbatim from the original slice. `begin[1]` is forced to 0 (axis 1 is
  // size 1 after per-head split). `size` comes from `out_dims` directly.
  auto build_per_head_slice =
      [&](const litert::Op& src_slice_op,
          const std::vector<int32_t>& out_dims) -> litert::Tensor {
    std::vector<int32_t> begin(out_dims.size(), 0);
    if (src_slice_op.Inputs().size() >= 2) {
      auto orig_begin = src_slice_op.Inputs()[1];
      auto orig_data = orig_begin.WeightsData<int32_t>();
      if (orig_data && orig_data->size() == out_dims.size()) {
        for (size_t i = 0; i < out_dims.size(); ++i) begin[i] = (*orig_data)[i];
        if (begin.size() >= 2) begin[1] = 0;
      }
    }
    auto begin_tensor = BuildInt32Const1D(builder, absl::MakeConstSpan(begin));
    auto size_tensor = BuildInt32Const1D(builder, absl::MakeConstSpan(out_dims));
    if (begin_tensor.Get() == nullptr || size_tensor.Get() == nullptr) {
      return litert::Tensor(nullptr);
    }
    auto slice_out = CloneTensorWithShape(
        builder, src_slice_op.Outputs()[0], absl::MakeConstSpan(out_dims));
    if (slice_out.Get() == nullptr) return litert::Tensor(nullptr);
    std::vector<litert::Tensor> slice_inputs{softmax_out, begin_tensor,
                                             size_tensor};
    std::vector<litert::Tensor> slice_outputs{slice_out};
    builder.BuildOp(kLiteRtOpCodeTflSlice, slice_inputs, slice_outputs);
    LITERT_LOG(LITERT_INFO, "  per-head slice: begin=[%d,%d,%d,%d] size=%s",
               begin.size() >= 1 ? begin[0] : -1,
               begin.size() >= 2 ? begin[1] : -1,
               begin.size() >= 3 ? begin[2] : -1,
               begin.size() >= 4 ? begin[3] : -1,
               FormatTensorShape(slice_out).c_str());
    return slice_out;
  };

  // 7. Slice (left, V-cache portion).
  auto sl_out = build_per_head_slice(slice_left_op, dims_kv(slice_left_op));
  if (sl_out.Get() == nullptr) return null_tensor;

  // 8. Slice (right, V-slice portion).
  auto sr_out = build_per_head_slice(slice_right_op, dims_kv(slice_right_op));
  if (sr_out.Get() == nullptr) return null_tensor;

  // 9. MM3 (attn * V_cache).
  auto mm3_out_dims = dims_kv(mm3_op);
  auto mm3_out = CloneTensorWithShape(builder, mm3_op.Outputs()[0],
                                      absl::MakeConstSpan(mm3_out_dims));
  if (mm3_out.Get() == nullptr) return null_tensor;
  std::vector<litert::Tensor> mm3_inputs{sl_out,
                                         litert::Tensor(v_cache_slice.Get())};
  std::vector<litert::Tensor> mm3_outputs{mm3_out};
  auto new_mm3 = builder.BuildOp(kLiteRtOpCodeTflBatchMatmul, mm3_inputs,
                                 mm3_outputs);
  CloneOpOptions(builder, mm3_op, new_mm3);

  // 10. MM4 (attn * V_slice).
  auto mm4_out_dims = dims_kv(mm4_op);
  auto mm4_out = CloneTensorWithShape(builder, mm4_op.Outputs()[0],
                                      absl::MakeConstSpan(mm4_out_dims));
  if (mm4_out.Get() == nullptr) return null_tensor;
  std::vector<litert::Tensor> mm4_inputs{sr_out,
                                         litert::Tensor(v_slice_slice.Get())};
  std::vector<litert::Tensor> mm4_outputs{mm4_out};
  auto new_mm4 = builder.BuildOp(kLiteRtOpCodeTflBatchMatmul, mm4_inputs,
                                 mm4_outputs);
  CloneOpOptions(builder, mm4_op, new_mm4);

  // 11. Final Add (per-head QKV).
  auto final_out_dims = dims_kv(final_add_op);
  auto final_out = CloneTensorWithShape(builder, final_add_op.Outputs()[0],
                                        absl::MakeConstSpan(final_out_dims));
  if (final_out.Get() == nullptr) return null_tensor;
  std::vector<litert::Tensor> final_inputs{mm3_out, mm4_out};
  std::vector<litert::Tensor> final_outputs{final_out};
  auto new_final = builder.BuildOp(kLiteRtOpCodeTflAdd, final_inputs,
                                   final_outputs);
  CloneOpOptions(builder, final_add_op, new_final);

  return final_out;
}

// Extract `n` per-head slices from `src` along axis 1 using one Slice op per
// head (instead of a single Split). Each slice keeps the source rank with
// axis 1 reduced to size 1. Workaround for QNN HTP validateOpConfig crash
// on Split with large num_splits.
std::vector<litert::Tensor> SliceAxis1(litert::Builder& builder,
                                       const litert::Tensor& src, int32_t n) {
  std::vector<litert::Tensor> outs;
  auto src_dims = DimsOf(src);
  if (src_dims.size() < 2 || src_dims[1] != n) return outs;

  std::vector<int32_t> slice_dims = src_dims;
  slice_dims[1] = 1;
  auto size_tensor =
      BuildInt32Const1D(builder, absl::MakeConstSpan(slice_dims));
  if (size_tensor.Get() == nullptr) return {};

  outs.reserve(n);
  for (int32_t i = 0; i < n; ++i) {
    std::vector<int32_t> begin(src_dims.size(), 0);
    begin[1] = i;
    auto begin_tensor =
        BuildInt32Const1D(builder, absl::MakeConstSpan(begin));
    if (begin_tensor.Get() == nullptr) return {};
    auto out = CloneTensorWithShape(builder, src,
                                    absl::MakeConstSpan(slice_dims));
    if (out.Get() == nullptr) return {};
    std::vector<litert::Tensor> inputs{litert::Tensor(src.Get()), begin_tensor,
                                       size_tensor};
    std::vector<litert::Tensor> outputs{out};
    builder.BuildOp(kLiteRtOpCodeTflSlice, inputs, outputs);
    outs.push_back(out);
  }
  return outs;
}

// Split `src` along axis 1 into `num_splits` slices. Each slice has the same
// rank as `src` with axis 1 reduced to size 1. Returns the per-slice
// tensors; empty on failure.
std::vector<litert::Tensor> SplitAxis1(litert::Builder& builder,
                                       const litert::Tensor& src,
                                       int32_t num_splits) {
  std::vector<litert::Tensor> split_outputs;
  auto src_dims = DimsOf(src);
  if (src_dims.size() < 2 || src_dims[1] != num_splits) return split_outputs;

  std::vector<int32_t> split_out_dims = src_dims;
  split_out_dims[1] = 1;  // each split output has axis 1 reduced to size 1

  litert::Tensor axis_tensor =
      BuildInt32Const1D(builder, absl::MakeConstSpan({1}));
  if (axis_tensor.Get() == nullptr) return {};

  // If the source is per-channel-quantized along axis 1 (the split axis),
  // each split slice should carry only the i-th channel's q-params.
  const bool src_pc_axis1 =
      src.QTypeId() == kLiteRtQuantizationPerChannel &&
      src.PerChannelQuantization().quantized_dimension == 1;

  split_outputs.reserve(num_splits);
  for (int32_t i = 0; i < num_splits; ++i) {
    auto t = CloneTensorWithShape(
        builder, src, absl::MakeConstSpan(split_out_dims),
        src_pc_axis1 ? std::optional<int32_t>(i) : std::nullopt);
    if (t.Get() == nullptr) return {};
    split_outputs.push_back(t);
  }

  std::vector<litert::Tensor> split_inputs{axis_tensor,
                                           litert::Tensor(src.Get())};
  litert::Op split_op =
      builder.BuildOp(kLiteRtOpCodeTflSplit, split_inputs, split_outputs);
  litert::SplitOptions split_opts;
  split_opts.num_splits = num_splits;
  auto opts_res = builder.SetOpOptions(split_op, std::move(split_opts));
  (void)opts_res;

  return split_outputs;
}

}  // namespace

extern "C" {

LiteRtStatus FuseMatMulRequantTransformation(LiteRtBuilder builder_ptr,
                                             LiteRtOp op) {
  litert::Builder builder(builder_ptr);
  litert::Op root_op(op);
  litert::Op matmul_op(nullptr);

  if (!litert::Match(
          root_op,
          litert::m_Op<kLiteRtOpCodeTflQuantize>(litert::m_CaptureOrSameAs(
              &matmul_op,
              litert::m_AllOf(
                  litert::m_HasOneUse(),
                  litert::m_OpCode<kLiteRtOpCodeTflBatchMatmul>()))))) {
    return kLiteRtStatusPatternNoMatch;
  }

  // Only fuse true requantizations (element type unchanged). A real
  // Quantize changes dtype and must be lowered separately.
  if (root_op.Inputs()[0].ElementType() != root_op.Outputs()[0].ElementType()) {
    return kLiteRtStatusPatternNoMatch;
  }

  litert::OpInputs inputs = matmul_op.Inputs();
  std::vector<litert::Tensor> inputs_vec(inputs.begin(), inputs.end());

  litert::Op new_matmul =
      builder.ReplaceOp(root_op, kLiteRtOpCodeTflBatchMatmul, inputs_vec);
  builder.EraseOp(matmul_op);

  // Carry over adj_x / adj_y / asymmetric_quantize_input from the original op.
  litert::BatchMatmulOptions options;
  LITERT_RETURN_IF_ERROR(options.InitFromOp(matmul_op.Get()));
  auto res = builder.SetOpOptions(new_matmul, std::move(options));
  if (!res) {
    return res.Error().Status();
  }

  LITERT_LOG(LITERT_INFO, "FuseMatMulRequant: fused BatchMatmul+Quantize.");
  return kLiteRtStatusOk;
}

LiteRtStatus MatchCompositeAttentionPattern(LiteRtOp op) {
  using litert::m_Any;
  using litert::m_CaptureOrSameAs;
  using litert::m_Op;
  using litert::m_OpCode;

  litert::Op root_op(op);
  litert::Op softmax_op(nullptr);
  litert::Tensor q_scaled(nullptr);

  auto match_q_scaled = m_CaptureOrSameAs(
      &q_scaled,
      m_Op<kLiteRtOpCodeTflReshape>(
          m_Op<kLiteRtOpCodeTflMul>(m_Any(), m_Any()), m_Any()));

  auto match_scores = m_Op<kLiteRtOpCodeTflConcatenation>(
      m_Op<kLiteRtOpCodeTflBatchMatmul>(match_q_scaled, m_Any()),
      m_Op<kLiteRtOpCodeTflBatchMatmul>(match_q_scaled, m_Any()));

  auto match_softmax = m_Op<kLiteRtOpCodeTflSoftmax>(
      m_Op<kLiteRtOpCodeTflReshape>(
          m_Op<kLiteRtOpCodeTflAdd>(
              m_Op<kLiteRtOpCodeTflReshape>(match_scores, m_Any()), m_Any()),
          m_Any()));

  auto match_left_arm = m_Op<kLiteRtOpCodeTflBatchMatmul>(
      m_Op<kLiteRtOpCodeTflSlice>(
          m_CaptureOrSameAs(&softmax_op, match_softmax), m_Any(), m_Any()),
      m_Any());

  auto match_right_arm = m_Op<kLiteRtOpCodeTflBatchMatmul>(
      m_Op<kLiteRtOpCodeTflSlice>(
          m_CaptureOrSameAs(&softmax_op, m_OpCode<kLiteRtOpCodeTflSoftmax>()),
          m_Any(), m_Any()),
      m_Any());

  auto match_root = m_Op<kLiteRtOpCodeTflReshape>(
      m_Op<kLiteRtOpCodeTflTranspose>(
          m_Op<kLiteRtOpCodeTflReshape>(
              m_Op<kLiteRtOpCodeTflAdd>(match_left_arm, match_right_arm),
              m_Any()),
          m_Any()),
      m_Any());

  return litert::Match(root_op, match_root) ? kLiteRtStatusOk
                                             : kLiteRtStatusPatternNoMatch;
}

LiteRtStatus ApplyCompositeAttentionTransformation(
    LiteRtBuilder builder_ptr, LiteRtOp op) {
  using litert::m_Any;
  using litert::m_CaptureOrSameAs;
  using litert::m_Op;
  using litert::m_OpCode;

  litert::Builder builder(builder_ptr);
  litert::Op root_op(op);

  // Shared-structure captures.
  litert::Op softmax_op(nullptr);
  litert::Tensor q_scaled(nullptr);

  // Reporting captures.
  litert::Op mul_op(nullptr);
  litert::Op r1_op(nullptr);
  litert::Op mm1_op(nullptr);
  litert::Op mm2_op(nullptr);
  litert::Op concat_op(nullptr);
  litert::Op r2_op(nullptr);
  litert::Op bias_add_op(nullptr);
  litert::Op r3_op(nullptr);
  litert::Op slice_left_op(nullptr);
  litert::Op slice_right_op(nullptr);
  litert::Op mm3_op(nullptr);
  litert::Op mm4_op(nullptr);
  litert::Op final_add_op(nullptr);
  litert::Op post_reshape_op(nullptr);
  litert::Op transpose_op(nullptr);

  auto match_q_scaled = m_CaptureOrSameAs(
      &q_scaled,
      m_CaptureOrSameAs(
          &r1_op,
          m_Op<kLiteRtOpCodeTflReshape>(
              m_CaptureOrSameAs(
                  &mul_op, m_Op<kLiteRtOpCodeTflMul>(m_Any(), m_Any())),
              m_Any())));

  auto match_scores = m_CaptureOrSameAs(
      &concat_op,
      m_Op<kLiteRtOpCodeTflConcatenation>(
          m_CaptureOrSameAs(
              &mm1_op,
              m_Op<kLiteRtOpCodeTflBatchMatmul>(match_q_scaled, m_Any())),
          m_CaptureOrSameAs(
              &mm2_op,
              m_Op<kLiteRtOpCodeTflBatchMatmul>(match_q_scaled, m_Any()))));

  auto match_softmax = m_Op<kLiteRtOpCodeTflSoftmax>(
      m_CaptureOrSameAs(
          &r3_op,
          m_Op<kLiteRtOpCodeTflReshape>(
              m_CaptureOrSameAs(
                  &bias_add_op,
                  m_Op<kLiteRtOpCodeTflAdd>(
                      m_CaptureOrSameAs(
                          &r2_op,
                          m_Op<kLiteRtOpCodeTflReshape>(match_scores, m_Any())),
                      m_Any())),
              m_Any())));

  auto match_left_arm = m_CaptureOrSameAs(
      &mm3_op,
      m_Op<kLiteRtOpCodeTflBatchMatmul>(
          m_CaptureOrSameAs(
              &slice_left_op,
              m_Op<kLiteRtOpCodeTflSlice>(
                  m_CaptureOrSameAs(&softmax_op, match_softmax), m_Any(),
                  m_Any())),
          m_Any()));

  auto match_right_arm = m_CaptureOrSameAs(
      &mm4_op,
      m_Op<kLiteRtOpCodeTflBatchMatmul>(
          m_CaptureOrSameAs(
              &slice_right_op,
              m_Op<kLiteRtOpCodeTflSlice>(
                  m_CaptureOrSameAs(&softmax_op,
                                    m_OpCode<kLiteRtOpCodeTflSoftmax>()),
                  m_Any(), m_Any())),
          m_Any()));

  auto match_qkv_add = m_CaptureOrSameAs(
      &final_add_op,
      m_Op<kLiteRtOpCodeTflAdd>(match_left_arm, match_right_arm));
  auto match_post_reshape = m_CaptureOrSameAs(
      &post_reshape_op,
      m_Op<kLiteRtOpCodeTflReshape>(match_qkv_add, m_Any()));
  auto match_transpose = m_CaptureOrSameAs(
      &transpose_op,
      m_Op<kLiteRtOpCodeTflTranspose>(match_post_reshape, m_Any()));
  auto match_root =
      m_Op<kLiteRtOpCodeTflReshape>(match_transpose, m_Any());

  if (!litert::Match(root_op, match_root)) {
    return kLiteRtStatusPatternNoMatch;
  }

  LITERT_LOG(LITERT_INFO, "MatchCompositeAttentionPattern captured ops:");
  LITERT_LOG(LITERT_INFO, "  Mul   (Q-scale)     out: %s",
             FormatOutputShape(mul_op).c_str());
  LITERT_LOG(LITERT_INFO, "  MM1   (Q * K_cache) out: %s",
             FormatOutputShape(mm1_op).c_str());
  LITERT_LOG(LITERT_INFO, "  MM2   (Q * K_slice) out: %s",
             FormatOutputShape(mm2_op).c_str());
  LITERT_LOG(LITERT_INFO, "  Add   (mask/bias)   out: %s",
             FormatOutputShape(bias_add_op).c_str());
  LITERT_LOG(LITERT_INFO, "  Slice (left)        out: %s",
             FormatOutputShape(slice_left_op).c_str());
  LITERT_LOG(LITERT_INFO, "  Slice (right)       out: %s",
             FormatOutputShape(slice_right_op).c_str());
  LITERT_LOG(LITERT_INFO, "  MM3   (attn * V_c)  out: %s",
             FormatOutputShape(mm3_op).c_str());
  LITERT_LOG(LITERT_INFO, "  MM4   (attn * V_s)  out: %s",
             FormatOutputShape(mm4_op).c_str());
  LITERT_LOG(LITERT_INFO, "  Add   (final QKV)   out: %s",
             FormatOutputShape(final_add_op).c_str());
  LITERT_LOG(LITERT_INFO, "  Reshape (post-QKV)  out: %s",
             FormatOutputShape(post_reshape_op).c_str());
  LITERT_LOG(LITERT_INFO, "  Transpose           out: %s",
             FormatOutputShape(transpose_op).c_str());
  LITERT_LOG(LITERT_INFO, "  Reshape (root)      out: %s",
             FormatOutputShape(root_op).c_str());

  if (mul_op.Inputs().empty() || mm1_op.Inputs().size() < 2 ||
      mm2_op.Inputs().size() < 2 || mm3_op.Inputs().size() < 2 ||
      mm4_op.Inputs().size() < 2) {
    LITERT_LOG(LITERT_WARNING,
               "Captured ops missing expected inputs; skipping head-count.");
    return kLiteRtStatusPatternNoMatch;
  }

  litert::Tensor scale_mul_in = mul_op.Inputs()[0];
  litert::Tensor k_cache = mm1_op.Inputs()[1];
  litert::Tensor k_slice = mm2_op.Inputs()[1];
  litert::Tensor v_cache = mm3_op.Inputs()[1];
  litert::Tensor v_slice = mm4_op.Inputs()[1];

  const int32_t num_attn_heads = DimAt(scale_mul_in, 1);
  const int32_t num_kv_heads = DimAt(k_cache, 1);

  if (num_attn_heads <= 0 || num_kv_heads <= 0 ||
      num_attn_heads % num_kv_heads != 0) {
    LITERT_LOG(LITERT_WARNING, "Bad head dims (attn=%d, kv=%d); skipping.",
               num_attn_heads, num_kv_heads);
    return kLiteRtStatusPatternNoMatch;
  }
  if (DimAt(k_slice, 1) != num_kv_heads || DimAt(v_cache, 1) != num_kv_heads ||
      DimAt(v_slice, 1) != num_kv_heads) {
    LITERT_LOG(LITERT_WARNING,
               "num_kv_heads=%d mismatch: [k_cache: %d, k_slice: %d, v_cache: "
               "%d, v_slice: %d].",
               num_kv_heads, DimAt(k_cache, 1), DimAt(k_slice, 1),
               DimAt(v_cache, 1), DimAt(v_slice, 1));
    return kLiteRtStatusPatternNoMatch;
  }
  const int32_t num_attn_per_kv_heads = num_attn_heads / num_kv_heads;

  LITERT_LOG(LITERT_INFO,
             "Head counts: num_attn_heads=%d, num_kv_heads=%d, "
             "num_attn_per_kv_heads=%d",
             num_attn_heads, num_kv_heads, num_attn_per_kv_heads);

  // --- Step 0 diagnostic dump (see PHASE_2C_PLAN.md). Remove in Step 6. ---
  LITERT_LOG(LITERT_INFO, "DIAG bias        : %s",
             DescribeTensor(bias_add_op.Inputs()[1]).c_str());
  LITERT_LOG(LITERT_INFO, "DIAG scale_mul_in: %s",
             DescribeTensor(scale_mul_in).c_str());
  LITERT_LOG(LITERT_INFO, "DIAG k_cache     : %s",
             DescribeTensor(k_cache).c_str());
  LITERT_LOG(LITERT_INFO, "DIAG k_slice     : %s",
             DescribeTensor(k_slice).c_str());
  LITERT_LOG(LITERT_INFO, "DIAG v_cache     : %s",
             DescribeTensor(v_cache).c_str());
  LITERT_LOG(LITERT_INFO, "DIAG v_slice     : %s",
             DescribeTensor(v_slice).c_str());
  if (slice_left_op.Inputs().size() >= 3) {
    LITERT_LOG(LITERT_INFO, "DIAG sl_begin    : %s  (%s)",
               DescribeTensor(slice_left_op.Inputs()[1]).c_str(),
               DescribeInt32Const(slice_left_op.Inputs()[1]).c_str());
    LITERT_LOG(LITERT_INFO, "DIAG sl_size     : %s  (%s)",
               DescribeTensor(slice_left_op.Inputs()[2]).c_str(),
               DescribeInt32Const(slice_left_op.Inputs()[2]).c_str());
  }
  if (slice_right_op.Inputs().size() >= 3) {
    LITERT_LOG(LITERT_INFO, "DIAG sr_begin    : %s  (%s)",
               DescribeTensor(slice_right_op.Inputs()[1]).c_str(),
               DescribeInt32Const(slice_right_op.Inputs()[1]).c_str());
    LITERT_LOG(LITERT_INFO, "DIAG sr_size     : %s  (%s)",
               DescribeTensor(slice_right_op.Inputs()[2]).c_str(),
               DescribeInt32Const(slice_right_op.Inputs()[2]).c_str());
  }
  auto describe_out = [](const char* label, const litert::Op& op) {
    LITERT_LOG(LITERT_INFO, "DIAG out %-10s: %s", label,
               op.Outputs().empty()
                   ? "<no-output>"
                   : DescribeTensor(op.Outputs()[0]).c_str());
  };
  describe_out("mul", mul_op);
  describe_out("mm1", mm1_op);
  describe_out("mm2", mm2_op);
  describe_out("concat", concat_op);
  describe_out("bias_add", bias_add_op);
  describe_out("softmax", softmax_op);
  describe_out("slice_L", slice_left_op);
  describe_out("slice_R", slice_right_op);
  describe_out("mm3", mm3_op);
  describe_out("mm4", mm4_op);
  describe_out("final_add", final_add_op);
  // --- end Step 0 diagnostic dump ---
  LITERT_LOG(LITERT_INFO, "Unpack plan (axis=1):");
  LITERT_LOG(LITERT_INFO, "  scale_mul_input %s -> %d slices",
             FormatTensorShape(scale_mul_in).c_str(), num_attn_heads);
  LITERT_LOG(LITERT_INFO, "  K_cache         %s -> %d slices",
             FormatTensorShape(k_cache).c_str(), num_kv_heads);
  LITERT_LOG(LITERT_INFO, "  K_slice         %s -> %d slices",
             FormatTensorShape(k_slice).c_str(), num_kv_heads);
  LITERT_LOG(LITERT_INFO, "  V_cache         %s -> %d slices",
             FormatTensorShape(v_cache).c_str(), num_kv_heads);
  LITERT_LOG(LITERT_INFO, "  V_slice         %s -> %d slices",
             FormatTensorShape(v_slice).c_str(), num_kv_heads);

  // =========================================================================
  // Debug flags — toggle to reproduce/isolate the Phase 2c crash.
  // See PHASE_2C_PLAN.md § "Next session" for the full investigation log.
  //
  // Default (safe baseline): kCommitPhase2c=false.
  //   Build all rewrite ops in-builder (dry-run), return PatternNoMatch,
  //   live subgraph unchanged. Baseline compile: 8 tests pass, 231904-byte
  //   flatbuffer, no crash.
  //
  //   kMinRewriteSmokeTest [TESTED: PASSES]
  //     Replaces root R5 with an identical Reshape (same inputs). No SHA,
  //     no Splits. Proves ReplaceOp/EraseOp mechanics are fine.
  //
  //   kCommitPhase2c [TESTED: CRASHES QNN validateOpConfig]
  //     Full rewrite: 5 Splits + 16 SHA heads + final Concat + ReplaceOp(R5)
  //     + EraseOp(16 original ops). Crashes at partition position 26 inside
  //     QnnBackend_validateOpConfig. Position is independent of whether
  //     scale_mul uses Split(16) or 16×Slice — the 26th op being validated
  //     crashes regardless of its type.
  //
  //   kSkipEraseOriginals [UNTESTED — next to try]
  //     Like kCommitPhase2c but keeps all 16 original pattern ops alive.
  //     Adds SHA as dead code on top of the original graph. If this passes
  //     QNN, the crash is caused by the erase-and-replace sequence, not the
  //     SHA ops themselves.
  //
  //   kScaleMulUseSlice [TESTED: same crash position as Split(16)]
  //     Replace Split(16) for scale_mul with 16 per-head Slice ops.
  //     Did not change the crash position — not a Split-specific issue.
  // =========================================================================
  constexpr bool kMinRewriteSmokeTest  = false;
  constexpr bool kCommitPhase2c        = false;  // ← flip to true to reproduce crash  // ← flip to true to reproduce crash
  constexpr bool kSkipEraseOriginals   = false;  // ← flip to true (with kCommitPhase2c=true) to test
  constexpr bool kScaleMulUseSlice     = false;  // ← use Slice-per-head instead of Split for Q

  // Min-rewrite smoke test: identity replacement, no other ops touched.
  if (kMinRewriteSmokeTest) {
    litert::OpInputs root_inputs = root_op.Inputs();
    std::vector<litert::Tensor> root_inputs_vec(root_inputs.begin(),
                                                root_inputs.end());
    builder.ReplaceOp(root_op, kLiteRtOpCodeTflReshape, root_inputs_vec);
    LITERT_LOG(LITERT_INFO, "Min-rewrite smoke test: identical R5 replacement.");
    return kLiteRtStatusOk;
  }

  // Phase 2a: unpack the 5 head-source tensors along axis 1.
  // scale_mul_in: num_attn_heads slices; K/V: num_kv_heads slices each.
  auto scale_mul_slices = kScaleMulUseSlice
                              ? SliceAxis1(builder, scale_mul_in, num_attn_heads)
                              : SplitAxis1(builder, scale_mul_in, num_attn_heads);
  auto k_cache_slices = SplitAxis1(builder, k_cache, num_kv_heads);
  auto k_slice_slices = SplitAxis1(builder, k_slice, num_kv_heads);
  auto v_cache_slices = SplitAxis1(builder, v_cache, num_kv_heads);
  auto v_slice_slices = SplitAxis1(builder, v_slice, num_kv_heads);

  if (scale_mul_slices.empty() || k_cache_slices.empty() ||
      k_slice_slices.empty() || v_cache_slices.empty() ||
      v_slice_slices.empty()) {
    LITERT_LOG(LITERT_WARNING, "Unpack failed for at least one tensor.");
    return kLiteRtStatusPatternNoMatch;
  }
  LITERT_LOG(LITERT_INFO, "Split per-slice shapes (rank preserved):");
  LITERT_LOG(LITERT_INFO, "  scale_mul_slices[0] %s (x%d)",
             FormatTensorShape(scale_mul_slices[0]).c_str(),
             static_cast<int>(scale_mul_slices.size()));
  LITERT_LOG(LITERT_INFO, "  k_cache_slices[0]   %s (x%d)",
             FormatTensorShape(k_cache_slices[0]).c_str(),
             static_cast<int>(k_cache_slices.size()));
  LITERT_LOG(LITERT_INFO, "  k_slice_slices[0]   %s (x%d)",
             FormatTensorShape(k_slice_slices[0]).c_str(),
             static_cast<int>(k_slice_slices.size()));
  LITERT_LOG(LITERT_INFO, "  v_cache_slices[0]   %s (x%d)",
             FormatTensorShape(v_cache_slices[0]).c_str(),
             static_cast<int>(v_cache_slices.size()));
  LITERT_LOG(LITERT_INFO, "  v_slice_slices[0]   %s (x%d)",
             FormatTensorShape(v_slice_slices[0]).c_str(),
             static_cast<int>(v_slice_slices.size()));

  // Phase 2b: build a per-head SHA subgraph for every attention head.
  std::vector<litert::Tensor> sha_outputs;
  sha_outputs.reserve(num_attn_heads);
  for (int32_t i = 0; i < num_attn_heads; ++i) {
    const int32_t kv_idx = i / num_attn_per_kv_heads;
    auto sha_out = BuildSingleSHA(
        builder, num_attn_per_kv_heads, scale_mul_slices[i],
        k_cache_slices[kv_idx], k_slice_slices[kv_idx],
        v_cache_slices[kv_idx], v_slice_slices[kv_idx], mul_op, mm1_op,
        mm2_op, concat_op, bias_add_op, softmax_op, slice_left_op,
        slice_right_op, mm3_op, mm4_op, final_add_op);
    if (sha_out.Get() == nullptr) {
      LITERT_LOG(LITERT_WARNING, "BuildSingleSHA failed for head %d.", i);
      return kLiteRtStatusPatternNoMatch;
    }
    sha_outputs.push_back(sha_out);
  }
  LITERT_LOG(LITERT_INFO, "Built %d per-head SHA subgraphs. Head[0] out: %s",
             static_cast<int>(sha_outputs.size()),
             FormatTensorShape(sha_outputs[0]).c_str());

  // Phase 2c: stitch + replace + erase. Gated by kCommitPhase2c above.
  if (!kCommitPhase2c) {
    LITERT_LOG(LITERT_INFO,
               "Phase 2c gated off — %d SHA heads built in-builder (dry-run).",
               num_attn_heads);
    return kLiteRtStatusPatternNoMatch;
  }

  if (sha_outputs.size() != static_cast<size_t>(num_attn_heads)) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto per_head_out_dims = DimsOf(sha_outputs[0]);
  if (per_head_out_dims.size() < 4) return kLiteRtStatusPatternNoMatch;
  std::vector<int32_t> stitched_dims = per_head_out_dims;
  stitched_dims[3] = per_head_out_dims[3] * num_attn_heads;
  auto stitched = CloneTensorWithShape(
      builder, final_add_op.Outputs()[0], absl::MakeConstSpan(stitched_dims));
  if (stitched.Get() == nullptr) return kLiteRtStatusPatternNoMatch;
  std::vector<litert::Tensor> final_concat_inputs(sha_outputs.begin(),
                                                  sha_outputs.end());
  std::vector<litert::Tensor> final_concat_outputs{stitched};
  auto final_concat = builder.BuildOp(kLiteRtOpCodeTflConcatenation,
                                      final_concat_inputs,
                                      final_concat_outputs);
  litert::ConcatenationOptions concat_opts;
  concat_opts.axis = 3;
  concat_opts.fused_activation_function = litert::kActivationFunctionTypeNone;
  auto copts_res = builder.SetOpOptions(final_concat, std::move(concat_opts));
  (void)copts_res;

  // Replace the root Reshape (R5) with a new Reshape that reshapes the
  // stitched concat output into R5's original output shape. ReplaceOp reuses
  // R5's output tensor so all downstream consumers remain correctly wired.
  auto root_out_dims = DimsOf(root_op.Outputs()[0]);
  auto final_shape_const =
      BuildInt32Const1D(builder, absl::MakeConstSpan(root_out_dims));
  if (final_shape_const.Get() == nullptr) return kLiteRtStatusPatternNoMatch;
  std::vector<litert::Tensor> final_reshape_inputs{stitched, final_shape_const};
  builder.ReplaceOp(root_op, kLiteRtOpCodeTflReshape, final_reshape_inputs);

  // Erase the remaining 16 original pattern ops, leaves-first.
  if (!kSkipEraseOriginals) {
    litert::Op ops_to_erase[] = {transpose_op,   post_reshape_op, final_add_op,
                                 mm3_op,         mm4_op,          slice_left_op,
                                 slice_right_op, softmax_op,      r3_op,
                                 bias_add_op,    r2_op,           concat_op,
                                 mm1_op,         mm2_op,          r1_op,
                                 mul_op};
    for (auto& old_op : ops_to_erase) builder.EraseOp(old_op);
  } else {
    LITERT_LOG(LITERT_INFO,
               "Skipping erase of originals (diagnostic). New ops added on "
               "top of original pattern.");
  }

  LITERT_LOG(LITERT_INFO,
             "MHA -> SHA rewrite committed: %d SHA heads, stitched %s, "
             "erased_originals=%d.",
             num_attn_heads, FormatTensorShape(stitched).c_str(),
             kSkipEraseOriginals ? 0 : 16);
  return kLiteRtStatusOk;
}

LiteRtStatus DebugReplaceQScaleMulWithAdd(LiteRtBuilder builder_ptr,
                                          LiteRtOp op) {
  using litert::m_Any;
  using litert::m_CaptureOrSameAs;
  using litert::m_Op;
  using litert::m_OpCode;

  litert::Builder builder(builder_ptr);
  litert::Op root_op(op);

  // Shared structural captures.
  litert::Op softmax_op(nullptr);
  litert::Tensor q_scaled(nullptr);

  // Capture we care about: mul_op (the Q-scale Mul to replace).
  litert::Op mul_op(nullptr);

  // Full 17-op matcher — same as ApplyCompositeAttentionTransformation.
  auto match_q_scaled = m_CaptureOrSameAs(
      &q_scaled,
      m_Op<kLiteRtOpCodeTflReshape>(
          m_CaptureOrSameAs(
              &mul_op, m_Op<kLiteRtOpCodeTflMul>(m_Any(), m_Any())),
          m_Any()));

  auto match_scores = m_Op<kLiteRtOpCodeTflConcatenation>(
      m_Op<kLiteRtOpCodeTflBatchMatmul>(match_q_scaled, m_Any()),
      m_Op<kLiteRtOpCodeTflBatchMatmul>(match_q_scaled, m_Any()));

  auto match_softmax = m_Op<kLiteRtOpCodeTflSoftmax>(
      m_Op<kLiteRtOpCodeTflReshape>(
          m_Op<kLiteRtOpCodeTflAdd>(
              m_Op<kLiteRtOpCodeTflReshape>(match_scores, m_Any()), m_Any()),
          m_Any()));

  auto match_left_arm = m_Op<kLiteRtOpCodeTflBatchMatmul>(
      m_Op<kLiteRtOpCodeTflSlice>(
          m_CaptureOrSameAs(&softmax_op, match_softmax), m_Any(), m_Any()),
      m_Any());

  auto match_right_arm = m_Op<kLiteRtOpCodeTflBatchMatmul>(
      m_Op<kLiteRtOpCodeTflSlice>(
          m_CaptureOrSameAs(&softmax_op, m_OpCode<kLiteRtOpCodeTflSoftmax>()),
          m_Any(), m_Any()),
      m_Any());

  auto match_root = m_Op<kLiteRtOpCodeTflReshape>(
      m_Op<kLiteRtOpCodeTflTranspose>(
          m_Op<kLiteRtOpCodeTflReshape>(
              m_Op<kLiteRtOpCodeTflAdd>(match_left_arm, match_right_arm),
              m_Any()),
          m_Any()),
      m_Any());

  if (!litert::Match(root_op, match_root)) {
    return kLiteRtStatusPatternNoMatch;
  }

  // Collect Mul inputs before ReplaceOp erases mul_op (Drop disconnects them).
  litert::Tensor scale_mul_in = mul_op.Inputs()[0];  // Q tensor
  litert::Tensor scale_const  = mul_op.Inputs()[1];  // scale factor

  // --- Replace Q-scale Mul → Add, with the original Mul inputs ---
  // The Add's output tensor reuses the original Mul output (via ReplaceOp)
  // and is renamed for Netron visibility.
  std::vector<litert::Tensor> add_inputs{litert::Tensor(scale_mul_in.Get()),
                                         litert::Tensor(scale_const.Get())};
  litert::Op new_add =
      builder.ReplaceOp(mul_op, kLiteRtOpCodeTflAdd, add_inputs);

  if (!new_add.Outputs().empty()) {
    new_add.Outputs()[0].Get()->SetName("DBG_KVSwapAttn_QScaleAdd_ReplacedMul");
  }

  litert::AddOptions add_opts;
  add_opts.fused_activation_function = litert::kActivationFunctionTypeNone;
  auto opts_res = builder.SetOpOptions(new_add, std::move(add_opts));
  (void)opts_res;

  LITERT_LOG(LITERT_INFO,
             "DebugReplaceQScaleMulWithAdd: replaced Q-scale Mul(%s) with Add.",
             FormatTensorShape(scale_mul_in).c_str());
  return kLiteRtStatusOk;
}

}  // extern "C"
