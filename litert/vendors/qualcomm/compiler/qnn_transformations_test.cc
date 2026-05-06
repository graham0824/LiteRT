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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_builder.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/core/model/model.h"
#include "litert/vendors/qualcomm/compiler/qnn_transformations.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert {
namespace {

TEST(QnnFuseMatMulRequantTest, SuccessFusesRequantization) {
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder;

  auto& input0 = subgraph.EmplaceTensor();
  auto& input1 = subgraph.EmplaceTensor();
  auto& matmul_to_convert = subgraph.EmplaceTensor();
  auto& output0 = subgraph.EmplaceTensor();

  matmul_to_convert.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 10}));
  output0.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 10}));

  output0.SetQarams(MakePerTensorQuantization(0.5f, 10));

  auto& matmul0 = subgraph.EmplaceOp();
  matmul0.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  litert::internal::TflOptions matmul0_opts;
  matmul0_opts.type = tflite::BuiltinOptions_BatchMatMulOptions;
  auto options = std::make_unique<tflite::BatchMatMulOptionsT>();
  options->adj_x = true;
  matmul0_opts.value = options.release();
  litert::internal::SetTflOptions(matmul0, std::move(matmul0_opts));

  internal::AttachInput(&input0, matmul0);
  internal::AttachInput(&input1, matmul0);
  internal::AttachOutput(&matmul_to_convert, matmul0);

  auto& convert = subgraph.EmplaceOp();
  convert.SetOpCode(kLiteRtOpCodeTflQuantize);
  internal::AttachInput(&matmul_to_convert, convert);
  internal::AttachOutput(&output0, convert);

  EXPECT_EQ(FuseMatMulRequantTransformation(&builder, &convert),
            kLiteRtStatusOk);
  builder.ApplyChanges(&subgraph);

  int matmul_count = 0;
  int quantize_count = 0;
  for (const auto& op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflBatchMatmul) matmul_count++;
    if (op->OpCode() == kLiteRtOpCodeTflQuantize) quantize_count++;
  }
  EXPECT_EQ(matmul_count, 1);
  EXPECT_EQ(quantize_count, 0);
  ASSERT_EQ(subgraph.Ops().size(), 1);

  const auto& matmul_final = *subgraph.Ops().front();
  EXPECT_EQ(matmul_final.OpCode(), kLiteRtOpCodeTflBatchMatmul);
  ASSERT_EQ(matmul_final.Outputs().size(), 1);
  EXPECT_EQ(matmul_final.Outputs()[0], &output0);

  EXPECT_EQ(output0.Qparams().first, kLiteRtQuantizationPerTensor);
  EXPECT_FLOAT_EQ(output0.Qparams().second.per_tensor.scale, 0.5f);
  EXPECT_EQ(output0.Qparams().second.per_tensor.zero_point, 10);

  const auto& opts = litert::internal::GetTflOptions(matmul_final);
  ASSERT_TRUE(opts.value != nullptr);
  EXPECT_TRUE(opts.AsBatchMatMulOptions()->adj_x);
}

TEST(QnnFuseMatMulRequantTest, NoMatchOnElementTypeMismatch) {
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder;

  auto& input0 = subgraph.EmplaceTensor();
  auto& input1 = subgraph.EmplaceTensor();
  auto& matmul_to_convert = subgraph.EmplaceTensor();
  auto& output0 = subgraph.EmplaceTensor();

  matmul_to_convert.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 10}));
  output0.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 10}));

  auto& matmul0 = subgraph.EmplaceOp();
  matmul0.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  internal::AttachInput(&input0, matmul0);
  internal::AttachInput(&input1, matmul0);
  internal::AttachOutput(&matmul_to_convert, matmul0);

  auto& convert = subgraph.EmplaceOp();
  convert.SetOpCode(kLiteRtOpCodeTflQuantize);
  internal::AttachInput(&matmul_to_convert, convert);
  internal::AttachOutput(&output0, convert);

  EXPECT_EQ(FuseMatMulRequantTransformation(&builder, &convert),
            kLiteRtStatusPatternNoMatch);

  builder.ApplyChanges(&subgraph);

  ASSERT_EQ(subgraph.Ops().size(), 2);
  EXPECT_EQ(subgraph.Ops()[0]->OpCode(), kLiteRtOpCodeTflBatchMatmul);
  EXPECT_EQ(subgraph.Ops()[1]->OpCode(), kLiteRtOpCodeTflQuantize);
}

TEST(QnnFuseMatMulRequantTest, ComplexDagPreservesDownstreamOps) {
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder;

  auto& input0 = subgraph.EmplaceTensor();
  auto& input1 = subgraph.EmplaceTensor();
  auto& input2 = subgraph.EmplaceTensor();
  auto& input3 = subgraph.EmplaceTensor();
  auto& inter0 = subgraph.EmplaceTensor();
  auto& inter1 = subgraph.EmplaceTensor();
  auto& inter2 = subgraph.EmplaceTensor();
  auto& out = subgraph.EmplaceTensor();

  inter0.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 10}));
  inter1.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 10}));
  inter2.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 10}));
  out.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 20}));

  auto& matmul0 = subgraph.EmplaceOp();
  matmul0.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  {
    litert::internal::TflOptions opts;
    opts.type = tflite::BuiltinOptions_BatchMatMulOptions;
    opts.value = new tflite::BatchMatMulOptionsT();
    litert::internal::SetTflOptions(matmul0, std::move(opts));
  }
  internal::AttachInput(&input0, matmul0);
  internal::AttachInput(&input1, matmul0);
  internal::AttachOutput(&inter0, matmul0);

  auto& quant = subgraph.EmplaceOp();
  quant.SetOpCode(kLiteRtOpCodeTflQuantize);
  internal::AttachInput(&inter0, quant);
  internal::AttachOutput(&inter1, quant);

  auto& matmul1 = subgraph.EmplaceOp();
  matmul1.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  {
    litert::internal::TflOptions opts;
    opts.type = tflite::BuiltinOptions_BatchMatMulOptions;
    opts.value = new tflite::BatchMatMulOptionsT();
    litert::internal::SetTflOptions(matmul1, std::move(opts));
  }
  internal::AttachInput(&input2, matmul1);
  internal::AttachInput(&input3, matmul1);
  internal::AttachOutput(&inter2, matmul1);

  auto& concat = subgraph.EmplaceOp();
  concat.SetOpCode(kLiteRtOpCodeTflConcatenation);
  internal::AttachInput(&inter1, concat);
  internal::AttachInput(&inter2, concat);
  internal::AttachOutput(&out, concat);

  EXPECT_EQ(FuseMatMulRequantTransformation(&builder, &quant),
            kLiteRtStatusOk);
  builder.ApplyChanges(&subgraph);

  int matmul_count = 0;
  int quantize_count = 0;
  int concat_count = 0;
  for (const auto& op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflBatchMatmul) matmul_count++;
    if (op->OpCode() == kLiteRtOpCodeTflQuantize) quantize_count++;
    if (op->OpCode() == kLiteRtOpCodeTflConcatenation) concat_count++;
  }
  EXPECT_EQ(matmul_count, 2);
  EXPECT_EQ(quantize_count, 0);
  EXPECT_EQ(concat_count, 1);

  LiteRtOpT* final_concat = nullptr;
  for (const auto& op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflConcatenation) {
      final_concat = op;
      break;
    }
  }
  ASSERT_NE(final_concat, nullptr);
  ASSERT_EQ(final_concat->Inputs().size(), 2);
  EXPECT_EQ(final_concat->Inputs()[0], &inter1);
  EXPECT_EQ(final_concat->Inputs()[1], &inter2);

  LiteRtOpT* new_matmul0 = nullptr;
  LiteRtOpT* matmul1_ptr = nullptr;
  for (const auto& op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflBatchMatmul) {
      if (!op->Outputs().empty()) {
        if (op->Outputs()[0] == &inter1) new_matmul0 = op;
        if (op->Outputs()[0] == &inter2) matmul1_ptr = op;
      }
    }
  }
  ASSERT_NE(new_matmul0, nullptr);
  ASSERT_NE(matmul1_ptr, nullptr);
  EXPECT_EQ(inter1.DefiningOp(), new_matmul0);

  ASSERT_EQ(inter1.NumUses(), 1);
  EXPECT_EQ(inter1.Users()[0], final_concat);
  EXPECT_EQ(inter1.UserArgInds()[0], 0);
  ASSERT_EQ(inter2.NumUses(), 1);
  EXPECT_EQ(inter2.Users()[0], final_concat);
  EXPECT_EQ(inter2.UserArgInds()[0], 1);

  ASSERT_EQ(input0.NumUses(), 1);
  EXPECT_EQ(input0.Users()[0], new_matmul0);
}

TEST(QnnFuseMatMulRequantTest, SharedInputPreservesBothUsers) {
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder;

  auto& input_shared = subgraph.EmplaceTensor();
  auto& input1 = subgraph.EmplaceTensor();
  auto& input2 = subgraph.EmplaceTensor();
  auto& inter0 = subgraph.EmplaceTensor();
  auto& inter1 = subgraph.EmplaceTensor();
  auto& out1 = subgraph.EmplaceTensor();

  inter0.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 10}));
  inter1.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 10}));
  out1.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 10}));

  auto& matmul0 = subgraph.EmplaceOp();
  matmul0.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  {
    litert::internal::TflOptions opts;
    opts.type = tflite::BuiltinOptions_BatchMatMulOptions;
    opts.value = new tflite::BatchMatMulOptionsT();
    litert::internal::SetTflOptions(matmul0, std::move(opts));
  }
  internal::AttachInput(&input_shared, matmul0);
  internal::AttachInput(&input1, matmul0);
  internal::AttachOutput(&inter0, matmul0);

  auto& quant = subgraph.EmplaceOp();
  quant.SetOpCode(kLiteRtOpCodeTflQuantize);
  internal::AttachInput(&inter0, quant);
  internal::AttachOutput(&inter1, quant);

  auto& matmul1 = subgraph.EmplaceOp();
  matmul1.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  {
    litert::internal::TflOptions opts;
    opts.type = tflite::BuiltinOptions_BatchMatMulOptions;
    opts.value = new tflite::BatchMatMulOptionsT();
    litert::internal::SetTflOptions(matmul1, std::move(opts));
  }
  internal::AttachInput(&input_shared, matmul1);
  internal::AttachInput(&input2, matmul1);
  internal::AttachOutput(&out1, matmul1);

  ASSERT_EQ(input_shared.NumUses(), 2);

  EXPECT_EQ(FuseMatMulRequantTransformation(&builder, &quant),
            kLiteRtStatusOk);
  builder.ApplyChanges(&subgraph);

  LiteRtOpT* new_matmul0 = nullptr;
  LiteRtOpT* matmul1_ptr = nullptr;
  for (const auto& op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflBatchMatmul) {
      if (!op->Outputs().empty()) {
        if (op->Outputs()[0] == &inter1) new_matmul0 = op;
        if (op->Outputs()[0] == &out1) matmul1_ptr = op;
      }
    }
  }
  ASSERT_NE(new_matmul0, nullptr);
  ASSERT_NE(matmul1_ptr, nullptr);

  ASSERT_EQ(input_shared.NumUses(), 2);
  bool found_new_matmul = false;
  bool found_matmul1 = false;
  for (size_t i = 0; i < input_shared.NumUses(); ++i) {
    if (input_shared.Users()[i] == new_matmul0) found_new_matmul = true;
    if (input_shared.Users()[i] == matmul1_ptr) found_matmul1 = true;
  }
  EXPECT_TRUE(found_new_matmul);
  EXPECT_TRUE(found_matmul1);
}

// ---------------------------------------------------------------------------
// MatchCompositeAttentionPattern tests
// ---------------------------------------------------------------------------

// Helper: build the full composite-attention subgraph and return a pointer to
// the root Add op (A2). All objects are owned by `subgraph`.
//
// Graph topology (matching the full KV-swapped attention pattern):
//   IN_TOP -> M0(Mul) -> R1(Reshape) -+-> MM1(BatchMatmul) -+
//                                     +-> MM2(BatchMatmul) -+-> C1(Concat)
//   C1 -> R2(Reshape) -> A1(Add, +IN_LEFT) -> R3(Reshape) -> S1(Softmax)
//   S1 -> SL(Slice) -> MM3(BatchMatmul) -+
//   S1 -> SR(Slice) -> MM4(BatchMatmul) -+-> A2(Add) -> R4(Reshape)
//                                                    -> T1(Transpose)
//                                                    -> R5(Reshape) [root]
static LiteRtOpT* BuildCompositeAttentionSubgraph(LiteRtSubgraphT& sg) {
  // ---- tensors ----
  auto& in_top = sg.EmplaceTensor();
  auto& scale = sg.EmplaceTensor();
  auto& in_left = sg.EmplaceTensor();

  // shape constants (Reshape second input)
  auto& r1_shape = sg.EmplaceTensor();
  auto& r2_shape = sg.EmplaceTensor();
  auto& r3_shape = sg.EmplaceTensor();
  auto& r4_shape = sg.EmplaceTensor();
  auto& r5_shape = sg.EmplaceTensor();

  // transpose perm
  auto& t1_perm = sg.EmplaceTensor();

  // weight tensors for matmuls
  auto& mm1_weight = sg.EmplaceTensor();
  auto& mm2_weight = sg.EmplaceTensor();
  auto& mm3_weight = sg.EmplaceTensor();
  auto& mm4_weight = sg.EmplaceTensor();

  // slice params
  auto& sl_begin = sg.EmplaceTensor();
  auto& sl_size = sg.EmplaceTensor();
  auto& sr_begin = sg.EmplaceTensor();
  auto& sr_size = sg.EmplaceTensor();

  // intermediate outputs
  auto& m0_out = sg.EmplaceTensor();
  auto& r1_out = sg.EmplaceTensor();
  auto& mm1_out = sg.EmplaceTensor();
  auto& mm2_out = sg.EmplaceTensor();
  auto& c1_out = sg.EmplaceTensor();
  auto& r2_out = sg.EmplaceTensor();
  auto& a1_out = sg.EmplaceTensor();
  auto& r3_out = sg.EmplaceTensor();
  auto& s1_out = sg.EmplaceTensor();
  auto& sl_out = sg.EmplaceTensor();
  auto& sr_out = sg.EmplaceTensor();
  auto& mm3_out = sg.EmplaceTensor();
  auto& mm4_out = sg.EmplaceTensor();
  auto& a2_out = sg.EmplaceTensor();
  auto& r4_out = sg.EmplaceTensor();
  auto& t1_out = sg.EmplaceTensor();
  auto& r5_out = sg.EmplaceTensor();

  // ---- ops ----
  // M0: Mul(in_top, scale) -> m0_out
  auto& m0 = sg.EmplaceOp();
  m0.SetOpCode(kLiteRtOpCodeTflMul);
  internal::AttachInput(&in_top, m0);
  internal::AttachInput(&scale, m0);
  internal::AttachOutput(&m0_out, m0);

  // R1: Reshape(m0_out, r1_shape) -> r1_out
  auto& r1 = sg.EmplaceOp();
  r1.SetOpCode(kLiteRtOpCodeTflReshape);
  internal::AttachInput(&m0_out, r1);
  internal::AttachInput(&r1_shape, r1);
  internal::AttachOutput(&r1_out, r1);

  // MM1: BatchMatmul(r1_out, mm1_weight) -> mm1_out
  auto& mm1 = sg.EmplaceOp();
  mm1.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  internal::AttachInput(&r1_out, mm1);
  internal::AttachInput(&mm1_weight, mm1);
  internal::AttachOutput(&mm1_out, mm1);

  // MM2: BatchMatmul(r1_out, mm2_weight) -> mm2_out
  auto& mm2 = sg.EmplaceOp();
  mm2.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  internal::AttachInput(&r1_out, mm2);
  internal::AttachInput(&mm2_weight, mm2);
  internal::AttachOutput(&mm2_out, mm2);

  // C1: Concat(mm1_out, mm2_out) -> c1_out
  auto& c1 = sg.EmplaceOp();
  c1.SetOpCode(kLiteRtOpCodeTflConcatenation);
  internal::AttachInput(&mm1_out, c1);
  internal::AttachInput(&mm2_out, c1);
  internal::AttachOutput(&c1_out, c1);

  // R2: Reshape(c1_out, r2_shape) -> r2_out
  auto& r2 = sg.EmplaceOp();
  r2.SetOpCode(kLiteRtOpCodeTflReshape);
  internal::AttachInput(&c1_out, r2);
  internal::AttachInput(&r2_shape, r2);
  internal::AttachOutput(&r2_out, r2);

  // A1: Add(r2_out, in_left) -> a1_out
  auto& a1 = sg.EmplaceOp();
  a1.SetOpCode(kLiteRtOpCodeTflAdd);
  internal::AttachInput(&r2_out, a1);
  internal::AttachInput(&in_left, a1);
  internal::AttachOutput(&a1_out, a1);

  // R3: Reshape(a1_out, r3_shape) -> r3_out
  auto& r3 = sg.EmplaceOp();
  r3.SetOpCode(kLiteRtOpCodeTflReshape);
  internal::AttachInput(&a1_out, r3);
  internal::AttachInput(&r3_shape, r3);
  internal::AttachOutput(&r3_out, r3);

  // S1: Softmax(r3_out) -> s1_out
  auto& s1 = sg.EmplaceOp();
  s1.SetOpCode(kLiteRtOpCodeTflSoftmax);
  internal::AttachInput(&r3_out, s1);
  internal::AttachOutput(&s1_out, s1);

  // SL: Slice(s1_out, sl_begin, sl_size) -> sl_out
  auto& sl = sg.EmplaceOp();
  sl.SetOpCode(kLiteRtOpCodeTflSlice);
  internal::AttachInput(&s1_out, sl);
  internal::AttachInput(&sl_begin, sl);
  internal::AttachInput(&sl_size, sl);
  internal::AttachOutput(&sl_out, sl);

  // SR: Slice(s1_out, sr_begin, sr_size) -> sr_out
  auto& sr = sg.EmplaceOp();
  sr.SetOpCode(kLiteRtOpCodeTflSlice);
  internal::AttachInput(&s1_out, sr);
  internal::AttachInput(&sr_begin, sr);
  internal::AttachInput(&sr_size, sr);
  internal::AttachOutput(&sr_out, sr);

  // MM3: BatchMatmul(sl_out, mm3_weight) -> mm3_out
  auto& mm3 = sg.EmplaceOp();
  mm3.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  internal::AttachInput(&sl_out, mm3);
  internal::AttachInput(&mm3_weight, mm3);
  internal::AttachOutput(&mm3_out, mm3);

  // MM4: BatchMatmul(sr_out, mm4_weight) -> mm4_out
  auto& mm4 = sg.EmplaceOp();
  mm4.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  internal::AttachInput(&sr_out, mm4);
  internal::AttachInput(&mm4_weight, mm4);
  internal::AttachOutput(&mm4_out, mm4);

  // A2: Add(mm3_out, mm4_out) -> a2_out
  auto& a2 = sg.EmplaceOp();
  a2.SetOpCode(kLiteRtOpCodeTflAdd);
  internal::AttachInput(&mm3_out, a2);
  internal::AttachInput(&mm4_out, a2);
  internal::AttachOutput(&a2_out, a2);

  // R4: Reshape(a2_out, r4_shape) -> r4_out
  auto& r4 = sg.EmplaceOp();
  r4.SetOpCode(kLiteRtOpCodeTflReshape);
  internal::AttachInput(&a2_out, r4);
  internal::AttachInput(&r4_shape, r4);
  internal::AttachOutput(&r4_out, r4);

  // T1: Transpose(r4_out, t1_perm) -> t1_out
  auto& t1 = sg.EmplaceOp();
  t1.SetOpCode(kLiteRtOpCodeTflTranspose);
  internal::AttachInput(&r4_out, t1);
  internal::AttachInput(&t1_perm, t1);
  internal::AttachOutput(&t1_out, t1);

  // R5: Reshape(t1_out, r5_shape) -> r5_out  [root]
  auto& r5 = sg.EmplaceOp();
  r5.SetOpCode(kLiteRtOpCodeTflReshape);
  internal::AttachInput(&t1_out, r5);
  internal::AttachInput(&r5_shape, r5);
  internal::AttachOutput(&r5_out, r5);

  return &r5;
}

TEST(MatchCompositeAttentionPatternTest, SuccessMatchesFullPattern) {
  LiteRtSubgraphT subgraph;
  LiteRtOpT* a2 = BuildCompositeAttentionSubgraph(subgraph);

  EXPECT_EQ(MatchCompositeAttentionPattern(a2), kLiteRtStatusOk);
}

TEST(MatchCompositeAttentionPatternTest, NoMatchOnWrongRootOp) {
  LiteRtSubgraphT subgraph;
  // Build the full pattern but pass the Softmax op instead of the root Add.
  BuildCompositeAttentionSubgraph(subgraph);

  LiteRtOpT* softmax_op = nullptr;
  for (const auto& op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflSoftmax) {
      softmax_op = op;
      break;
    }
  }
  ASSERT_NE(softmax_op, nullptr);
  EXPECT_EQ(MatchCompositeAttentionPattern(softmax_op),
            kLiteRtStatusPatternNoMatch);
}

TEST(MatchCompositeAttentionPatternTest, NoMatchWhenSlicesUseDifferentSoftmax) {
  LiteRtSubgraphT subgraph;
  // Build the base pattern but wire SR to a second, independent Softmax.
  auto& in_top = subgraph.EmplaceTensor();
  auto& scale = subgraph.EmplaceTensor();
  auto& in_left = subgraph.EmplaceTensor();
  auto& r1_shape = subgraph.EmplaceTensor();
  auto& r2_shape = subgraph.EmplaceTensor();
  auto& r3_shape = subgraph.EmplaceTensor();
  auto& mm1_weight = subgraph.EmplaceTensor();
  auto& mm2_weight = subgraph.EmplaceTensor();
  auto& mm3_weight = subgraph.EmplaceTensor();
  auto& mm4_weight = subgraph.EmplaceTensor();
  auto& sl_begin = subgraph.EmplaceTensor();
  auto& sl_size = subgraph.EmplaceTensor();
  auto& sr_begin = subgraph.EmplaceTensor();
  auto& sr_size = subgraph.EmplaceTensor();
  auto& m0_out = subgraph.EmplaceTensor();
  auto& r1_out = subgraph.EmplaceTensor();
  auto& mm1_out = subgraph.EmplaceTensor();
  auto& mm2_out = subgraph.EmplaceTensor();
  auto& c1_out = subgraph.EmplaceTensor();
  auto& r2_out = subgraph.EmplaceTensor();
  auto& a1_out = subgraph.EmplaceTensor();
  auto& r3_out = subgraph.EmplaceTensor();
  auto& s1_out = subgraph.EmplaceTensor();
  auto& s2_out = subgraph.EmplaceTensor();  // second, independent softmax
  auto& sl_out = subgraph.EmplaceTensor();
  auto& sr_out = subgraph.EmplaceTensor();
  auto& mm3_out = subgraph.EmplaceTensor();
  auto& mm4_out = subgraph.EmplaceTensor();
  auto& a2_out = subgraph.EmplaceTensor();

  // Q-scale prefix: Mul -> Reshape shared by both projection matmuls.
  auto& m0 = subgraph.EmplaceOp();
  m0.SetOpCode(kLiteRtOpCodeTflMul);
  internal::AttachInput(&in_top, m0);
  internal::AttachInput(&scale, m0);
  internal::AttachOutput(&m0_out, m0);

  auto& r1 = subgraph.EmplaceOp();
  r1.SetOpCode(kLiteRtOpCodeTflReshape);
  internal::AttachInput(&m0_out, r1);
  internal::AttachInput(&r1_shape, r1);
  internal::AttachOutput(&r1_out, r1);

  auto& mm1 = subgraph.EmplaceOp();
  mm1.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  internal::AttachInput(&r1_out, mm1);
  internal::AttachInput(&mm1_weight, mm1);
  internal::AttachOutput(&mm1_out, mm1);

  auto& mm2 = subgraph.EmplaceOp();
  mm2.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  internal::AttachInput(&r1_out, mm2);
  internal::AttachInput(&mm2_weight, mm2);
  internal::AttachOutput(&mm2_out, mm2);

  auto& c1 = subgraph.EmplaceOp();
  c1.SetOpCode(kLiteRtOpCodeTflConcatenation);
  internal::AttachInput(&mm1_out, c1);
  internal::AttachInput(&mm2_out, c1);
  internal::AttachOutput(&c1_out, c1);

  auto& r2 = subgraph.EmplaceOp();
  r2.SetOpCode(kLiteRtOpCodeTflReshape);
  internal::AttachInput(&c1_out, r2);
  internal::AttachInput(&r2_shape, r2);
  internal::AttachOutput(&r2_out, r2);

  auto& a1 = subgraph.EmplaceOp();
  a1.SetOpCode(kLiteRtOpCodeTflAdd);
  internal::AttachInput(&r2_out, a1);
  internal::AttachInput(&in_left, a1);
  internal::AttachOutput(&a1_out, a1);

  auto& r3 = subgraph.EmplaceOp();
  r3.SetOpCode(kLiteRtOpCodeTflReshape);
  internal::AttachInput(&a1_out, r3);
  internal::AttachInput(&r3_shape, r3);
  internal::AttachOutput(&r3_out, r3);

  // S1: the "real" softmax
  auto& s1 = subgraph.EmplaceOp();
  s1.SetOpCode(kLiteRtOpCodeTflSoftmax);
  internal::AttachInput(&r3_out, s1);
  internal::AttachOutput(&s1_out, s1);

  // S2: a second, independent softmax (fed by the same r3_out for simplicity)
  auto& s2 = subgraph.EmplaceOp();
  s2.SetOpCode(kLiteRtOpCodeTflSoftmax);
  internal::AttachInput(&r3_out, s2);
  internal::AttachOutput(&s2_out, s2);

  // SL from S1, SR from S2 (mismatch).
  auto& sl = subgraph.EmplaceOp();
  sl.SetOpCode(kLiteRtOpCodeTflSlice);
  internal::AttachInput(&s1_out, sl);
  internal::AttachInput(&sl_begin, sl);
  internal::AttachInput(&sl_size, sl);
  internal::AttachOutput(&sl_out, sl);

  auto& sr = subgraph.EmplaceOp();
  sr.SetOpCode(kLiteRtOpCodeTflSlice);
  internal::AttachInput(&s2_out, sr);  // <-- different softmax
  internal::AttachInput(&sr_begin, sr);
  internal::AttachInput(&sr_size, sr);
  internal::AttachOutput(&sr_out, sr);

  auto& mm3 = subgraph.EmplaceOp();
  mm3.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  internal::AttachInput(&sl_out, mm3);
  internal::AttachInput(&mm3_weight, mm3);
  internal::AttachOutput(&mm3_out, mm3);

  auto& mm4 = subgraph.EmplaceOp();
  mm4.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  internal::AttachInput(&sr_out, mm4);
  internal::AttachInput(&mm4_weight, mm4);
  internal::AttachOutput(&mm4_out, mm4);

  auto& a2 = subgraph.EmplaceOp();
  a2.SetOpCode(kLiteRtOpCodeTflAdd);
  internal::AttachInput(&mm3_out, a2);
  internal::AttachInput(&mm4_out, a2);
  internal::AttachOutput(&a2_out, a2);

  EXPECT_EQ(MatchCompositeAttentionPattern(&a2), kLiteRtStatusPatternNoMatch);
}

TEST(MatchCompositeAttentionPatternTest, NoMatchWhenQScaleMulIsMissing) {
  // Build the full pattern but replace the pre-reshape Mul with an Add.
  LiteRtSubgraphT sg;
  auto& in_top = sg.EmplaceTensor();
  auto& scale = sg.EmplaceTensor();
  auto& in_left = sg.EmplaceTensor();
  auto& r1_shape = sg.EmplaceTensor();
  auto& r2_shape = sg.EmplaceTensor();
  auto& r3_shape = sg.EmplaceTensor();
  auto& mm1_weight = sg.EmplaceTensor();
  auto& mm2_weight = sg.EmplaceTensor();
  auto& mm3_weight = sg.EmplaceTensor();
  auto& mm4_weight = sg.EmplaceTensor();
  auto& sl_begin = sg.EmplaceTensor();
  auto& sl_size = sg.EmplaceTensor();
  auto& sr_begin = sg.EmplaceTensor();
  auto& sr_size = sg.EmplaceTensor();
  auto& m0_out = sg.EmplaceTensor();
  auto& r1_out = sg.EmplaceTensor();
  auto& mm1_out = sg.EmplaceTensor();
  auto& mm2_out = sg.EmplaceTensor();
  auto& c1_out = sg.EmplaceTensor();
  auto& r2_out = sg.EmplaceTensor();
  auto& a1_out = sg.EmplaceTensor();
  auto& r3_out = sg.EmplaceTensor();
  auto& s1_out = sg.EmplaceTensor();
  auto& sl_out = sg.EmplaceTensor();
  auto& sr_out = sg.EmplaceTensor();
  auto& mm3_out = sg.EmplaceTensor();
  auto& mm4_out = sg.EmplaceTensor();
  auto& a2_out = sg.EmplaceTensor();

  // Add instead of Mul — should break the match.
  auto& m0 = sg.EmplaceOp();
  m0.SetOpCode(kLiteRtOpCodeTflAdd);
  internal::AttachInput(&in_top, m0);
  internal::AttachInput(&scale, m0);
  internal::AttachOutput(&m0_out, m0);

  auto& r1 = sg.EmplaceOp();
  r1.SetOpCode(kLiteRtOpCodeTflReshape);
  internal::AttachInput(&m0_out, r1);
  internal::AttachInput(&r1_shape, r1);
  internal::AttachOutput(&r1_out, r1);

  auto& mm1 = sg.EmplaceOp();
  mm1.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  internal::AttachInput(&r1_out, mm1);
  internal::AttachInput(&mm1_weight, mm1);
  internal::AttachOutput(&mm1_out, mm1);

  auto& mm2 = sg.EmplaceOp();
  mm2.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  internal::AttachInput(&r1_out, mm2);
  internal::AttachInput(&mm2_weight, mm2);
  internal::AttachOutput(&mm2_out, mm2);

  auto& c1 = sg.EmplaceOp();
  c1.SetOpCode(kLiteRtOpCodeTflConcatenation);
  internal::AttachInput(&mm1_out, c1);
  internal::AttachInput(&mm2_out, c1);
  internal::AttachOutput(&c1_out, c1);

  auto& r2 = sg.EmplaceOp();
  r2.SetOpCode(kLiteRtOpCodeTflReshape);
  internal::AttachInput(&c1_out, r2);
  internal::AttachInput(&r2_shape, r2);
  internal::AttachOutput(&r2_out, r2);

  auto& a1 = sg.EmplaceOp();
  a1.SetOpCode(kLiteRtOpCodeTflAdd);
  internal::AttachInput(&r2_out, a1);
  internal::AttachInput(&in_left, a1);
  internal::AttachOutput(&a1_out, a1);

  auto& r3 = sg.EmplaceOp();
  r3.SetOpCode(kLiteRtOpCodeTflReshape);
  internal::AttachInput(&a1_out, r3);
  internal::AttachInput(&r3_shape, r3);
  internal::AttachOutput(&r3_out, r3);

  auto& s1 = sg.EmplaceOp();
  s1.SetOpCode(kLiteRtOpCodeTflSoftmax);
  internal::AttachInput(&r3_out, s1);
  internal::AttachOutput(&s1_out, s1);

  auto& sl = sg.EmplaceOp();
  sl.SetOpCode(kLiteRtOpCodeTflSlice);
  internal::AttachInput(&s1_out, sl);
  internal::AttachInput(&sl_begin, sl);
  internal::AttachInput(&sl_size, sl);
  internal::AttachOutput(&sl_out, sl);

  auto& sr = sg.EmplaceOp();
  sr.SetOpCode(kLiteRtOpCodeTflSlice);
  internal::AttachInput(&s1_out, sr);
  internal::AttachInput(&sr_begin, sr);
  internal::AttachInput(&sr_size, sr);
  internal::AttachOutput(&sr_out, sr);

  auto& mm3 = sg.EmplaceOp();
  mm3.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  internal::AttachInput(&sl_out, mm3);
  internal::AttachInput(&mm3_weight, mm3);
  internal::AttachOutput(&mm3_out, mm3);

  auto& mm4 = sg.EmplaceOp();
  mm4.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  internal::AttachInput(&sr_out, mm4);
  internal::AttachInput(&mm4_weight, mm4);
  internal::AttachOutput(&mm4_out, mm4);

  auto& a2 = sg.EmplaceOp();
  a2.SetOpCode(kLiteRtOpCodeTflAdd);
  internal::AttachInput(&mm3_out, a2);
  internal::AttachInput(&mm4_out, a2);
  internal::AttachOutput(&a2_out, a2);

  EXPECT_EQ(MatchCompositeAttentionPattern(&a2), kLiteRtStatusPatternNoMatch);
}

}  // namespace
}  // namespace litert
