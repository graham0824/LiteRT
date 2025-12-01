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

#include <algorithm>

#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_builder.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/core/model/model.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/schema/schema_generated.h"

using litert::Builder;
using litert::Op;
using litert::OpInputs;
using litert::OpOutputs;

extern "C" {

LiteRtStatus MatMulConvertTransformation(LiteRtBuilder builder_ptr,
                                         LiteRtOp op) {
  Builder builder = Builder(builder_ptr);
  Op root_op = Op(op);

  if (root_op.Code() != kLiteRtOpCodeTflConcatenation ||
      root_op.Inputs().size() != 2) {
    return kLiteRtStatusPatternNoMatch;
  }

  Op quant = Op(root_op.Inputs().at(0).DefiningOp().value().op);
  Op matmul = Op(root_op.Inputs().at(1).DefiningOp().value().op);
  if (quant.Code() != kLiteRtOpCodeTflQuantize) {
    std::swap(quant, matmul);
  }
  if (quant.Code() != kLiteRtOpCodeTflQuantize ||
      matmul.Code() != kLiteRtOpCodeTflBatchMatmul) {
    return kLiteRtStatusPatternNoMatch;
  }

  Op matmul_requant = Op(quant.Inputs().front().DefiningOp().value().op);
  if (matmul_requant.Code() != kLiteRtOpCodeTflBatchMatmul) {
    return kLiteRtStatusPatternNoMatch;
  }

  // Reuse the inputs of the matmul_requant.
  OpInputs inputs = matmul_requant.Inputs();
  // Reuse the outputs of the mean op.
  OpOutputs outputs = quant.Outputs();
  // Build the MatMul op with new outputs.
  Op new_matmul = builder.BuildOp(kLiteRtOpCodeTflBatchMatmul, inputs, outputs);
  // Set MatMul options
  tflite::BatchMatMulOptionsT options;
  options.adj_x = false;
  options.adj_y = true;
  options.asymmetric_quantize_inputs = false;
  ::litert::internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_BatchMatMulOptions;
  tfl_options.Set(std::move(options));
  ::litert::internal::SetTflOptions(*new_matmul.Get(), std::move(tfl_options));
  // Erase the original MatMul and Quant ops.
  builder.EraseOp(quant);
  builder.EraseOp(matmul_requant);
  LITERT_LOG(LITERT_INFO, "Hello MatMulConvert!");
  return kLiteRtStatusOk;
}

LiteRtStatus SqrtMeanSquareTransformation(LiteRtBuilder builder_ptr,
                                          LiteRtOp op) {
  Builder builder = Builder(builder_ptr);
  Op root_op = Op(op);

  // Pattern Match
  if (root_op.Code() != kLiteRtOpCodeTflSqrt) {
    return kLiteRtStatusPatternNoMatch;
  }
  Op mean_op = Op(root_op.Inputs().front().DefiningOp().value().op);
  if (mean_op.Code() != kLiteRtOpCodeTflMean) {
    return kLiteRtStatusPatternNoMatch;
  }
  Op square_op = Op(mean_op.Inputs().front().DefiningOp().value().op);
  if (square_op.Code() != kLiteRtOpCodeTflMul) {
    return kLiteRtStatusPatternNoMatch;
  }
  if (square_op.Inputs().size() != 2) {
    return kLiteRtStatusPatternNoMatch;
  }
  if (square_op.Inputs().at(0).Get() != square_op.Inputs().at(1).Get()) {
    return kLiteRtStatusPatternNoMatch;
  }
  // Reuse the inputs of the mul(square op).
  OpInputs inputs = square_op.Inputs();
  // Reuse the outputs of the mean op.
  OpOutputs outputs = mean_op.Outputs();
  // Build the abs op.
  builder.BuildOp(kLiteRtOpCodeTflAbs, inputs, outputs);
  // Erase the original ops.
  builder.EraseOp(square_op);
  builder.EraseOp(mean_op);
  return kLiteRtStatusOk;
}

LiteRtStatus DummyTransformation(LiteRtBuilder builder_ptr, LiteRtOp op) {
  return kLiteRtStatusPatternNoMatch;
}

}  // extern "C"
