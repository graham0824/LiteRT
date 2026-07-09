// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/spatial_transform_nd_op_builder.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

namespace {

constexpr size_t kInputIndex = 0;
constexpr size_t kBlockShapeIndex = 1;
// crops (BatchToSpaceNd) or paddings (SpaceToBatchNd).
constexpr size_t kPadOrCropsIndex = 2;
constexpr size_t kOutputIndex = 0;

// BatchToSpaceNd and SpaceToBatchNd share the same TFLite input layout
// (input, static block_shape, static crops/paddings) and the same QNN op
// shape (input tensor + block_size tensor param + a crops/pad_amount tensor
// param). Only the QNN op type and the second param name differ.
std::vector<OpWrapper> BuildSpatialTransformNdOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const char* qnn_op_type,
    const char* block_size_param_name, const char* pad_or_crops_param_name) {
  std::vector<OpWrapper> res;

  TensorWrapper& block_shape_tensor = inputs[kBlockShapeIndex];
  if (!block_shape_tensor.IsTensorStatic()) {
    QNN_LOG_ERROR("QNN only supports static block_shape tensor.")
    return res;
  }
  auto* converted_block_shape =
      tensor_pool.ConvertStaticTensorFrom<std::uint32_t>(block_shape_tensor);
  if (converted_block_shape == nullptr) {
    QNN_LOG_ERROR("Failed to convert uint32 block_shape tensor.")
    return res;
  }

  TensorWrapper& pad_or_crops_tensor = inputs[kPadOrCropsIndex];
  if (!pad_or_crops_tensor.IsTensorStatic()) {
    QNN_LOG_ERROR("QNN only supports static crops/paddings tensor.")
    return res;
  }
  auto* converted_pad_or_crops =
      tensor_pool.ConvertStaticTensorFrom<std::uint32_t>(pad_or_crops_tensor);
  if (converted_pad_or_crops == nullptr) {
    QNN_LOG_ERROR("Failed to convert uint32 crops/paddings tensor.")
    return res;
  }

  OpWrapper& op = CreateOpWrapper(res, qnn_op_type);
  op.AddInputTensor(inputs[kInputIndex]);
  op.AddOutputTensor(outputs[kOutputIndex]);
  op.AddTensorParam(block_size_param_name, *converted_block_shape);
  op.AddTensorParam(pad_or_crops_param_name, *converted_pad_or_crops);

  return res;
}

}  // namespace

std::vector<OpWrapper> BuildBatchToSpaceNdOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  return BuildSpatialTransformNdOp(tensor_pool, inputs, outputs,
                                   QNN_OP_BATCH_TO_SPACE,
                                   QNN_OP_BATCH_TO_SPACE_PARAM_BLOCK_SIZE,
                                   QNN_OP_BATCH_TO_SPACE_PARAM_CROPS);
}

std::vector<OpWrapper> BuildSpaceToBatchNdOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  return BuildSpatialTransformNdOp(tensor_pool, inputs, outputs,
                                   QNN_OP_SPACE_TO_BATCH,
                                   QNN_OP_SPACE_TO_BATCH_PARAM_BLOCK_SIZE,
                                   QNN_OP_SPACE_TO_BATCH_PARAM_PAD_AMOUNT);
}

}  // namespace qnn
