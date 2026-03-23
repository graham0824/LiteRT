// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/matmul_convert.h"

#include <cstddef>
#include <functional>
#include <vector>

#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

namespace qnn {

size_t FuseMatMulConvertDecode(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  // Connection check
  if (ops[start_index].GetOutputTensor(0) !=
      ops[start_index + 1].GetInputTensor(0)) {
    return 1;
  }
  // Graph transform
  QNN_LOG_INFO("[G2G] MatMul-convert fusion (Decode)");
  ops[start_index].SwapOutputs(ops[start_index + 1]);
  if (validate_op_config(ops[start_index])) {
    ops.erase(ops.begin() + start_index + 1);
  } else {
    QNN_LOG_WARNING(
        "[G2G] Validation failed. Rolling back to the original graph.");
    ops[start_index].SwapOutputs(ops[start_index + 1]);
  }
  return 1;
}

size_t FuseMatMulConvertPrefill(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  // Connection check
  if (ops[start_index].GetOutputTensor(0) !=
      ops[start_index + 2].GetInputTensor(0)) {
    return 1;
  }
  // Graph transform
  QNN_LOG_INFO("[G2G] MatMul-convert fusion (Prefill)");
  ops[start_index].SwapOutputs(ops[start_index + 2]);
  if (validate_op_config(ops[start_index])) {
    ops.erase(ops.begin() + start_index + 2);
  } else {
    QNN_LOG_WARNING(
        "[G2G] Validation failed. Rolling back to the original graph.");
    ops[start_index].SwapOutputs(ops[start_index + 1]);
  }
  return 1;
}

size_t FuseConvertMatMul(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  // Connection check
  if (ops[start_index].GetOutputTensor(0) !=
      ops[start_index + 1].GetInputTensor(1)) {
    return 1;
  }
  // Graph transform
  QNN_LOG_INFO("[G2G] Convert-MatMul");
  auto& convert_output_0 = ops[start_index].GetOutputTensor(0);
  QNN_LOG_INFO("[G2G] Convert-MatMul 0");
  // Loop thru all output ops.
  const auto& matmul_op_names = convert_output_0.GetConsumerOpNames();
  // Find ops by name.
  std::vector<OpWrapper*> matmul_ops;
  for (auto& op : ops) {
    for (const auto& name : matmul_op_names) {
        // QNN_LOG_INFO("[G2G] Convert-MatMul target %s", name.c_str());
      if (absl::StrContains(op.GetName(), name)) {
        QNN_LOG_INFO("[G2G] Convert-MatMul found %s", name.c_str());
        matmul_ops.push_back(&op);
        break;
      }
    }
  }
  for (auto* matmul_op : matmul_ops) {
    matmul_op->AttachInput(ops[start_index].GetInputTensor(0), 1);
  }
  QNN_LOG_INFO("[G2G] Convert-MatMul 2");
  if (validate_op_config(ops[start_index])) {
    ops.erase(ops.begin() + start_index);
    QNN_LOG_INFO("[G2G] Convert-MatMul 3");
  } else {
    QNN_LOG_WARNING(
        "[G2G] Validation failed. Rolling back to the original graph.");
    // for (auto* matmul_op : matmul_ops) {
    //   ops[start_index].SwapInput(*matmul_op, 0, 1);
    // }
  }
  return 1;
}
}  // namespace qnn
