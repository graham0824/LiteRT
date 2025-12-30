// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/transpose_op_builder.h"

#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/ir_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace qnn {

std::vector<OpWrapper> BuildTransposeOp(
    IrPool<TensorWrapper>& tensor_pool,
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  TensorWrapper& perm_tensor = inputs[1];
  if (!perm_tensor.IsTensorStatic()) {
    QNN_LOG_ERROR("The param 'perm' of Transpose OP is not static.");
    return res;
  }

  TensorWrapper& cloned_perm_tensor = tensor_pool.Emplace();
  CloneStaticTensorFrom(cloned_perm_tensor, "", perm_tensor,
                        QNN_DATATYPE_UINT_32);

  auto& transpose_op = CreateOpWrapper(res, QNN_OP_TRANSPOSE);
  transpose_op.AddInputTensor(inputs[0]);
  transpose_op.AddOutputTensor(outputs[0]);
  transpose_op.AddTensorParam(QNN_OP_TRANSPOSE_PARAM_PERM, cloned_perm_tensor);

  return res;
}

}  // namespace qnn
