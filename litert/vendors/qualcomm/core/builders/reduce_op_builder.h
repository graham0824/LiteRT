// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_REDUCE_OP_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_REDUCE_OP_BUILDER_H_

#include <vector>

#include "litert/vendors/qualcomm/core/ir_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

std::vector<OpWrapper> BuildReduceSumOp(
    IrPool<TensorWrapper>& tensor_pool,
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, bool keep_dims);

std::vector<OpWrapper> BuildReduceMeanOp(
    IrPool<TensorWrapper>& tensor_pool,
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, bool keep_dims);

std::vector<OpWrapper> BuildReduceMaxOp(
    IrPool<TensorWrapper>& tensor_pool,
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, bool keep_dims);

std::vector<OpWrapper> BuildReduceMinOp(
    IrPool<TensorWrapper>& tensor_pool,
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, bool keep_dims);

std::vector<OpWrapper> BuildReduceAnyOp(
    IrPool<TensorWrapper>& tensor_pool,
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, bool keep_dims);

std::vector<OpWrapper> BuildReduceAllOp(
    IrPool<TensorWrapper>& tensor_pool,
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, bool keep_dims);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_REDUCE_OP_BUILDER_H_
