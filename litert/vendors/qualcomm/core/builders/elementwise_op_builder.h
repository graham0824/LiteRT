// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_ELEMENTWISE_OP_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_ELEMENTWISE_OP_BUILDER_H_

#include <vector>

#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

std::vector<OpWrapper> BuildElementwiseAddOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseSubOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseMulOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseDivOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseSinOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseCeilOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseCosOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseHardSwishOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseRsqrtOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseSqrtOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseSquareOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseSquaredDifferenceOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseLessOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseGreaterOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseAndOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseMinimumOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseMaximumOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseEluOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseFloorOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseFloorDivOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseNotEqualOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseOrOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwisePowerOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseLessEqualOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseNotOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseGreaterEqualOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseExpOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseEqualOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseLogOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseAbsOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseNegOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseRoundOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildElementwiseSignOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_ELEMENTWISE_OP_BUILDER_H_
