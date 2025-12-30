// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_QUANTIZE_OP_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_QUANTIZE_OP_BUILDER_H_

#include <vector>

#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

std::vector<OpWrapper> BuildQuantizeOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildDequantizeOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_QUANTIZE_OP_BUILDER_H_
