// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_EMBEDDING_LOOKUP_OP_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_EMBEDDING_LOOKUP_OP_BUILDER_H_

#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

OpWrapper CreateGatherOp(const TensorWrapper& table,
                         const TensorWrapper& indices,
                         const TensorWrapper& output, std::int32_t axis);

std::vector<OpWrapper> BuildEmbeddingLookupOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

// Handles the fp32-activation / int2-per-channel-weight variant of
// EmbeddingLookup by emitting a QNN custom op for the HVX op package.
// The weight is converted to self-packed uint8 with tile permutation at
// compile time; per-channel scales become an explicit fp32 input tensor.
//
// Debug / accuracy reference: when compiled with -DDQ_FC, this instead
// dequantizes the per-channel int2 weight into a plain fp32 embedding table at
// compile time (S * (q - Z) per channel) and emits a standard Gather. That path
// requires no HVX custom op package and lets the custom kernel's numerics be
// checked against a known-good float Gather. The two paths cannot coexist in one
// binary.
std::vector<OpWrapper> BuildEmbeddingLookupFpa2wOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs,
    const CustomOpPackage& custom_op_package);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_EMBEDDING_LOOKUP_OP_BUILDER_H_
