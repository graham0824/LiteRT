// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"

#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

std::vector<OpWrapper> BuildElementwiseAddOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_ADD);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseSubOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_SUBTRACT);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseMulOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_MULTIPLY);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseDivOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_DIVIDE);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseSinOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_SIN);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseCeilOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_UNARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_CEIL);

  return res;
}

std::vector<OpWrapper> BuildElementwiseCosOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_COS);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseHardSwishOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  OpWrapper& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_NEURON);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SWISH);

  return res;
}

std::vector<OpWrapper> BuildElementwiseRsqrtOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_RSQRT);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseSqrtOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_SQUARE_ROOT);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseSquareOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  OpWrapper& elementwise_op =
      CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_MULTIPLY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseSquaredDifferenceOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op =
      CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_SQUARED_DIFFERENCE);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseLessOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS);

  return res;
}

std::vector<OpWrapper> BuildElementwiseGreaterOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER);

  return res;
}

std::vector<OpWrapper> BuildElementwiseAndOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_AND);

  return res;
}

std::vector<OpWrapper> BuildElementwiseMinimumOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MINIMUM);

  return res;
}

std::vector<OpWrapper> BuildElementwiseMaximumOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MAXIMUM);

  return res;
}

std::vector<OpWrapper> BuildElementwiseEluOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  OpWrapper& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_NEURON);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_NEURON_OPERATION_ELU);

  return res;
}

std::vector<OpWrapper> BuildElementwiseFloorOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_UNARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_FLOOR);

  return res;
}

std::vector<OpWrapper> BuildElementwiseFloorDivOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddInputTensor(inputs[1]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_FLOOR_DIV);

  return res;
}

std::vector<OpWrapper> BuildElementwiseNotEqualOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddInputTensor(inputs[1]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_NOT_EQUAL);

  return res;
}

std::vector<OpWrapper> BuildElementwiseOrOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_OR);

  return res;
}

std::vector<OpWrapper> BuildElementwisePowerOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_POWER);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  return res;
}

std::vector<OpWrapper> BuildElementwiseLessEqualOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddInputTensor(inputs[1]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS_EQUAL);

  return res;
}

std::vector<OpWrapper> BuildElementwiseNotOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_NOT);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  return res;
}

std::vector<OpWrapper> BuildElementwiseGreaterEqualOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddInputTensor(inputs[1]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER_EQUAL);

  return res;
}

std::vector<OpWrapper> BuildElementwiseExpOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_UNARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_EXP);

  return res;
}

std::vector<OpWrapper> BuildElementwiseEqualOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddInputTensor(inputs[1]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_EQUAL);

  return res;
}

std::vector<OpWrapper> BuildElementwiseLogOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_UNARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_LOG);

  return res;
}

std::vector<OpWrapper> BuildElementwiseAbsOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_UNARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ABS);

  return res;
}

std::vector<OpWrapper> BuildElementwiseNegOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_UNARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NEG);

  return res;
}

std::vector<OpWrapper> BuildElementwiseRoundOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_UNARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ROUND);

  return res;
}

std::vector<OpWrapper> BuildElementwiseSignOp(
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_UNARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIGN);

  return res;
}

}  // namespace qnn
