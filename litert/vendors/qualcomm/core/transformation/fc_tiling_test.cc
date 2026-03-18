#include <gtest/gtest.h>
#include <numeric>
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/builders/fully_connected_op_builder.h"
#include "litert/vendors/qualcomm/core/transformation/graph_to_graph.h"

namespace qnn {
namespace {

TEST(FCTest, FcTiling) {
  TensorPool tensor_pool;
  QuantizeParamsWrapperVariant quant_param;
  quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(1e-4f, 0);
  auto& input = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_8,
                                               quant_param, {1, 1, 1536});
  std::vector<int8_t> weight_data;
  size_t num_element = 12288 * 1536;
  weight_data.resize(num_element);
  std::iota(weight_data.begin(), weight_data.end(), 0);
  auto& weight = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, quant_param, {12288, 1536},
      weight_data.size() * sizeof(weight_data[0]), weight_data.data());
  auto& output = tensor_pool.CloneNativeTensorFrom(input, {1, 1, 12288});
  auto fc = BuildFullyConnectedOp(tensor_pool, {input, weight}, {output}, true,
                                  false);
  std::vector<OpWrapper> op_wrappers;
  std::move(fc.begin(), fc.end(), std::back_inserter(op_wrappers));
  const ::qnn::G2GConfig g2g_option = ::qnn::G2GConfig::kMatMulConvert;
  GraphToGraphTransform(g2g_option, op_wrappers, tensor_pool,
                        [](OpWrapper& op) { return true; });
  ASSERT_EQ(op_wrappers.size(), 5);
}
}  // namespace
}  // namespace qnn