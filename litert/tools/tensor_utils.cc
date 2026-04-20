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

#include "litert/tools/tensor_utils.h"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"

namespace litert {
namespace tensor_utils {

Expected<void> FillInputBuffersWithCustomData(
    const CompiledModel& compiled_model, size_t signature_index,
    std::vector<TensorBuffer>& input_buffers, absl::string_view input_dir) {
  ABSL_LOG(INFO) << "Using inputs from: " << input_dir;
  LITERT_ASSIGN_OR_RETURN(
      const auto input_names,
      compiled_model.GetSignatureInputNames(signature_index));
  for (size_t i = 0; i < input_names.size(); ++i) {
    const auto& input_name = input_names[i];
    auto& input_buffer = input_buffers[i];
    const auto input_file_path = std::filesystem::path(std::string(input_dir)) /
                                 (std::string(input_name.data()) + ".raw");
    LITERT_ASSIGN_OR_RETURN(auto data, tensor_utils::ReadTensorDataFromRawFile(
                                           input_file_path.string()));
    LITERT_RETURN_IF_ERROR(
        tensor_utils::FillBufferWithCustomData(input_buffer, data));
  }
  return {};
}

Expected<void> WriteOutputBuffersToRawFiles(
    const CompiledModel& compiled_model, size_t signature_index,
    std::vector<TensorBuffer>& output_buffers, absl::string_view output_dir) {
  ABSL_LOG(INFO) << "Writing outputs to: " << output_dir;
  std::filesystem::create_directories(std::string(output_dir));
  LITERT_ASSIGN_OR_RETURN(
      const auto output_names,
      compiled_model.GetSignatureOutputNames(signature_index));
  for (size_t i = 0; i < output_names.size(); ++i) {
    const auto& output_name = output_names[i];
    auto& output_buffer = output_buffers[i];
    LITERT_ASSIGN_OR_RETURN(auto size, output_buffer.PackedSize());
    std::vector<uint8_t> data(size);
    LITERT_RETURN_IF_ERROR(output_buffer.Read<uint8_t>(absl::MakeSpan(data)));
    const auto output_file_path =
        std::filesystem::path(std::string(output_dir)) /
        (std::string(output_name.data()) + ".raw");
    LITERT_RETURN_IF_ERROR(
        tensor_utils::WriteTensorDataToRawFile(output_file_path.string(), data));
    ABSL_LOG(INFO) << "Wrote output tensor \"" << output_name << "\" ("
                   << size << " bytes) to " << output_file_path;
  }
  return {};
}

}  // namespace tensor_utils
}  // namespace litert
