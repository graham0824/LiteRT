# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.20)

# Tool display
add_library(litert_tool_display STATIC
    tool_display.cc
)

target_include_directories(litert_tool_display
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_tool_display
    PUBLIC
        litert_cc_api
        litert_cc_internal
        absl::strings
        absl::string_view
        absl::span
        absl::status
        absl::statusor
)

# LiteRT tensor_utils
add_library(litert_tensor_utils STATIC
    tensor_utils.cc
)

target_include_directories(litert_tensor_utils
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_tensor_utils
    PRIVATE
        absl::log
        absl::log_internal_message
        absl::span
        absl::string_view
        litert_cc_api
)


# LiteRT Dump
add_library(litert_dump STATIC
    dump.cc
)

target_include_directories(litert_dump
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_dump
    PUBLIC
        litert_core
        litert_core_model
        tensorflow-lite
        absl::strings
        absl::status
        absl::statusor
)

# Apply plugin
add_library(litert_apply_plugin STATIC
    apply_plugin.cc
)

target_include_directories(litert_apply_plugin
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_apply_plugin
    PUBLIC
        litert_core
        litert_core_model
        absl::strings
        absl::status
        absl::statusor
)

# Add flags
add_subdirectory(flags)

# Common libraries for tools that use TFLite
set(LITERT_TOOL_COMMON_LIBS
    litert_c_api
    litert_cc_api
    litert_cc_internal
    litert_cc_options
    litert_compiler_plugin
    litert_core
    litert_core_model
    litert_runtime
    litert_options_parser_registry
)

set(ABSEIL_TOOL_LIBS
    absl::check
    absl::die_if_null
    absl::flags
    absl::flags_parse
    absl::log
    absl::log_internal_check_op
    absl::log_internal_message
    absl::random_random
    absl::string_view
    absl::span
    absl::status
    absl::statusor
)

# TFLite's CMake file only exposes the benchmark as an executable target. Reuse
# that target's sources, minus its main(), so LiteRT can link its own benchmark
# wrapper against the same implementation.
if(NOT TARGET benchmark_model)
    message(FATAL_ERROR "TFLite benchmark_model target is required to build benchmark_litert_model")
endif()
get_target_property(LITERT_TFLITE_BENCHMARK_SRCS benchmark_model SOURCES)
list(FILTER LITERT_TFLITE_BENCHMARK_SRCS EXCLUDE REGEX "(^|/)benchmark_main\\.cc$")
if(NOT LITERT_TFLITE_BENCHMARK_SRCS)
    message(FATAL_ERROR "Unable to discover TFLite benchmark_model sources")
endif()
add_library(litert_tflite_benchmark_model STATIC
    ${LITERT_TFLITE_BENCHMARK_SRCS}
)

get_target_property(
    LITERT_TFLITE_BENCHMARK_COMPILE_OPTIONS
    benchmark_model
    COMPILE_OPTIONS
)
if(LITERT_TFLITE_BENCHMARK_COMPILE_OPTIONS)
    target_compile_options(litert_tflite_benchmark_model
        PRIVATE
            ${LITERT_TFLITE_BENCHMARK_COMPILE_OPTIONS}
    )
endif()

get_target_property(
    LITERT_TFLITE_BENCHMARK_COMPILE_DEFINITIONS
    benchmark_model
    COMPILE_DEFINITIONS
)
if(LITERT_TFLITE_BENCHMARK_COMPILE_DEFINITIONS)
    target_compile_definitions(litert_tflite_benchmark_model
        PRIVATE
            ${LITERT_TFLITE_BENCHMARK_COMPILE_DEFINITIONS}
    )
endif()

get_target_property(
    LITERT_TFLITE_BENCHMARK_INCLUDE_DIRS
    benchmark_model
    INCLUDE_DIRECTORIES
)
if(LITERT_TFLITE_BENCHMARK_INCLUDE_DIRS)
    target_include_directories(litert_tflite_benchmark_model
        PRIVATE
            ${LITERT_TFLITE_BENCHMARK_INCLUDE_DIRS}
    )
endif()

target_include_directories(litert_tflite_benchmark_model
    PUBLIC
        $<BUILD_INTERFACE:${TFLITE_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${TFLITE_SOURCE_DIR}>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}/third_party/xla>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}/third_party/xla/third_party/tsl>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
)

get_target_property(
    LITERT_TFLITE_BENCHMARK_LIBS
    benchmark_model
    LINK_LIBRARIES
)
if(LITERT_TFLITE_BENCHMARK_LIBS)
    target_link_libraries(litert_tflite_benchmark_model
        PUBLIC
            ${LITERT_TFLITE_BENCHMARK_LIBS}
    )
endif()

# LiteRT benchmark model library
add_library(litert_benchmark_litert_model STATIC
    benchmark_litert_model.cc
    ../core/util/perfetto_profiling.cc
)

target_include_directories(litert_benchmark_litert_model
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
        $<BUILD_INTERFACE:${TFLITE_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_benchmark_litert_model
    PUBLIC
        litert_tflite_benchmark_model
        ${LITERT_TOOL_COMMON_LIBS}
        tensorflow-lite
        benchmark_result_proto
        model_runtime_info_proto
        absl::flat_hash_map
        absl::strings
        absl::string_view
        absl::span
        absl::status
        absl::statusor
)

# benchmark_litert_model
add_executable(benchmark_litert_model
    benchmark_litert_model_main.cc
)

target_include_directories(benchmark_litert_model
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${TENSORFLOW_SOURCE_DIR}
        ${TFLITE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}
)

if(ANDROID)
    target_compile_definitions(benchmark_litert_model PRIVATE ABSL_FLAGS_STRIP_NAMES=0)
endif()

if(ANDROID)
    target_link_libraries(benchmark_litert_model
        PRIVATE
            litert_benchmark_litert_model
            litert_tflite_benchmark_model
            "-Wl,--whole-archive"
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            litert_tool_flags_qualcomm
            litert_tool_flags_samsung
            "-Wl,--no-whole-archive"
            ${LITERT_TOOL_COMMON_LIBS}
            tensorflow-lite
            ${ABSEIL_TOOL_LIBS}
            absl::flags_reflection
            absl::flat_hash_map
            absl::strings
            Threads::Threads
    )
elseif(APPLE)
    target_link_libraries(benchmark_litert_model
        PRIVATE
            litert_benchmark_litert_model
            litert_tflite_benchmark_model
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            litert_tool_flags_qualcomm
            litert_tool_flags_samsung
            ${LITERT_TOOL_COMMON_LIBS}
            tensorflow-lite
            ${ABSEIL_TOOL_LIBS}
            absl::flags_reflection
            absl::flat_hash_map
            absl::strings
            Threads::Threads
    )
else()
    target_link_libraries(benchmark_litert_model
        PRIVATE
            "-Wl,--whole-archive"
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            litert_tool_flags_qualcomm
            litert_tool_flags_samsung
            "-Wl,--no-whole-archive"
            Threads::Threads
            tensorflow-lite
            "-Wl,--start-group"
            litert_benchmark_litert_model
            litert_tflite_benchmark_model
            ${ABSEIL_TOOL_LIBS}
            ${LITERT_TOOL_COMMON_LIBS}
            absl::flags_reflection
            absl::flat_hash_map
            absl::strings
            "-Wl,--end-group"
            ${CMAKE_DL_LIBS}
    )
endif()

# NPU numerics check
add_executable(litert_npu_numerics_check
    npu_numerics_check.cc
)

target_include_directories(litert_npu_numerics_check
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

if(ANDROID)
    target_link_libraries(litert_npu_numerics_check
        PRIVATE
            litert_tool_display
            "-Wl,--whole-archive"
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            litert_tool_flags_qualcomm
            litert_tool_flags_samsung
            "-Wl,--no-whole-archive"
            ${LITERT_TOOL_COMMON_LIBS}
            tensorflow-lite
            ${ABSEIL_TOOL_LIBS}
            Threads::Threads
            litert_tensor_utils
    )
elseif(APPLE)
    target_link_libraries(litert_npu_numerics_check
        PRIVATE
            litert_tool_display
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            litert_tool_flags_qualcomm
            litert_tool_flags_samsung
            ${LITERT_TOOL_COMMON_LIBS}
            tensorflow-lite
            ${ABSEIL_TOOL_LIBS}
            Threads::Threads
            litert_tensor_utils
    )
else()
    target_link_libraries(litert_npu_numerics_check
        PRIVATE
            litert_tool_display
            "-Wl,--whole-archive"
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            litert_tool_flags_qualcomm
            litert_tool_flags_samsung
            "-Wl,--no-whole-archive"
            Threads::Threads
            tensorflow-lite
            "-Wl,--start-group"
            ${ABSEIL_TOOL_LIBS}
            ${LITERT_TOOL_COMMON_LIBS}
            "-Wl,--end-group"
            ${CMAKE_DL_LIBS}
            litert_tensor_utils
    )
endif()

# run_model
add_executable(run_model
    run_model.cc
)

target_include_directories(run_model
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${TENSORFLOW_SOURCE_DIR}
)

# absl strips flag names/registration on Android by default (ABSL_FLAGS_STRIP_NAMES=1).
# This tool relies on absl::ParseCommandLine, so we must override that behaviour.
if(ANDROID)
    target_compile_definitions(run_model PRIVATE ABSL_FLAGS_STRIP_NAMES=0)
endif()

# Platform-aware linking
if(ANDROID)
    target_link_libraries(run_model
        PRIVATE
            litert_tool_display
            "-Wl,--whole-archive"
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            litert_tool_flags_qualcomm
            litert_tool_flags_samsung
            "-Wl,--no-whole-archive"
            ${LITERT_TOOL_COMMON_LIBS}
            tensorflow-lite
            ${ABSEIL_TOOL_LIBS}
            Threads::Threads
            litert_tensor_utils
    )
elseif(APPLE)
    target_link_libraries(run_model
        PRIVATE
            litert_tool_display
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            litert_tool_flags_qualcomm
            litert_tool_flags_samsung
            ${LITERT_TOOL_COMMON_LIBS}
            tensorflow-lite
            ${ABSEIL_TOOL_LIBS}
            Threads::Threads
            litert_tensor_utils
    )
else()
    target_link_libraries(run_model
        PRIVATE
            litert_tool_display
            "-Wl,--whole-archive"
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            litert_tool_flags_qualcomm
            litert_tool_flags_samsung
            "-Wl,--no-whole-archive"
            Threads::Threads
            tensorflow-lite
            "-Wl,--start-group"
            ${ABSEIL_TOOL_LIBS}
            ${LITERT_TOOL_COMMON_LIBS}
            "-Wl,--end-group"
            ${CMAKE_DL_LIBS}
            litert_tensor_utils
    )
endif()

# analyze_model
add_executable(analyze_model
    analyze_model_main.cc
)

target_include_directories(analyze_model
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${TENSORFLOW_SOURCE_DIR}
)

# Platform-aware linking for analyze_model
if(ANDROID OR APPLE)
    target_link_libraries(analyze_model
        PRIVATE
            litert_dump
            litert_tool_display
            litert_c_api
            litert_core
            litert_core_model
            tensorflow-lite
            ${ABSEIL_TOOL_LIBS}
            Threads::Threads
    )
else()
    target_link_libraries(analyze_model
        PRIVATE
            Threads::Threads
            tensorflow-lite
            "-Wl,--start-group"
            ${ABSEIL_TOOL_LIBS}
            litert_dump
            litert_tool_display
            litert_c_api
            litert_core
            litert_core_model
            "-Wl,--end-group"
            ${CMAKE_DL_LIBS}
    )
endif()

# apply_plugin
add_executable(apply_plugin_main
    apply_plugin_main.cc
)

target_include_directories(apply_plugin_main
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${TENSORFLOW_SOURCE_DIR}
)

# Platform-aware linking
if(ANDROID)
    target_link_libraries(apply_plugin_main
        PRIVATE
            litert_tool_display
            litert_apply_plugin
            litert_dump
            "-Wl,--whole-archive"
            litert_tool_flags_apply_plugin
            litert_tool_flags_common
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            litert_tool_flags_qualcomm
            litert_tool_flags_samsung
            "-Wl,--no-whole-archive"
            ${LITERT_TOOL_COMMON_LIBS}
            tensorflow-lite
            ${ABSEIL_TOOL_LIBS}
            Threads::Threads
        )
elseif(APPLE)
    target_link_libraries(apply_plugin_main
        PRIVATE
            litert_tool_display
            litert_apply_plugin
            litert_dump
            litert_tool_flags_apply_plugin
            litert_tool_flags_common
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            litert_tool_flags_qualcomm
            litert_tool_flags_samsung
            ${LITERT_TOOL_COMMON_LIBS}
            tensorflow-lite
            ${ABSEIL_TOOL_LIBS}
            Threads::Threads
        )
else()
    target_link_libraries(apply_plugin_main
        PRIVATE
            "-Wl,--whole-archive"
            litert_tool_flags_apply_plugin
            litert_tool_flags_common
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            litert_tool_flags_qualcomm
            litert_tool_flags_samsung
            "-Wl,--no-whole-archive"
            Threads::Threads
            tensorflow-lite
            "-Wl,--start-group"
            litert_tool_display
            litert_apply_plugin
            litert_dump
            ${ABSEIL_TOOL_LIBS}
            ${LITERT_TOOL_COMMON_LIBS}
            "-Wl,--end-group"
            ${CMAKE_DL_LIBS}
    )
endif()

# extract_bytecode
add_executable(extract_bytecode
    extract_bytecode.cc
)

target_include_directories(extract_bytecode
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${TENSORFLOW_SOURCE_DIR}
)

if(ANDROID OR APPLE)
    target_link_libraries(extract_bytecode
        PRIVATE
            litert_tool_display
            litert_cc_internal
            litert_core
            litert_core_model
            ${ABSEIL_TOOL_LIBS}
            Threads::Threads
    )
else()
    target_link_libraries(extract_bytecode
        PRIVATE
            Threads::Threads
            "-Wl,--start-group"
            litert_tool_display
            ${LITERT_TOOL_COMMON_LIBS}
            ${ABSEIL_TOOL_LIBS}
            "-Wl,--end-group"
            ${CMAKE_DL_LIBS}
    )
endif()

# npu_numerics_check
add_executable(npu_numerics_check
    npu_numerics_check.cc
)

target_include_directories(npu_numerics_check
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${TENSORFLOW_SOURCE_DIR}
)

# Platform-aware linking
if(ANDROID OR APPLE)
    target_link_libraries(npu_numerics_check
        PRIVATE
            litert_tool_flags_qualcomm
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            ${LITERT_TOOL_COMMON_LIBS}
            tensorflow-lite
            ${ABSEIL_TOOL_LIBS}
            Threads::Threads
            litert_tensor_utils
    )
else()
    target_link_libraries(npu_numerics_check
        PRIVATE
            litert_tool_flags_qualcomm
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            Threads::Threads
            tensorflow-lite
            "-Wl,--start-group"
            ${ABSEIL_TOOL_LIBS}
            ${LITERT_TOOL_COMMON_LIBS}
            "-Wl,--end-group"
            ${CMAKE_DL_LIBS}
            litert_tensor_utils
    )
endif()

# Platform-specific configurations for all executables
if(LITERT_PLATFORM_ANDROID)
    target_link_libraries(analyze_model PRIVATE ${LOG_LIB})
    target_link_libraries(apply_plugin_main PRIVATE ${LOG_LIB})
    target_link_libraries(benchmark_litert_model PRIVATE ${LOG_LIB})
    target_link_libraries(extract_bytecode PRIVATE ${LOG_LIB})
    target_link_libraries(run_model PRIVATE ${LOG_LIB})
    # Statically link TFLite GPU/GL core objects for tool executables on Android.
    if(TARGET litert_tflite_gpu_gl_core)
        target_link_libraries(analyze_model PRIVATE litert_tflite_gpu_gl_core)
        target_link_libraries(apply_plugin_main PRIVATE litert_tflite_gpu_gl_core)
        target_link_libraries(benchmark_litert_model PRIVATE litert_tflite_gpu_gl_core)
        target_link_libraries(extract_bytecode PRIVATE litert_tflite_gpu_gl_core)
        target_link_libraries(litert_npu_numerics_check PRIVATE litert_tflite_gpu_gl_core)
        target_link_libraries(npu_numerics_check PRIVATE litert_tflite_gpu_gl_core)
        target_link_libraries(run_model PRIVATE litert_tflite_gpu_gl_core)
    endif()
endif()

# Install targets
install(TARGETS run_model analyze_model apply_plugin_main benchmark_litert_model extract_bytecode
    RUNTIME DESTINATION bin
)
