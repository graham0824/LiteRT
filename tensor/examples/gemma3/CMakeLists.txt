# Copyright 2026 Google LLC.
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

include(FetchContent)

function(_litert_gemma3_prepare_sentencepiece)
    if(TARGET litert_sentencepiece_processor)
        set(_litert_gemma3_sentencepiece_available TRUE PARENT_SCOPE)
        return()
    endif()

    if(COMMAND LiteRtDepsCache_SourceDirArgs)
        LiteRtDepsCache_SourceDirArgs(litert_sentencepiece _litert_sentencepiece_cache_args)
    else()
        set(_litert_sentencepiece_cache_args "")
    endif()
    FetchContent_Declare(
        litert_sentencepiece
        URL https://github.com/google/sentencepiece/archive/refs/tags/v0.2.0.tar.gz
        URL_HASH SHA256=9970f0a0afee1648890293321665e5b2efa04eaec9f1671fcf8048f456f5bb86
        ${_litert_sentencepiece_cache_args}
    )

    FetchContent_GetProperties(litert_sentencepiece)
    if(NOT litert_sentencepiece_POPULATED)
        FetchContent_Populate(litert_sentencepiece)
    endif()

    set(_spm_src "${litert_sentencepiece_SOURCE_DIR}/src")
    file(WRITE "${_spm_src}/config.h" "")

    file(GLOB _spm_patch_files
        "${_spm_src}/*.cc"
        "${_spm_src}/*.h"
    )
    foreach(_spm_patch_file IN LISTS _spm_patch_files)
        file(READ "${_spm_patch_file}" _spm_contents)
        string(REPLACE "#include \"third_party/absl/"
                       "#include \"absl/"
                       _spm_contents "${_spm_contents}")
        string(REPLACE "#include \"third_party/darts_clone/"
                       "#include \""
                       _spm_contents "${_spm_contents}")
        string(REPLACE "#include \"include/darts.h\""
                       "#include \"darts.h\""
                       _spm_contents "${_spm_contents}")
        file(WRITE "${_spm_patch_file}" "${_spm_contents}")
    endforeach()

    find_package(Protobuf REQUIRED)
    set(_spm_proto_srcs
        "${CMAKE_CURRENT_BINARY_DIR}/sentencepiece.pb.cc"
        "${CMAKE_CURRENT_BINARY_DIR}/sentencepiece_model.pb.cc"
    )
    set(_spm_proto_hdrs
        "${CMAKE_CURRENT_BINARY_DIR}/sentencepiece.pb.h"
        "${CMAKE_CURRENT_BINARY_DIR}/sentencepiece_model.pb.h"
    )

    set(LITERT_HOST_PROTOC "" CACHE FILEPATH
        "Host protoc executable used when cross-compiling Gemma3")
    set(_spm_protoc "")
    set(_spm_protoc_dep "")

    if(CMAKE_CROSSCOMPILING)
        if(LITERT_HOST_PROTOC)
            set(_spm_protoc "${LITERT_HOST_PROTOC}")
        else()
            find_program(_spm_protoc NAMES protoc)
        endif()
        if(_spm_protoc)
            message(STATUS "Using host protoc for Gemma3 SentencePiece: ${_spm_protoc}")
        else()
            message(WARNING
                "Skipping Gemma3 litert_main: Android/CMake cross-compilation "
                "requires a host protoc. Install protoc or pass "
                "-DLITERT_HOST_PROTOC=/path/to/protoc.")
            set(_litert_gemma3_sentencepiece_available FALSE PARENT_SCOPE)
            return()
        endif()
    endif()

    if(NOT _spm_protoc AND TARGET protobuf::protoc)
        set(_spm_protoc "$<TARGET_FILE:protobuf::protoc>")
        set(_spm_protoc_dep protobuf::protoc)
    elseif(NOT _spm_protoc AND TARGET protoc)
        set(_spm_protoc "$<TARGET_FILE:protoc>")
        set(_spm_protoc_dep protoc)
    elseif(NOT _spm_protoc AND Protobuf_PROTOC_EXECUTABLE)
        set(_spm_protoc "${Protobuf_PROTOC_EXECUTABLE}")
        set(_spm_protoc_dep "")
    elseif(NOT _spm_protoc)
        message(FATAL_ERROR "Could not find a protoc executable for SentencePiece.")
    endif()

    add_custom_command(
        OUTPUT ${_spm_proto_srcs} ${_spm_proto_hdrs}
        COMMAND ${_spm_protoc}
            --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
            -I ${_spm_src}
            "${_spm_src}/sentencepiece.proto"
            "${_spm_src}/sentencepiece_model.proto"
        DEPENDS
            ${_spm_protoc_dep}
            "${_spm_src}/sentencepiece.proto"
            "${_spm_src}/sentencepiece_model.proto"
        VERBATIM
    )

    add_library(litert_sentencepiece_processor STATIC
        "${_spm_src}/bpe_model.cc"
        "${_spm_src}/char_model.cc"
        "${_spm_src}/error.cc"
        "${_spm_src}/filesystem.cc"
        "${_spm_src}/model_factory.cc"
        "${_spm_src}/model_interface.cc"
        "${_spm_src}/normalizer.cc"
        "${_spm_src}/sentencepiece_processor.cc"
        "${_spm_src}/unigram_model.cc"
        "${_spm_src}/util.cc"
        "${_spm_src}/word_model.cc"
        ${_spm_proto_srcs}
        ${_spm_proto_hdrs}
    )

    target_include_directories(litert_sentencepiece_processor
        PUBLIC
            "${_spm_src}"
        PRIVATE
            "${CMAKE_CURRENT_BINARY_DIR}"
            "${litert_sentencepiece_SOURCE_DIR}/third_party/darts_clone"
    )

    target_compile_definitions(litert_sentencepiece_processor
        PUBLIC
            _USE_EXTERNAL_ABSL
            _USE_EXTERNAL_PROTOBUF
        PRIVATE
            ENABLE_NFKC_COMPILE
            SENTENCEPIECE_PG3_BUILD
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(litert_sentencepiece_processor
            PRIVATE
                -Wno-deprecated-declarations
                -Wno-sign-compare
                -Wno-unused-parameter)
    endif()

    target_link_libraries(litert_sentencepiece_processor
        PUBLIC
            absl::status
            absl::strings
        PRIVATE
            absl::check
            absl::flags
            absl::flags_parse
            absl::log
            protobuf::libprotobuf
            Threads::Threads
    )
    set(_litert_gemma3_sentencepiece_available TRUE PARENT_SCOPE)
endfunction()

function(_litert_gemma3_copy_gpu_accelerator target_name)
    if(NOT LITERT_ENABLE_GPU)
        return()
    endif()

    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _litert_gemma3_processor)
    set(_litert_gemma3_gpu_accelerator_relpath "")

    if(APPLE AND _litert_gemma3_processor MATCHES "^(arm64|aarch64)$")
        set(_litert_gemma3_gpu_accelerator_relpath
            "macos_arm64/libLiteRtMetalAccelerator.dylib")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND _litert_gemma3_processor MATCHES "^(x86_64|amd64)$")
        set(_litert_gemma3_gpu_accelerator_relpath
            "linux_x86_64/libLiteRtWebGpuAccelerator.so")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND _litert_gemma3_processor MATCHES "^(aarch64|arm64)$")
        set(_litert_gemma3_gpu_accelerator_relpath
            "linux_arm64/libLiteRtWebGpuAccelerator.so")
    elseif(WIN32)
        set(_litert_gemma3_gpu_accelerator_relpath
            "windows_x86_64/libLiteRtWebGpuAccelerator.dll")
    elseif(ANDROID AND _litert_gemma3_processor MATCHES "^(aarch64|arm64)$")
        set(_litert_gemma3_gpu_accelerator_relpath
            "android_arm64/libLiteRtClGlAccelerator.so")
    endif()

    if(NOT _litert_gemma3_gpu_accelerator_relpath)
        message(WARNING
            "LITERT_ENABLE_GPU=ON, but Gemma3 litert_main does not know which "
            "prebuilt GPU accelerator to download for this platform.")
        return()
    endif()

    set(_litert_gemma3_gpu_accelerator_url
        "https://storage.googleapis.com/litert/binaries/latest/${_litert_gemma3_gpu_accelerator_relpath}")
    get_filename_component(_litert_gemma3_gpu_accelerator_name
        "${_litert_gemma3_gpu_accelerator_relpath}" NAME)

    # Use FetchContent so the prebuilt accelerator is cached under
    # FETCHCONTENT_BASE_DIR and only re-downloaded when the URL changes.
    # DOWNLOAD_NO_EXTRACT because this is a single shared library, not an
    # archive (the default extract step would otherwise fail).
    include(FetchContent)
    FetchContent_Declare(litert_gemma3_gpu_accelerator
        URL "${_litert_gemma3_gpu_accelerator_url}"
        DOWNLOAD_NO_EXTRACT TRUE
        TLS_VERIFY ON)
    message(STATUS
        "Downloading LiteRT GPU accelerator from ${_litert_gemma3_gpu_accelerator_url}")
    FetchContent_GetProperties(litert_gemma3_gpu_accelerator)
    if(NOT litert_gemma3_gpu_accelerator_POPULATED)
        FetchContent_Populate(litert_gemma3_gpu_accelerator)
    endif()
    set(_litert_gemma3_gpu_accelerator
        "${litert_gemma3_gpu_accelerator_SOURCE_DIR}/${_litert_gemma3_gpu_accelerator_name}")

    add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_litert_gemma3_gpu_accelerator}"
            "$<TARGET_FILE_DIR:${target_name}>/${_litert_gemma3_gpu_accelerator_name}"
        COMMENT "Copying LiteRT GPU accelerator next to ${target_name}")
endfunction()

_litert_gemma3_prepare_sentencepiece()
if(NOT _litert_gemma3_sentencepiece_available)
    return()
endif()

add_executable(litert_main EXCLUDE_FROM_ALL
    graph_helpers.cc
    litert_main.cc
    safetensor_loader.cc
    tokenizer.cc
)

target_compile_definitions(litert_main
    PRIVATE
        GOOGLE_COMMANDLINEFLAGS_FULL_API=1
)
if(ANDROID)
    target_compile_definitions(litert_main PRIVATE ABSL_FLAGS_STRIP_NAMES=0)
endif()

target_include_directories(litert_main
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../.."
)

target_link_libraries(litert_main
    PRIVATE
        litert_sentencepiece_processor
        "-Wl,--start-group"
        litert_cc_api
        litert_runtime_c_api_static
        litert_runtime
        litert_c_api
        litert_tensor_api_core
        litert_tensor_api_tflite_backend
        "-Wl,--end-group"
        absl::algorithm_container
        absl::flat_hash_map
        absl::flags
        absl::flags_parse
        absl::log
        absl::status
        absl::statusor
        absl::str_format
        absl::strings
        absl::string_view
        absl::synchronization
        absl::time
        absl::span
)

set_target_properties(litert_main PROPERTIES
    BUILD_RPATH "$<TARGET_FILE_DIR:litert_main>"
)

_litert_gemma3_copy_gpu_accelerator(litert_main)
