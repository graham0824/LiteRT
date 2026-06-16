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

if(COMMAND LiteRtDepsCache_SourceDirArgs)
    LiteRtDepsCache_SourceDirArgs(stb _stb_cache_args)
else()
    set(_stb_cache_args "")
endif()
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG c0c982601f40183e74d84a61237e968dca08380e
    ${_stb_cache_args}
)

FetchContent_GetProperties(stb)
if(NOT stb_POPULATED)
    FetchContent_Populate(stb)
endif()

get_filename_component(LITERT_REPO_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

add_library(litert_tensor_segmentation_image_utils STATIC
    image_utils.cc
)

target_include_directories(litert_tensor_segmentation_image_utils
    PUBLIC
        ${LITERT_REPO_ROOT}
    PRIVATE
        ${stb_SOURCE_DIR}
)

target_link_libraries(litert_tensor_segmentation_image_utils
    PUBLIC
        absl::algorithm_container
)

add_executable(litert_tensor_segmentation_example
    segmentation_example.cc
)

target_link_libraries(litert_tensor_segmentation_example
    PRIVATE
        litert_tensor_segmentation_image_utils
        "-Wl,--start-group"
        litert_cc_api
        litert_runtime_c_api_static
        litert_runtime
        litert_c_api
        litert_tensor_api_core
        litert_tensor_api_tflite_backend
        "-Wl,--end-group"
        absl::check
        absl::flat_hash_map
        absl::log
        absl::span
)

set_target_properties(litert_tensor_segmentation_example PROPERTIES
    BUILD_RPATH "$<TARGET_FILE_DIR:litert_tensor_segmentation_example>"
)

if(LITERT_ENABLE_GPU)
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _litert_segmentation_processor)
    set(_litert_segmentation_gpu_accelerator_relpath "")

    if(APPLE AND _litert_segmentation_processor MATCHES "^(arm64|aarch64)$")
        set(_litert_segmentation_gpu_accelerator_relpath
            "macos_arm64/libLiteRtMetalAccelerator.dylib")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND _litert_segmentation_processor MATCHES "^(x86_64|amd64)$")
        set(_litert_segmentation_gpu_accelerator_relpath
            "linux_x86_64/libLiteRtWebGpuAccelerator.so")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND _litert_segmentation_processor MATCHES "^(aarch64|arm64)$")
        set(_litert_segmentation_gpu_accelerator_relpath
            "linux_arm64/libLiteRtWebGpuAccelerator.so")
    elseif(WIN32)
        set(_litert_segmentation_gpu_accelerator_relpath
            "windows_x86_64/libLiteRtWebGpuAccelerator.dll")
    elseif(ANDROID AND _litert_segmentation_processor MATCHES "^(aarch64|arm64)$")
        set(_litert_segmentation_gpu_accelerator_relpath
            "android_arm64/libLiteRtClGlAccelerator.so")
    endif()

    if(_litert_segmentation_gpu_accelerator_relpath)
        set(_litert_segmentation_gpu_accelerator_url
            "https://storage.googleapis.com/litert/binaries/latest/${_litert_segmentation_gpu_accelerator_relpath}")
        get_filename_component(_litert_segmentation_gpu_accelerator_name
            "${_litert_segmentation_gpu_accelerator_relpath}" NAME)

        # Use FetchContent so the prebuilt accelerator is cached under
        # FETCHCONTENT_BASE_DIR and only re-downloaded when the URL changes.
        # DOWNLOAD_NO_EXTRACT because this is a single shared library, not an
        # archive (the default extract step would otherwise fail).
        include(FetchContent)
        FetchContent_Declare(litert_segmentation_gpu_accelerator
            URL "${_litert_segmentation_gpu_accelerator_url}"
            DOWNLOAD_NO_EXTRACT TRUE
            TLS_VERIFY ON)
        message(STATUS
            "Downloading LiteRT GPU accelerator from ${_litert_segmentation_gpu_accelerator_url}")
        FetchContent_GetProperties(litert_segmentation_gpu_accelerator)
        if(NOT litert_segmentation_gpu_accelerator_POPULATED)
            FetchContent_Populate(litert_segmentation_gpu_accelerator)
        endif()
        set(_litert_segmentation_gpu_accelerator
            "${litert_segmentation_gpu_accelerator_SOURCE_DIR}/${_litert_segmentation_gpu_accelerator_name}")

        add_custom_command(TARGET litert_tensor_segmentation_example POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_litert_segmentation_gpu_accelerator}"
                "$<TARGET_FILE_DIR:litert_tensor_segmentation_example>/${_litert_segmentation_gpu_accelerator_name}"
            COMMENT "Copying LiteRT GPU accelerator next to segmentation example")
    else()
        message(WARNING
            "LITERT_ENABLE_GPU=ON, but the segmentation example does not know "
            "which prebuilt GPU accelerator to download for this platform.")
    endif()
endif()
