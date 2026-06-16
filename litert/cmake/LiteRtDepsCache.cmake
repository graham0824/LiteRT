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

# Helper that routes FetchContent dependency *sources* into a persistent,
# shared cache directory (LITERT_DEPS_CACHE_DIR) while leaving the disposable
# <name>-build / <name>-subbuild trees under FETCHCONTENT_BASE_DIR (the build
# dir). The goal is that a fresh build directory can reuse sources that were
# already downloaded by a previous build instead of re-cloning / re-extracting
# them.
#
# Two FetchContent mechanisms are combined:
#   * SOURCE_DIR <path>            -- on the first fetch, the populated source is
#                                     written into the cache instead of the
#                                     build tree.
#   * FETCHCONTENT_SOURCE_DIR_<N>  -- once the cached source exists, this tells
#                                     FetchContent the source is already present
#                                     so the download/clone step is skipped
#                                     entirely (no -subbuild work, no re-clone).
#
# The stamp files that record "already populated" live under
# FETCHCONTENT_BASE_DIR/<name>-subbuild, i.e. in the build dir. A fresh build
# dir therefore has no stamps and FetchContent would normally re-fetch, wiping
# the cached source. FETCHCONTENT_SOURCE_DIR_<N> avoids that: when the cached
# source is present we point FetchContent straight at it.

# Compute the extra FetchContent_Declare() arguments (if any) needed to cache
# the source for CONTENT_NAME, and, when the cached source already exists, set
# the FETCHCONTENT_SOURCE_DIR_<N> cache variable so the download is skipped.
#
# CONTENT_NAME: the FetchContent content name (e.g. abseil-cpp, googletest).
# OUT_VAR:      name of a variable, set in the caller's scope, to a list of
#               extra arguments to splice into FetchContent_Declare(). It is set
#               to an empty list when caching is disabled or when the source is
#               reused via FETCHCONTENT_SOURCE_DIR_<N>.
function(LiteRtDepsCache_SourceDirArgs CONTENT_NAME OUT_VAR)
  set(${OUT_VAR} "" PARENT_SCOPE)

  if(NOT LITERT_DEPS_CACHE_DIR)
    return()
  endif()

  string(TOLOWER "${CONTENT_NAME}" _name_lower)
  string(TOUPPER "${CONTENT_NAME}" _name_upper)
  set(_cached_src "${LITERT_DEPS_CACHE_DIR}/${_name_lower}-src")

  # If the cached source already has content, reuse it as-is and skip download.
  # FetchContent matches FETCHCONTENT_SOURCE_DIR_<uppercaseName>.
  if(EXISTS "${_cached_src}")
    file(GLOB _cached_src_contents "${_cached_src}/*")
    if(_cached_src_contents)
      set(FETCHCONTENT_SOURCE_DIR_${_name_upper} "${_cached_src}"
          CACHE INTERNAL "Reuse cached source for ${CONTENT_NAME}")
      message(STATUS
        "LiteRT deps cache: reusing ${CONTENT_NAME} source from ${_cached_src}")
      return()
    endif()
  endif()

  # First fetch: route the populated source into the cache. The download step
  # (and its stamps) still runs under FETCHCONTENT_BASE_DIR/<name>-subbuild.
  message(STATUS
    "LiteRT deps cache: routing ${CONTENT_NAME} source to ${_cached_src}")
  set(${OUT_VAR} SOURCE_DIR "${_cached_src}" PARENT_SCOPE)
endfunction()
