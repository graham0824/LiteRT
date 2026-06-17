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
# VERSION (optional, 3rd arg): a version/tag string for the dependency. When
#               given, the cache dir is keyed by it (<name>-<version>-src) so a
#               version bump maps to a *different* directory and is fetched
#               automatically instead of silently reusing the old source. When
#               omitted the dir is just <name>-src (callers with no clean
#               version string, e.g. opaque URLs or git hashes, omit it and keep
#               the original behavior, including the manual-clear-on-bump caveat).
function(LiteRtDepsCache_SourceDirArgs CONTENT_NAME OUT_VAR)
  set(${OUT_VAR} "" PARENT_SCOPE)

  if(NOT LITERT_DEPS_CACHE_DIR)
    return()
  endif()

  # Optional VERSION (3rd positional arg) keys the cache dir by version.
  set(_version "")
  if(ARGC GREATER 2)
    set(_version "${ARGV2}")
  endif()

  string(TOLOWER "${CONTENT_NAME}" _name_lower)
  string(TOUPPER "${CONTENT_NAME}" _name_upper)
  if(_version)
    # Sanitize so the version is safe as a single path segment.
    string(REGEX REPLACE "[^A-Za-z0-9._-]" "_" _version_sanitized "${_version}")
    set(_cached_src "${LITERT_DEPS_CACHE_DIR}/${_name_lower}-${_version_sanitized}-src")
  else()
    set(_cached_src "${LITERT_DEPS_CACHE_DIR}/${_name_lower}-src")
  endif()

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

  # First fetch (or the cached source was cleared, e.g. after a version bump):
  # route the populated source into the cache. The download step (and its
  # stamps) still runs under FETCHCONTENT_BASE_DIR/<name>-subbuild.
  #
  # Clear any stale FETCHCONTENT_SOURCE_DIR_<N> override left in the CMake cache
  # by a previous "reuse" configure. Without this, deleting the cached source
  # (to force a re-download) leaves a dangling override and FetchContent fails
  # with "Manually specified source directory is missing".
  unset(FETCHCONTENT_SOURCE_DIR_${_name_upper} CACHE)
  message(STATUS
    "LiteRT deps cache: routing ${CONTENT_NAME} source to ${_cached_src}")
  set(${OUT_VAR} SOURCE_DIR "${_cached_src}" PARENT_SCOPE)
endfunction()
