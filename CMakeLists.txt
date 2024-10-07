cmake_minimum_required(VERSION 3.18)
project(executorch_all)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(ExternalProject)

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/executorch)
  ExternalProject_Add(
    executorch_src
    PREFIX executorch
    GIT_REPOSITORY https://github.com/mybigday/node-executorch.git
    GIT_TAG dev
    GIT_SHALLOW TRUE
    GIT_SUBMODULES_RECURSE TRUE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND "")
  set(EXECUTORCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/executorch)
else()
  set(EXECUTORCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/executorch)
endif()

# Apply patch for cpuinfo

find_program(GIT git REQUIRED)

cmake_language(DEFER DIRECTORY ${CMAKE_SOURCE_DIR} CALL _reverse_patch_cpuinfo
               ())
function(_reverse_patch_cpuinfo)
  # reverse patch
  execute_process(
    COMMAND ${GIT} apply ${CMAKE_CURRENT_SOURCE_DIR}/scripts/cpuinfo.patch -R
    WORKING_DIRECTORY ${EXECUTORCH_DIR}/backends/xnnpack/third-party/cpuinfo)
endfunction(_reverse_patch_cpuinfo)

add_custom_target(
  patch_cpuinfo ALL
  COMMAND ${GIT} apply ${CMAKE_CURRENT_SOURCE_DIR}/scripts/cpuinfo.patch || true
  WORKING_DIRECTORY ${EXECUTORCH_DIR}/backends/xnnpack/third-party/cpuinfo)

# ExecuTorch main

if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_SYSTEM_PROCESSOR STREQUAL
                                            "ARM64")
  get_filename_component(
    CMAKE_TOOLCHAIN_FILE
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/mingw-w64-aarch64.clang.toolchain.cmake
    ABSOLUTE)
  if(QNN_SDK_ROOT AND CMAKE_BUILD_TYPE STREQUAL "Release")
    list(APPEND EXECUTORCH_CMAKE_ARGS "-DEXECUTORCH_BUILD_QNN=ON")
  endif()
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_SYSTEM_PROCESSOR STREQUAL
                                            "x86_64")
  get_filename_component(
    CMAKE_TOOLCHAIN_FILE
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/mingw-w64-x86_64.clang.toolchain.cmake
    ABSOLUTE)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_BUILD_TYPE STREQUAL "Release")
  set(EXECUTORCH_CMAKE_ARGS
      "${EXECUTORCH_CMAKE_ARGS} -DEXECUTORCH_BUILD_COREML=ON")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  list(APPEND EXECUTORCH_CMAKE_ARGS "-DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON")
  list(APPEND EXECUTORCH_CMAKE_ARGS "-DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON")
  list(APPEND EXECUTORCH_CMAKE_ARGS "-DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON")
  list(APPEND EXECUTORCH_CMAKE_ARGS "-DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON")
  list(APPEND EXECUTORCH_CMAKE_ARGS "-DEXECUTORCH_BUILD_CPUINFO=ON")
  list(APPEND EXECUTORCH_CMAKE_ARGS "-DEXECUTORCH_XNNPACK_SHARED_WORKSPACE=ON")
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "ARM64" AND NOT CMAKE_SYSTEM_NAME STREQUAL
                                                 "Darwin")
    list(APPEND EXECUTORCH_CMAKE_ARGS "-DEXECUTORCH_XNNPACK_ENABLE_KLEIDI=ON")
  endif()
endif()

ExternalProject_Add(
  executorch
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/executorch
  SOURCE_DIR ${EXECUTORCH_DIR}
  DEPENDS patch_cpuinfo
  CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE:PATH=${CMAKE_TOOLCHAIN_FILE}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_ASM_COMPILER=${CMAKE_ASM_COMPILER}
             -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}
             -DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}
             -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
             -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
             -DEXECUTORCH_ENABLE_LOGGING=ON
             -DEXECUTORCH_BUILD_XNNPACK=ON
             -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON
             -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON
             -DQNN_SDK_ROOT:PATH=$ENV{QNN_SDK_ROOT}
             ${EXECUTORCH_CMAKE_ARGS})

ExternalProject_Get_Property(executorch SOURCE_DIR)

# TOKENIZERS

set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(executorch/extension/llm/third-party/abseil-cpp)
add_subdirectory(executorch/extension/llm/third-party/re2)
add_subdirectory(executorch/extension/llm/third-party/sentencepiece)

file(GLOB TOKENIZER_SRCS executorch/extension/llm/tokenizer/bpe_tokenizer.cpp
     executorch/extension/llm/tokenizer/tiktoken.cpp)
file(GLOB TOKENIZER_HDRS executorch/extension/llm/tokenizer/*.h)
add_library(tokenizer STATIC ${TOKENIZER_SRCS} ${TOKENIZER_HDRS})
target_include_directories(
  tokenizer
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/executorch/extension/llm/third-party/abseil-cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/executorch/extension/llm/third-party/re2)
target_link_libraries(tokenizer PUBLIC re2::re2 sentencepiece-static)

install(
  TARGETS tokenizer
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# SAMPLER

file(GLOB SAMPLER_SRCS executorch/extension/llm/sampler/sampler.cpp)
file(GLOB SAMPLER_HDRS executorch/extension/llm/sampler/sampler.h)
add_library(sampler STATIC ${SAMPLER_SRCS} ${SAMPLER_HDRS})
target_include_directories(sampler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sampler PUBLIC tokenizer)

install(
  TARGETS sampler
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
