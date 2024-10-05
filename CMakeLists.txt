cmake_minimum_required(VERSION 3.10)
project(executorch_all)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(executorch/build/Utils.cmake)

# CONFIG

option(EXECUTORCH_INSTALL_PATH "Path to install executorch" "" FORCE)

set(link_libraries "executorch" "executorch_core")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(
    APPEND link_libraries
    extension_data_loader
    extension_module_static
    portable_ops_lib
    portable_kernels
  )
else()
  list(
    APPEND link_libraries
    extension_data_loader
    extension_module_static
    optimized_native_cpu_ops_lib
    optimized_kernels
    portable_kernels
    eigen_blas
    quantized_ops_lib
    quantized_kernels
    custom_ops
    extension_tensor
    extension_runner_util
    cpuinfo
    pthreadpool
    xnnpack_backend
    XNNPACK
    microkernels-prod
  )
  if (CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "ARM64")
    list(APPEND link_libraries "qnn_executorch_backend")
  endif()
  if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(ET_CMAKE_ARGS "${ET_CMAKE_ARGS} -DEXECUTORCH_BUILD_COREML=ON")
  endif()
endif()

set(executorch_DIR ${EXECUTORCH_INSTALL_PATH})
find_package(executorch REQUIRED CONFIG)

# TOKENIZERS

set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(executorch/extension/llm/third-party/abseil-cpp)
add_subdirectory(executorch/extension/llm/third-party/re2)
add_subdirectory(executorch/extension/llm/third-party/sentencepiece)

file(GLOB_RECURSE TOKENIZER_SRCS
  executorch/extension/llm/tokenizer/bpe_tokenizer.cpp
  executorch/extension/llm/tokenizer/tiktoken.cpp
)
file(GLOB_RECURSE TOKENIZER_HDRS
  executorch/extension/llm/tokenizer/*.h
)
add_library(tokenizer STATIC
  ${TOKENIZER_SRCS}
  ${TOKENIZER_HDRS}
)
target_include_directories(tokenizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# list(
#   APPEND link_libraries
#   tokenizer
#   re2::re2
#   sentencepiece-static
# )


# ALL

add_library(executorch_all INTERFACE)

target_link_options_shared_lib(executorch)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_shared_lib(portable_ops_lib)
else()
  target_link_options_shared_lib(optimized_native_cpu_ops_lib)
  target_link_options_shared_lib(quantized_ops_lib)
  target_link_options_shared_lib(custom_ops)
  target_link_options_shared_lib(xnnpack_backend)
  if (CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "ARM64")
    target_link_options_shared_lib(qnn_executorch_backend)
  endif()
  if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_options_shared_lib(coremldelegate)
  endif()
endif()

target_link_libraries(executorch_all INTERFACE ${link_libraries})

# INSTALL

install(TARGETS executorch_all
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
