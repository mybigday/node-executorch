cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0042 NEW)

project(node-executorch)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT DEFINED napi_build_version)
  set(napi_build_version 6)
endif()
add_definitions(-DNAPI_VERSION=${napi_build_version})
message(STATUS "NAPI_VERSION: ${napi_build_version}")

set(CMAKE_SYSTEM_PROCESSOR ${NODE_ARCH})
string(TOLOWER ${CMAKE_SYSTEM_NAME} PLATFORM)
string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} ARCH)

# normalize the platform name to match the node platform names
if(PLATFORM STREQUAL "windows")
  set(PLATFORM "win32")
endif()

# normalize the arch name to match the node arch names
if(ARCH STREQUAL "x86_64" OR ARCH STREQUAL "amd64")
  set(ARCH "x64")
elseif(ARCH STREQUAL "i386")
  set(ARCH "x86")
elseif(ARCH STREQUAL "arm64x" OR ARCH STREQUAL "aarch64")
  set(ARCH "arm64")
endif()

set(OUTPUT_BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin/${PLATFORM}/${ARCH})

message(STATUS "Platform: ${PLATFORM}")
message(STATUS "Arch: ${ARCH}")

# This is the funtion to use -Wl, --whole-archive to link static library NB:
# target_link_options is broken for this case, it only append the interface link
# options of the first library.
function(kernel_link_options target_name)
  # target_link_options(${target_name} INTERFACE
  # "$<LINK_LIBRARY:WHOLE_ARCHIVE,target_name>")
  target_link_options(
    ${target_name} INTERFACE "SHELL:LINKER:--whole-archive \
    $<TARGET_FILE:${target_name}> \
    LINKER:--no-whole-archive"
  )
endfunction()

# Same as kernel_link_options but it's for MacOS linker
function(macos_kernel_link_options target_name)
  target_link_options(
    ${target_name} INTERFACE
    "SHELL:LINKER:-force_load,$<TARGET_FILE:${target_name}>"
  )
endfunction()

# Ensure that the load-time constructor functions run. By default, the linker
# would remove them since there are no other references to them.
function(target_link_options_shared_lib target_name)
  if(APPLE)
    macos_kernel_link_options(${target_name})
  else()
    kernel_link_options(${target_name})
  endif()
endfunction()

file(
  GLOB SOURCES
  src/addon.cc
  src/Module.h
  src/Module.cpp
  src/Tensor.h
  src/Tensor.cpp
  src/utils.h
  src/utils.cpp
)

if(NOT EXECUTORCH_SRC_ROOT)
  include(FetchContent)
  FetchContent_Declare(
    executorch
    GIT_REPOSITORY https://github.com/pytorch/executorch.git
    GIT_TAG main
  )
  FetchContent_MakeAvailable(executorch)
  set(EXECUTORCH_SRC_ROOT ${executorch_SOURCE_DIR})
endif()

find_package(executorch REQUIRED CONFIG)
set(link_libraries executorch extension_module extension_data_loader)

if(TARGET optimized_native_cpu_ops_lib)
  message(STATUS "Optimized kernels enabled")
  list(
    APPEND
    link_libraries
    optimized_native_cpu_ops_lib
    optimized_kernels
    portable_kernels
    cpublas
    eigen_blas
  )
  target_link_options_shared_lib(optimized_native_cpu_ops_lib)
else()
  list(APPEND link_libraries portable_ops_lib portable_kernels)
  target_link_options_shared_lib(portable_ops_lib)
endif()

# quantized_ops_lib: Register quantized op kernels into the runtime
target_link_options_shared_lib(quantized_ops_lib)
list(APPEND link_libraries quantized_kernels quantized_ops_lib)

if(TARGET cpuinfo)
  list(APPEND link_libraries cpuinfo)
endif()

if(TARGET pthreadpool)
  list(APPEND link_libraries pthreadpool)
endif()

# custom_ops
if(TARGET custom_ops)
  message(STATUS "Custom ops enabled")
  list(APPEND link_libraries custom_ops)
  target_link_options_shared_lib(custom_ops)
endif()

# XNNPACK
if(TARGET xnnpack_backend)
  message(STATUS "XNNPACK backend enabled")
  list(APPEND link_libraries xnnpack_backend XNNPACK)
  target_link_options_shared_lib(xnnpack_backend)
endif()

# Vulkan backend
if(TARGET vulkan_backend)
  message(STATUS "Vulkan backend enabled")
  list(APPEND link_libraries vulkan_backend)
  target_link_options_shared_lib(vulkan_backend)
endif()

# Qnn backend
if(TARGET qnn_executorch_backend)
  message(STATUS "QNN backend enabled")
  list(APPEND link_libraries qnn_executorch_backend)
  target_link_options_shared_lib(qnn_executorch_backend)
endif()

# MPS backend
if(TARGET mpsdelegate)
  message(STATUS "MPS backend enabled")
  list(
    APPEND
    link_libraries
    mpsdelegate
    "-framework Foundation"
    "-weak_framework MetalPerformanceShaders"
    "-weak_framework MetalPerformanceShadersGraph"
    "-weak_framework Metal"
  )
  target_link_options_shared_lib(mpsdelegate)
endif()

include_directories(
  ${CMAKE_JS_INC}
  ${executorch_INCLUDE_DIRS}
  "${EXECUTORCH_SRC_ROOT}/.."
)
add_library(${PROJECT_NAME} SHARED ${SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
target_link_libraries(
  ${PROJECT_NAME}
  ${CMAKE_JS_LIB}
  ${link_libraries}
)

if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
    # Generate node.lib
    execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()

add_custom_target(
  copy_assets ALL DEPENDS ${PROJECT_NAME}
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${OUTPUT_BIN_DIR}/$<TARGET_FILE_NAME:${PROJECT_NAME}>
  COMMENT "Copying assets"
)
