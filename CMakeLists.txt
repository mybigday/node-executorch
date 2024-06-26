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

include(FetchContent)
include(cmake/Utils.cmake)

file(
  GLOB SOURCES
  src/addon.cc
  src/Module.h
  src/Module.cpp
  src/Tensor.h
  src/Tensor.cpp
  src/Sampler.h
  src/Sampler.cpp
  src/utils.h
  src/utils.cpp
  "${EXECUTORCH_SRC_ROOT}/examples/models/llama2/sampler/sampler.cpp"
)

if(MINGW)
  if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    string(REPLACE "/DELAYLOAD:NODE.EXE" "" CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS})
  endif()
  file(GLOB WIN_DELAY_LOAD node_modules/cmake-js/src/win_delay_load_hook.cc)
  list(APPEND SOURCES ${WIN_DELAY_LOAD})

  if(NOT CMAKE_JS_NODELIB_TARGET)
    set(CMAKE_JS_NODELIB_TARGET "${CMAKE_BINARY_DIR}/node.lib")
  endif()
  if(NOT CMAKE_JS_NODELIB_DEF)
    set(CMAKE_JS_NODELIB_DEF "${CMAKE_SOURCE_DIR}/node_modules/node-api-headers/def/node_api.def")
  endif()
  execute_process(COMMAND ${TOOLCHAIN_PREFIX}-dlltool -d ${CMAKE_JS_NODELIB_DEF} -l ${CMAKE_JS_NODELIB_TARGET})
endif()

if(NOT EXECUTORCH_SRC_ROOT)
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

# copy assets
add_custom_target(
  copy_assets ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_BIN_DIR}
  COMMENT "Creating output directory"
)

# copy libextension_module.so or libextension_module.dll
add_custom_command(
  TARGET copy_assets
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${LIB_extension_module} ${OUTPUT_BIN_DIR}
  COMMENT "Copying extension_module"
)

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

if(TARGET quantized_ops_lib)
  message(STATUS "Quantized kernels enabled")
  list(APPEND link_libraries quantized_kernels quantized_ops_lib)
  target_link_options_shared_lib(quantized_ops_lib)
endif()

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
  add_custom_command(
    TARGET copy_assets
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${LIB_qnn_executorch_backend} ${OUTPUT_BIN_DIR}
    COMMENT "Copying assets"
  )
  if(QNN_SDK_ROOT)
    # copy QNN SDK libraries
    set(QNN_PLATFORM "")
    if(WIN32 AND ARCH STREQUAL "arm64")
      set(QNN_PLATFORM "aarch64-windows-msvc")
    elseif(LINUX AND ARCH STREQUAL "arm64")
      set(QNN_PLATFORM "aarch64-ubuntu-gcc9.4")
    endif()
    if(QNN_PLATFORM)
      file(
        GLOB QNN_LIBS
        ${QNN_SDK_ROOT}/lib/${QNN_PLATFORM}/libQnnSystem.so
        ${QNN_SDK_ROOT}/lib/${QNN_PLATFORM}/libQnnSaver.so
        ${QNN_SDK_ROOT}/lib/${QNN_PLATFORM}/libQnnCpu.so
        ${QNN_SDK_ROOT}/lib/${QNN_PLATFORM}/libQnnDsp.so
        ${QNN_SDK_ROOT}/lib/${QNN_PLATFORM}/libQnnDspV66Stub.so
        ${QNN_SDK_ROOT}/lib/${QNN_PLATFORM}/libQnnHtp.so
        ${QNN_SDK_ROOT}/lib/${QNN_PLATFORM}/libQnnHtpPrepare.so
        ${QNN_SDK_ROOT}/lib/${QNN_PLATFORM}/libQnnHtpV68Stub.so
        ${QNN_SDK_ROOT}/lib/${QNN_PLATFORM}/QnnSystem.dll
        ${QNN_SDK_ROOT}/lib/${QNN_PLATFORM}/QnnSaver.dll
        ${QNN_SDK_ROOT}/lib/${QNN_PLATFORM}/QnnCpu.dll
        ${QNN_SDK_ROOT}/lib/${QNN_PLATFORM}/QnnDsp.dll
        ${QNN_SDK_ROOT}/lib/${QNN_PLATFORM}/QnnDspV66Stub.dll
        ${QNN_SDK_ROOT}/lib/${QNN_PLATFORM}/QnnHtp.dll
        ${QNN_SDK_ROOT}/lib/${QNN_PLATFORM}/QnnHtpPrepare.dll
        ${QNN_SDK_ROOT}/lib/${QNN_PLATFORM}/QnnHtpV68Stub.dll
        ${QNN_SDK_ROOT}/lib/${QNN_PLATFORM}/QnnHtpV73Stub.dll
        ${QNN_SDK_ROOT}/lib/hexagon-v66/unsigned/libQnnHtpV66Skel.so
        ${QNN_SDK_ROOT}/lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so
        ${QNN_SDK_ROOT}/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so
        ${QNN_SDK_ROOT}/lib/hexagon-v73/unsigned/libqnnhtpv73.cat

      )
      add_custom_command(
        TARGET copy_assets
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${QNN_LIBS} ${OUTPUT_BIN_DIR}
        COMMENT "Copying assets"
      )
    endif()
  endif()
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
  ${executorch_INCLUDE_DIRS}
  "${EXECUTORCH_SRC_ROOT}/.."
  ${CMAKE_JS_INC}
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3)
endif()

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${CMAKE_JS_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
target_link_libraries(
  ${PROJECT_NAME}
  ${link_libraries}
  ${CMAKE_JS_LIB}
)

if(CMAKE_JS_NODELIB_TARGET AND MINGW)
  set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-lnode -L ${CMAKE_BINARY_DIR}")
endif()

add_dependencies(copy_assets ${PROJECT_NAME})

add_custom_command(
  TARGET copy_assets
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${OUTPUT_BIN_DIR}/$<TARGET_FILE_NAME:${PROJECT_NAME}>
  COMMENT "Copying assets"
)

if(MINGW)
  # copy libunwind.dll, libc++.dll form the MinGW directory
  find_program(MINGW_CC ${CMAKE_C_COMPILER})
  get_filename_component(MINGW_DIR ${MINGW_CC} DIRECTORY)
  get_filename_component(MINGW_DIR ${MINGW_DIR} DIRECTORY)
  if(ARCH STREQUAL "arm64")
    set(MINGW_DIR ${MINGW_DIR}/aarch64-w64-mingw32)
  else()
    set(MINGW_DIR ${MINGW_DIR}/x86_64-w64-mingw32)
  endif()
  file(
    GLOB MINGW_DLLS
    ${MINGW_DIR}/bin/libunwind.dll
    ${MINGW_DIR}/bin/libc++.dll
  )
  add_custom_command(
    TARGET copy_assets
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${MINGW_DLLS} ${OUTPUT_BIN_DIR}
    COMMENT "Copying MinGW DLLs"
  )
endif()
