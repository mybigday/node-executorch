#!/bin/bash

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <path-to-executorch-src>"
  exit 1
fi

ET_SRC=$(realpath $1)
PROJECT_DIR=$(realpath $(dirname $0))

if [ ! -d "$ET_SRC/cmake-linux-x64-out" ]; then
  pushd $ET_SRC
  cmake . \
    -B cmake-linux-x64-out -DCMAKE_INSTALL_PREFIX=cmake-linux-x64-out \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_CPUINFO=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_PTHREADPOOL=ON
  cmake --build cmake-linux-x64-out --config Release --target install -j4
  popd
fi

rm -rf build && \
  yarn build --CDCMAKE_PREFIX_PATH=$ET_SRC/cmake-linux-x64-out \
    --CDEXECUTORCH_SRC_ROOT=$ET_SRC

if [ ! -d "$ET_SRC/cmake-linux-arm64-out" ]; then
  pushd $ET_SRC
  cmake . \
    -B cmake-linux-arm64-out -DCMAKE_INSTALL_PREFIX=cmake-linux-arm64-out \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_CPUINFO=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_PTHREADPOOL=ON \
    -DCMAKE_TOOLCHAIN_FILE="$PROJECT_DIR/cmake/clang-aarch64-linux-gnu.toolchain.cmake"
  cmake --build cmake-linux-arm64-out --config Release --target install -j4
  popd
fi

rm -rf build && \
yarn build --CDCMAKE_PREFIX_PATH=$ET_SRC/cmake-linux-arm64-out \
  --CDEXECUTORCH_SRC_ROOT=$ET_SRC --CDCMAKE_TOOLCHAIN_FILE=cmake/clang-aarch64-linux-gnu.toolchain.cmake

if [ ! -d "$ET_SRC/cmake-win-x64-out" ]; then
  pushd $ET_SRC
  cmake . \
    -B cmake-win-x64-out -DCMAKE_INSTALL_PREFIX=cmake-win-x64-out \
    -DEXECUTORCH_ENABLE_LOGGING=1 \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_CPUINFO=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_PTHREADPOOL=ON \
    -DCMAKE_TOOLCHAIN_FILE="$PROJECT_DIR/cmake/mingw-w64-x86_64.toolchain.cmake"
  cmake --build cmake-win-x64-out --config Release --target install -j4
  popd
fi

rm -rf build && \
  yarn build --CDCMAKE_PREFIX_PATH=$ET_SRC/cmake-win-arm64-out \
    --CDEXECUTORCH_SRC_ROOT=$ET_SRC --CDCMAKE_TOOLCHAIN_FILE=cmake/mingw-w64-aarch64.toolchain.cmake

if [ ! -d "$ET_SRC/cmake-win-arm64-out" ]; then
  pushd $ET_SRC
  if [ -z "$QNN_SDK_ROOT" ]; then
    cmake . \
      -B cmake-win-arm64-out -DCMAKE_INSTALL_PREFIX=cmmake-win-arm64-out \
      -DEXECUTORCH_ENABLE_LOGGING=1 \
      -DCMAKE_BUILD_TYPE=Release \
      -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
      -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
      -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
      -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
      -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
      -DEXECUTORCH_BUILD_CPUINFO=ON \
      -DEXECUTORCH_BUILD_XNNPACK=ON \
      -DEXECUTORCH_BUILD_PTHREADPOOL=ON \
      -DCMAKE_TOOLCHAIN_FILE="$PROJECT_DIR/cmake/mingw-w64-aarch64.toolchain.cmake"
    cmake --build cmake-win-arm64-out --config Release --target install -j4
  else
    cmake . \
      -B cmake-win-arm64-out -DCMAKE_INSTALL_PREFIX=cmmake-win-arm64-out \
      -DEXECUTORCH_ENABLE_LOGGING=1 \
      -DCMAKE_BUILD_TYPE=Release \
      -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
      -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
      -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
      -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
      -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
      -DEXECUTORCH_BUILD_QNN=ON \
      -DQNN_SDK_ROOT=$QNN_SDK_ROOT \
      -DEXECUTORCH_BUILD_CPUINFO=ON \
      -DEXECUTORCH_BUILD_XNNPACK=ON \
      -DEXECUTORCH_BUILD_PTHREADPOOL=ON \
      -DEXECUTORCH_BUILD_SDK=ON \
      -DCMAKE_TOOLCHAIN_FILE="$PROJECT_DIR/cmake/mingw-w64-aarch64.toolchain.cmake"
    cmake --build cmake-win-arm64-out --config Release --target install -j4
  fi
  popd
fi

rm -rf build && \
yarn build --CDCMAKE_PREFIX_PATH=$ET_SRC/cmake-win-x64-out \
  --CDEXECUTORCH_SRC_ROOT=$ET_SRC --CDCMAKE_TOOLCHAIN_FILE=cmake/mingw-w64-x86_64.toolchain.cmake
