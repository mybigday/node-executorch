#!/bin/bash

set -e

TARGET_OS=$1
TARGET_ARCH=$2
OUT_DIR=$3

if [ -z "$TARGET_OS" ] || [ -z "$TARGET_ARCH" ]; then
    echo "Usage: $0 <target_os> <target_arch>"
    exit 1
fi

if [ "$TARGET_OS" = "windows" ] && [ "$TARGET_ARCH" = "aarch64" ]; then
    echo "Cross compiling for Windows on ARM64 is not supported"
    exit 1
fi
