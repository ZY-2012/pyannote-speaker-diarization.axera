#!/bin/bash
# Download the cross toolchain + AX650 BSP SDK for cpp/ builds.
set -e
cd "$(dirname "$0")"

TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-/data/shared/huyuan/toolchains}"
mkdir -p "$TOOLCHAIN_ROOT"

# 1) gcc-arm-9.2 aarch64 cross compiler
GCC_TAR="gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"
if [ ! -d "$TOOLCHAIN_ROOT/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu" ]; then
    echo "downloading $GCC_TAR ..."
    wget -q "https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/$GCC_TAR" \
        -O "$TOOLCHAIN_ROOT/$GCC_TAR"
    tar -xf "$TOOLCHAIN_ROOT/$GCC_TAR" -C "$TOOLCHAIN_ROOT"
fi

# 2) AX650 BSP SDK (msp/out provides ax_engine_api.h + libax_engine/libax_sys/libax_interpreter)
if [ ! -d "$TOOLCHAIN_ROOT/ax650n_bsp_sdk/msp/out" ]; then
    echo "cloning ax650n_bsp_sdk ..."
    git clone --depth=1 https://github.com/AXERA-TECH/ax650n_bsp_sdk.git \
        "$TOOLCHAIN_ROOT/ax650n_bsp_sdk"
fi

echo "toolchain ready under $TOOLCHAIN_ROOT"
