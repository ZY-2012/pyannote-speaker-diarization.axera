#!/bin/bash
# Cross-compile community1_diar for AX650N.
#
# Prerequisites (see download_toolchains.sh):
#   - gcc-arm-9.2 aarch64 cross compiler
#   - ax650n BSP SDK (msp/out with include/ax_engine_api.h + lib/)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

rm -rf build_ax650
mkdir -p build_ax650 && cd build_ax650

cmake .. \
  -DAX650=ON \
  -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-none-linux-gnu.toolchain.cmake \
  -DCMAKE_INSTALL_PREFIX=../install/ax650 \
  -DCMAKE_BUILD_TYPE=Release \
  $@

make -j$(nproc)
make install

echo ""
echo "========================================"
echo "Build complete!"
echo "Executable: $(pwd)/../install/ax650/community1_diar"
echo "========================================"
