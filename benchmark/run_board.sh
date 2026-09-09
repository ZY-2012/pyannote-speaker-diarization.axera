#!/bin/bash
# 板端批量评测：对 wav 目录逐文件跑 C++ 推理，输出 RTTM 目录。
# 用法：bash run_board.sh /path/to/wav_dir /path/to/out_dir
set -euo pipefail
cd "$(dirname "$0")/.."

WAV_DIR="${1:?usage: run_board.sh wav_dir out_dir}"
OUT_DIR="${2:?usage: run_board.sh wav_dir out_dir}"
mkdir -p "$OUT_DIR"

for wav in "$WAV_DIR"/*.wav; do
    fid=$(basename "$wav" .wav)
    [ -f "$OUT_DIR/$fid.rttm" ] && continue
    ./bin/community1_diar_ax650 --model-dir models --wav "$wav" \
        --out "$OUT_DIR/$fid.rttm" --step 2.0
done
echo "done: $(ls "$OUT_DIR"/*.rttm | wc -l) rttms"
