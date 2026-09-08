#!/bin/bash
# pyannote community-1 speaker diarization on AX650N.
#
#   bash run_ax650.sh                 # bundled sample, default settings
#   bash run_ax650.sh your.wav        # your own 16 kHz mono wav
#   bash run_ax650.sh your.wav out.rttm
#
# 输出 RTTM（SPEAKER 行）。可选环境变量：
#   STEP=2.0       分割滑动步长（默认 2s；1s 更细但慢 2.8×）
#   NUM_SPEAKERS=4 固定人数（0=自动估计，默认）
set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/soc/lib:${LD_LIBRARY_PATH:-}

WAV="${1:-samples/sample_meeting.wav}"
OUT="${2:-output.rttm}"
STEP="${STEP:-2.0}"
NUM_SPEAKERS="${NUM_SPEAKERS:-0}"

exec python3 python/example.py --wav "$WAV" --out "$OUT" \
    --step "$STEP" --num-speakers "$NUM_SPEAKERS"
