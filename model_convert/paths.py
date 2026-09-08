"""Shared paths for pyannote community-1 conversion.

Environment overrides:
    COMMUNITY1_SNAPSHOT  local snapshot of pyannote/speaker-diarization-community-1
                         (default: HF cache for the model)
    COMMUNITY1_WAV       calibration audio (default: AliMeeting R8001_M8004)
    COMMUNITY1_OUT       output root (default: this directory)
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

SNAPSHOT = Path(os.environ.get(
    'COMMUNITY1_SNAPSHOT',
    Path.home() / '.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1'
    / 'snapshots' / '3533c8cf8e369892e6b79ff1bf80f7b0286a54ee'))

OUT = Path(os.environ.get('COMMUNITY1_OUT', HERE)).resolve()
WAV = Path(os.environ.get(
    'COMMUNITY1_WAV',
    HERE.parent.parent / 'alimeeting/audio_mono/R8001_M8004_MS801.wav'))

SEG_DIR = SNAPSHOT / 'segmentation'
EMB_DIR = SNAPSHOT / 'embedding'
PLDA_DIR = SNAPSHOT / 'plda'

EXPORT_DIR = OUT / 'export'
CALIB_DIR = OUT / 'calib_data'
COMPILE_DIR = OUT / 'compile'

SEG_ONNX = EXPORT_DIR / 'segmentation.onnx'           # legacy single-graph (fails to compile)
SEG_SINCNET_ONNX = EXPORT_DIR / 'segmentation_sincnet.onnx'
SEG_LSTM_ONNX = EXPORT_DIR / 'segmentation_lstm.onnx'
EMB_ONNX = EXPORT_DIR / 'embedding.onnx'

SAMPLE_RATE = 16000
WINDOW_SEC = 10.0
NUM_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)          # 160000
SEG_FRAMES = 589                                      # PyanNet output frames for 10s
SEG_CLASSES = 7                                       # powerset classes (3 speakers, max 2)
EMB_DIM = 256


def require(path, hint):
    if not path.exists():
        raise SystemExit(f'missing {path}\n  {hint}')
    return path
