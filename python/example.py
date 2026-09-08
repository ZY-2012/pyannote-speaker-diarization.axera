#!/usr/bin/env python3
"""Run community-1 diarization with axmodels (or onnx for host validation).

Usage:
  python example.py --wav input.wav [--out out.rttm]
                   [--seg segmentation_sincnet.axmodel] [--emb embedding.axmodel]
                   [--assets dir] [--num-speakers 4] [--onnx]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from community1_sdk.pipeline import Pipeline, run_pipeline  # noqa: E402

HERE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav', required=True)
    ap.add_argument('--out', default='')
    ap.add_argument('--seg-a', default=str(HERE / 'models' / 'segmentation_sincnet.axmodel'))
    ap.add_argument('--seg-b1', default=str(HERE / 'models' / 'segmentation_convs1.axmodel'))
    ap.add_argument('--seg-b2', default=str(HERE / 'models' / 'segmentation_convs2.axmodel'))
    ap.add_argument('--emb', default=str(HERE / 'models' / 'embedding_convs.axmodel'))
    ap.add_argument('--assets', default=str(HERE / 'models'))
    ap.add_argument('--num-speakers', type=int, default=0, help='0 = auto estimate')
    ap.add_argument('--step', type=float, default=1.0, help='segmentation sliding step (s)')
    ap.add_argument('--threshold', type=float, default=0.6)
    ap.add_argument('--fa', type=float, default=0.07)
    ap.add_argument('--fb', type=float, default=0.8)
    ap.add_argument('--onnx', action='store_true', help='use onnxruntime (host only)')
    args = ap.parse_args()

    if args.onnx:
        ex = str(Path(__file__).resolve().parent.parent / 'model_convert' / 'export')
        args.seg_a = args.seg_a.replace('.axmodel', '.onnx').replace('models', 'model_convert/export')
        args.seg_b1 = args.seg_b1.replace('.axmodel', '.onnx').replace('models', 'model_convert/export')
        args.seg_b2 = args.seg_b2.replace('.axmodel', '.onnx').replace('models', 'model_convert/export')
        args.emb = args.emb.replace('.axmodel', '.onnx').replace('models', 'model_convert/export')
        args.assets = ex

    assets = Path(args.assets)
    pipeline = Pipeline(
        seg_a=args.seg_a,
        seg_b1=args.seg_b1,
        seg_b2=args.seg_b2,
        host_params=assets / 'host_frontend.npz',
        lstm_params=assets / 'lstm_params.npz',
        ffn_params=assets / 'ffn_params.npz',
        powerset_map=assets / 'powerset_mapping.npz',
        emb_model=args.emb,
        emb_tail=assets / 'host_emb_tail.npz',
        fbank_params=assets / 'fbank_params.npz',
        transform_npz=assets / 'xvec_transform.npz',
        plda_npz=assets / 'plda.npz',
        threshold=args.threshold, Fa=args.fa, Fb=args.fb,
        num_speakers=args.num_speakers or None,
        lstm_cells=None if args.onnx else [HERE / 'models' / f'lstm_cell_l{l}.axmodel'
                                           for l in range(4)],
        step_sec=args.step,
    )

    wav, sr = sf.read(args.wav, dtype='float32', always_2d=False)
    assert sr == 16000, f'expect 16 kHz audio, got {sr}'
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    print(f'audio {wav.shape[0] / sr:.1f}s', flush=True)

    t0 = time.time()
    uri = Path(args.wav).stem
    segments = run_pipeline(pipeline, wav, uri=uri)
    rt = time.time() - t0
    print(f'pipeline took {rt:.1f}s (RTF {rt / (wav.shape[0] / sr):.3f})')
    print(f'{len(segments)} segments')
    lines = [f'SPEAKER {uri} 1 {s:7.3f} {d:7.3f} <NA> <NA> {spk} <NA> <NA>'
             for spk, s, d in segments]
    if args.out:
        Path(args.out).write_text('\n'.join(lines) + '\n')
        print(f'wrote {args.out}')
    else:
        for line in lines[:20]:
            print(line)


if __name__ == '__main__':
    main()
