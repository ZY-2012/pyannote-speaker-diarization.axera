#!/usr/bin/env python3
"""Regenerate LSTM cell calibration for the current CHUNK size (from lstm.py)."""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CALIB_DIR, EXPORT_DIR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'python'))
from community1_sdk.lstm import CHUNK

HID = 128
WAVS = ['/data/shared/huyuan/vad_seg_sr_cluster/ami_eval/audio/EN2002d.Mix-Headset.wav',
        '/data/shared/huyuan/vad_seg_sr_cluster/alimeeting/audio_mono/R8001_M8004_MS801.wav']


def main():
    from pyannote.audio import Model
    D = ('/data/huyuan/.cache/huggingface/hub/models--pyannote--'
         'speaker-diarization-community-1/snapshots/'
         '3533c8cf8e369892e6b79ff1bf80f7b0286a54ee')
    seg = Model.from_pretrained(D + '/segmentation').eval()
    z = np.load(EXPORT_DIR / 'host_frontend.npz')

    def norm(x, w, b):
        return (x - x.mean(axis=2, keepdims=True)) / np.sqrt(
            x.var(axis=2, keepdims=True) + 1e-5) * w.reshape(1, -1, 1) + b.reshape(1, -1, 1)

    def leaky(x):
        return np.where(x > 0, x, 0.01 * x).astype(np.float32)

    sess = {l: ort.InferenceSession(str(EXPORT_DIR / f'lstm_cell_l{l}.onnx'),
                                    providers=['CPUExecutionProvider'])
            for l in range(4)}
    sA = ort.InferenceSession(str(EXPORT_DIR / 'segmentation_sincnet.onnx'),
                              providers=['CPUExecutionProvider'])
    sB1 = ort.InferenceSession(str(EXPORT_DIR / 'segmentation_convs1.onnx'),
                               providers=['CPUExecutionProvider'])
    sB2 = ort.InferenceSession(str(EXPORT_DIR / 'segmentation_convs2.onnx'),
                               providers=['CPUExecutionProvider'])

    all_inputs = {l: [] for l in range(4)}
    for wp in WAVS:
        w, sr = sf.read(wp, dtype='float32', always_2d=False)
        for ws in (0, 160000, 320000):
            ch = w[ws:ws + 160000].astype(np.float32)
            xn = (ch - ch.mean()) / np.sqrt(np.var(ch) + 1e-5) * z['norm_w'] + z['norm_b']
            cols = np.lib.stride_tricks.as_strided(xn, shape=(15975, 251),
                                                   strides=(40, 4)).copy()
            a = sA.run(None, {'unfolded': cols[None]})[0]
            a = leaky(norm(a, z['n0_w'], z['n0_b']))
            b1 = sB1.run(None, {'x': a})[0]
            b1 = leaky(norm(b1, z['n1_w'], z['n1_b']))
            b2 = sB2.run(None, {'x': b1})[0]
            x = leaky(norm(b2, z['n2_w'], z['n2_b']))[0].transpose(1, 0)
            for l in range(4):
                in_dim = x.shape[1]
                T = x.shape[0]
                bwd_seq = x[::-1].copy()
                h = np.zeros((1, 2 * HID), dtype=np.float32)
                c = np.zeros((1, 2 * HID), dtype=np.float32)
                for s in range(0, T, CHUNK):
                    chunk = x[s:s + CHUNK]
                    bchunk = bwd_seq[s:s + CHUNK]
                    xc = np.zeros((CHUNK, 2 * in_dim), dtype=np.float32)
                    xc[:len(chunk), :in_dim] = chunk
                    xc[:len(chunk), in_dim:] = bchunk
                    out = sess[l].run(None, {'x': xc, 'h0': h, 'c0': c})
                    yfull = np.concatenate([out[0], out[1]], 0)
                    h, c = out[2], out[3]
                    all_inputs[l].append((xc.copy(), h.copy(), c.copy()))
                # advance x for the next layer
                bwd_seq = x[::-1].copy()
                h = np.zeros((1, 2 * HID), dtype=np.float32)
                c = np.zeros((1, 2 * HID), dtype=np.float32)
                ys = []
                for s in range(0, T, CHUNK):
                    chunk = x[s:s + CHUNK]
                    bchunk = bwd_seq[s:s + CHUNK]
                    xc = np.zeros((CHUNK, 2 * in_dim), dtype=np.float32)
                    xc[:len(chunk), :in_dim] = chunk
                    xc[:len(chunk), in_dim:] = bchunk
                    out = sess[l].run(None, {'x': xc, 'h0': h, 'c0': c})
                    yfull = np.concatenate([out[0], out[1]], 0)
                    h, c = out[2], out[3]
                    ys.append(yfull[:len(chunk)])
                yall = np.concatenate(ys, axis=0)
                x = np.concatenate([yall[:, :HID], yall[:, HID:][::-1]], axis=1)
            print(f'window {ws} done', flush=True)

    for l in range(4):
        d = CALIB_DIR / f'lstmcell{l}'
        for name in ('x', 'h0', 'c0'):
            dd = d / name
            for old in dd.glob('*.npy'):
                old.unlink()
        samples = all_inputs[l][:64]
        for i, (xc, h, c) in enumerate(samples):
            np.save(d / 'x' / f'{i:04d}.npy', xc.astype(np.float32))
            np.save(d / 'h0' / f'{i:04d}.npy', h.astype(np.float32))
            np.save(d / 'c0' / f'{i:04d}.npy', c.astype(np.float32))
        print(f'layer {l} calib: {len(samples)} samples')


if __name__ == '__main__':
    main()
