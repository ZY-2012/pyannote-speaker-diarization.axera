#!/usr/bin/env python3
"""Export host-side params (npz) to raw float32/int32 .bin files for the C++ SDK.

Layout mirrors the Python SDK exactly; the C++ binary loads these at runtime.

  fbank.bin    int32[3] window_size window_shift padded | float32 window(400)
               | float32 mel_banks(80x513)
  frontend.bin float32: norm_w norm_b n0_w(80) n0_b(80) n1_w(60) n1_b(60)
               n2_w(60) n2_b(60)
  ffn.bin      float32: l0_w(256x128) l0_b(128) l1_w(128x128) l1_b(128)
               c_w(128x7) c_b(7)
  powerset.bin float32: mapping(7x3)
  embtail.bin  int32[3] n_frames n_weights feats | float32 seg1_w(256x5120)
               seg1_b(256)
  plda.bin     float32: mean1(256) mean2(128) lda(256x128) plda_mu(128)
               plda_tr(128x128) plda_psi(256)
"""
import sys
from pathlib import Path

import numpy as np
import scipy.linalg

ROOT = Path(__file__).resolve().parent.parent


def write_arrays(path, i32_arrays, f32_arrays):
    with open(path, 'wb') as f:
        for a in i32_arrays:
            f.write(np.ascontiguousarray(np.asarray(a), dtype=np.int32).tobytes())
        for a in f32_arrays:
            f.write(np.ascontiguousarray(np.asarray(a), dtype=np.float32).tobytes())


def main():
    z = np.load(ROOT / 'models' / 'fbank_params.npz')
    write_arrays(ROOT / 'models' / 'fbank.bin',
                 [[z['window_size'], z['window_shift'], z['padded_window_size']]],
                 [z['window'], z['mel_banks']])

    z = np.load(ROOT / 'models' / 'host_frontend.npz')
    write_arrays(ROOT / 'models' / 'frontend.bin', [],
                 [z['norm_w'], z['norm_b'],
                  z['n0_w'], z['n0_b'], z['n1_w'], z['n1_b'], z['n2_w'], z['n2_b']])

    z = np.load(ROOT / 'models' / 'ffn_params.npz')
    write_arrays(ROOT / 'models' / 'ffn.bin', [],
                 [z['l0_w'], z['l0_b'], z['l1_w'], z['l1_b'], z['c_w'], z['c_b']])

    z = np.load(ROOT / 'models' / 'powerset_mapping.npz')
    write_arrays(ROOT / 'models' / 'powerset.bin', [], [z['mapping']])

    z = np.load(ROOT / 'models' / 'host_emb_tail.npz')
    write_arrays(ROOT / 'models' / 'embtail.bin',
                 [[z['n_frames'], z['n_weights'], z['feats']]],
                 [z['seg1_w'], z['seg1_b']])

    # precompute the PLDA transform chain (host-side eigh), C++ stays Eigen-free
    x = np.load(ROOT / 'models' / 'xvec_transform.npz')
    p = np.load(ROOT / 'models' / 'plda.npz')
    W = np.linalg.inv(p['tr'].T.dot(p['tr']))
    B = np.linalg.inv((p['tr'].T / p['psi']).dot(p['tr']))
    acvar, wccn = scipy.linalg.eigh(B, W)
    plda_psi = acvar[::-1]
    plda_tr = wccn.T[::-1]
    write_arrays(ROOT / 'models' / 'plda.bin', [],
                 [x['mean1'], x['mean2'], x['lda'], p['mu'], plda_tr, plda_psi])
    print('params exported to models/*.bin')


if __name__ == '__main__':
    main()
