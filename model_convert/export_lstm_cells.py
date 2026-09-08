#!/usr/bin/env python3
"""Export the 4-layer bidirectional LSTM as 4 chunked cell axmodels.

Pulsar2 does not support the ONNX LSTM op, so the recurrence is unrolled into
basic ops: each cell processes CHUNK=32 timesteps of BOTH directions in one
call (fwd x and reversed x concatenated along the feature axis, block-diagonal
weights), host carries h/c state in FP32 across chunks (exact at the boundary,
quantized inside each 32-step chunk).

cellL0 : x (32, 120)  h0 (256,) c0 (256,) -> y (32, 256) hN (256,) cN (256,)
cellL1-3: x (32, 512)  (same state/output shapes)
"""
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from paths import EXPORT_DIR

CHUNK = 96
HID = 128
CLAMP_C = 8.0


def build_cell(Wih_f, bih_f, Whh_f, bhh_f, Wih_r, bih_r, Whh_r, bhh_r, in_dim):
    class Cell(nn.Module):
        def forward(self, x, h0, c0):          # x (96, 2*in), h0 (1,256), c0 (1,256)
            h = h0
            c = c0
            # precompute the x-dependent part ONCE per chunk (one big gemm),
            # leaving only the small (256x1024) h-gemm on the serial path
            X = torch.matmul(x, Wih) + bih     # (96, 1024)
            ys = []
            for t in range(x.shape[0]):
                g = X[t:t + 1] + torch.matmul(h, Wh) + bhh   # (1,1024)
                i = torch.sigmoid(g[:, 0:HID])
                f = torch.sigmoid(g[:, HID:2 * HID])
                gg = torch.tanh(g[:, 2 * HID:3 * HID])
                o = torch.sigmoid(g[:, 3 * HID:4 * HID])
                i2 = torch.sigmoid(g[:, 4 * HID:5 * HID])
                f2 = torch.sigmoid(g[:, 5 * HID:6 * HID])
                gg2 = torch.tanh(g[:, 6 * HID:7 * HID])
                o2 = torch.sigmoid(g[:, 7 * HID:8 * HID])
                c = torch.cat([f, f2], 1) * c + torch.cat([i, i2], 1) * torch.cat([gg, gg2], 1)
                # clamp c: tanh(c) saturates for |c| > ~6, so clamping is exact for
                # the h output and keeps the U16 quant step small (see README)
                c = torch.clamp(c, -CLAMP_C, CLAMP_C)
                h = torch.cat([o, o2], 1) * torch.tanh(c)
                ys.append(h)
            # PPQ breaks on a single Concat with >= ~128 inputs: split into two
            # half-chunk outputs (each Concat(CHUNK/2)) and cat them on the host.
            half = len(ys) // 2
            y0 = torch.cat(ys[:half], 0)
            y1 = torch.cat(ys[half:], 0)
            return y0, y1, h, c

    Wih = torch.zeros(2 * in_dim, 8 * HID)
    Wih[:in_dim, :4 * HID] = Wih_f.t()
    Wih[in_dim:, 4 * HID:] = Wih_r.t()
    bih = torch.cat([bih_f, bih_r], 0)[None, :]             # (1, 1024)
    Wh = torch.zeros(2 * HID, 8 * HID)
    Wh[:HID, :4 * HID] = Whh_f.t()
    Wh[HID:, 4 * HID:] = Whh_r.t()
    bhh = torch.cat([bhh_f, bhh_r], 0)[None, :]             # (1, 1024)
    return Cell(), Wih, bih, Wh, bhh


def main():
    z = np.load(EXPORT_DIR / 'lstm_params.npz')
    for l in range(4):
        in_dim = 60 if l == 0 else 256
        cell, Wih, bih, Wh, bhh = build_cell(
            torch.from_numpy(z[f'weight_ih_l{l}']), torch.from_numpy(z[f'bias_ih_l{l}']),
            torch.from_numpy(z[f'weight_hh_l{l}']), torch.from_numpy(z[f'bias_hh_l{l}']),
            torch.from_numpy(z[f'weight_ih_l{l}_reverse']), torch.from_numpy(z[f'bias_ih_l{l}_reverse']),
            torch.from_numpy(z[f'weight_hh_l{l}_reverse']), torch.from_numpy(z[f'bias_hh_l{l}_reverse']),
            in_dim)
        cell.Wih = Wih
        cell.bih = bih
        cell.Wh = Wh
        cell.bhh = bhh
        x = torch.randn(CHUNK, 2 * in_dim)
        h0 = torch.randn(1, 2 * HID)
        c0 = torch.randn(1, 2 * HID)
        torch.onnx.export(cell.eval(), (x, h0, c0),
                          str(EXPORT_DIR / f'lstm_cell_l{l}.onnx'),
                          input_names=['x', 'h0', 'c0'],
                          output_names=['y0', 'y1', 'hN', 'cN'],
                          dynamic_axes=None, opset_version=17, do_constant_folding=True)
        print(f'exported lstm_cell_l{l}.onnx  x {tuple(x.shape)}')


if __name__ == '__main__':
    main()
