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

CHUNK = 64
HID = 128
CLAMP_C = 8.0


def build_cell(Wih_f, bih_f, Whh_f, bhh_f, Wih_r, bih_r, Whh_r, bhh_r, in_dim):
    class Cell(nn.Module):
        def forward(self, x, h0, c0):          # x (64, 2*in), h0 (1,256), c0 (1,256)
            h = h0
            c = c0
            ys = []
            for t in range(x.shape[0]):
                xt = x[t:t + 1]                # (1, 2*in)
                # fused single gemm: [x, h] @ Wcat  (one NPU op per step)
                g = torch.matmul(torch.cat([xt, h], 1), Wcat) + bcat   # (1,1024)
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
            return torch.cat(ys, 0), h, c

    Wih = torch.zeros(2 * in_dim, 8 * HID)
    Wih[:in_dim, :4 * HID] = Wih_f.t()
    Wih[in_dim:, 4 * HID:] = Wih_r.t()
    Wh = torch.zeros(2 * HID, 8 * HID)
    Wh[:HID, :4 * HID] = Whh_f.t()
    Wh[HID:, 4 * HID:] = Whh_r.t()
    Wcat = torch.cat([Wih, Wh], 0)                          # (2*in + 2*HID, 1024)
    bcat = (torch.cat([bih_f, bih_r], 0) + torch.cat([bhh_f, bhh_r], 0))[None, :]
    return Cell(), Wcat, bcat


def main():
    z = np.load(EXPORT_DIR / 'lstm_params.npz')
    for l in range(4):
        in_dim = 60 if l == 0 else 256
        cell, Wcat, bcat = build_cell(
            torch.from_numpy(z[f'weight_ih_l{l}']), torch.from_numpy(z[f'bias_ih_l{l}']),
            torch.from_numpy(z[f'weight_hh_l{l}']), torch.from_numpy(z[f'bias_hh_l{l}']),
            torch.from_numpy(z[f'weight_ih_l{l}_reverse']), torch.from_numpy(z[f'bias_ih_l{l}_reverse']),
            torch.from_numpy(z[f'weight_hh_l{l}_reverse']), torch.from_numpy(z[f'bias_hh_l{l}_reverse']),
            in_dim)
        cell.Wcat = Wcat
        cell.bcat = bcat
        x = torch.randn(CHUNK, 2 * in_dim)
        h0 = torch.randn(1, 2 * HID)
        c0 = torch.randn(1, 2 * HID)
        torch.onnx.export(cell.eval(), (x, h0, c0),
                          str(EXPORT_DIR / f'lstm_cell_l{l}.onnx'),
                          input_names=['x', 'h0', 'c0'], output_names=['y', 'hN', 'cN'],
                          dynamic_axes=None, opset_version=17, do_constant_folding=True)
        print(f'exported lstm_cell_l{l}.onnx  x {tuple(x.shape)}')


if __name__ == '__main__':
    main()
