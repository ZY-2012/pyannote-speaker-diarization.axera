#!/usr/bin/env python3
"""Export the sincnet frontend as two NPU models with host-side InstanceNorm.

Findings (AX650 NPU3):
  * C=1 input Conv1d: computed wrong (check=2 xrun mismatch) -> replaced by
    im2col(host) + MatMul(NPU)
  * Unfold-in-graph: AxGather fails to tile -> im2col runs on host (~8 ms)
  * InstanceNorm (both the ONNX op and a manual decomposition): numerically
    broken in the U16 quantized domain on real data -> runs on host in FP32
  * Abs / MaxPool / LeakyRelu / multi-channel Conv / MatMul: verified accurate

Model A: unfolded (1,15975,251) -> MatMul -> Abs -> MaxPool -> (1,80,5325)
Model B: (1,80,5325) -> Conv(80->60,5) -> MaxPool -> Conv(60->60,5) -> MaxPool
         -> (1,60,589)
Host: InstanceNorm(80)+leaky after A, InstanceNorm(60)+leaky after B,
      InstanceNorm(60)+leaky at the end.
"""
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from paths import EXPORT_DIR, NUM_SAMPLES, SEG_DIR

torch.set_grad_enabled(False)

EPS = 1e-5


def build():
    from pyannote.audio import Model
    seg = Model.from_pretrained(str(SEG_DIR)).eval()
    sn = seg.sincnet
    w_sinc = sn.conv1d[0].filterbank.filters().detach().numpy()  # (80, 1, 251)

    class PartA(nn.Module):
        def __init__(self):
            super().__init__()
            self.w0 = nn.Parameter(torch.from_numpy(w_sinc[:, 0, :].T).contiguous())
            self.pool = sn.pool1d[0]

        def forward(self, unfolded):            # (1, 15975, 251)
            y = torch.matmul(unfolded, self.w0).transpose(1, 2)
            y = torch.abs(y)
            return self.pool(y)                 # (1, 80, 5325)

    class PartB1(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = sn.conv1d[1]
            self.pool1 = sn.pool1d[1]

        def forward(self, x):                   # (1, 80, 5325)
            return self.pool1(self.conv1(x))    # (1, 60, 1773)

    class PartB2(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv2 = sn.conv1d[2]
            self.pool2 = sn.pool1d[2]

        def forward(self, x):                   # (1, 60, 1773)
            return self.pool2(self.conv2(x))    # (1, 60, 589)

    part_a = PartA()
    part_b1 = PartB1()
    part_b2 = PartB2()

    # ---- validate against the original sincnet on real audio ----
    import soundfile as sf
    wav, sr = sf.read('/data/shared/huyuan/vad_seg_sr_cluster/alimeeting/audio_mono/'
                      'R8001_M8004_MS801.wav', dtype='float32', always_2d=False)
    chunk = wav[:NUM_SAMPLES].astype(np.float32)
    x = torch.from_numpy(chunk)[None, None]
    ref = sn(x)                                          # (1, 60, 589)

    xn = sn.wav_norm1d(x)[0, 0].numpy()
    n = 1 + (NUM_SAMPLES - 251) // 10
    cols = np.lib.stride_tricks.as_strided(xn, shape=(n, 251), strides=(40, 4)).copy()

    a = part_a(torch.from_numpy(cols)[None])
    a = F.leaky_relu(sn.norm1d[0](a))
    b1 = part_b1(a)
    b1 = F.leaky_relu(sn.norm1d[1](b1))
    b2 = part_b2(b1)
    b2 = F.leaky_relu(sn.norm1d[2](b2))
    print(f'split sincnet vs original: maxdiff {(b2 - ref).abs().max().item():.3e}')

    # host-side params: wav_norm + 3 InstanceNorms
    np.savez(EXPORT_DIR / 'host_frontend.npz',
             norm_w=sn.wav_norm1d.weight.detach().numpy(),
             norm_b=sn.wav_norm1d.bias.detach().numpy(),
             n0_w=sn.norm1d[0].weight.detach().numpy(),
             n0_b=sn.norm1d[0].bias.detach().numpy(),
             n1_w=sn.norm1d[1].weight.detach().numpy(),
             n1_b=sn.norm1d[1].bias.detach().numpy(),
             n2_w=sn.norm1d[2].weight.detach().numpy(),
             n2_b=sn.norm1d[2].bias.detach().numpy(),
             kernel=251, stride=10, n_out=n, eps=EPS)

    torch.onnx.export(part_a, torch.from_numpy(cols)[None],
                      str(EXPORT_DIR / 'segmentation_sincnet.onnx'),
                      input_names=['unfolded'], output_names=['pool0'],
                      dynamic_axes=None, opset_version=17, do_constant_folding=True)
    torch.onnx.export(part_b1, a,
                      str(EXPORT_DIR / 'segmentation_convs1.onnx'),
                      input_names=['x'], output_names=['pool1'],
                      dynamic_axes=None, opset_version=17, do_constant_folding=True)
    torch.onnx.export(part_b2, b1,
                      str(EXPORT_DIR / 'segmentation_convs2.onnx'),
                      input_names=['x'], output_names=['pool2'],
                      dynamic_axes=None, opset_version=17, do_constant_folding=True)
    print('exported 3 parts: sincnet (A) + convs1 (B1) + convs2 (B2)')


if __name__ == '__main__':
    build()
