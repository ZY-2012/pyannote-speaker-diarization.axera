#!/usr/bin/env python3
"""Export pyannote community-1 segmentation + embedding models to ONNX.

    segmentation.onnx : waveform (1,1,160000) -> powerset log-probs (1,589,7)
    embedding.onnx    : waveform (1,1,160000) + weights (1,589) -> xvector (1,256)

Both are static graphs (no temporal state), exported at training input size
(10 s @ 16 kHz mono) which is exactly how the pyannote pipeline calls them.

Self-check: torch vs onnxruntime cosine on random + real calibration audio.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from paths import (EMB_DIR, EMB_DIM, EMB_ONNX, EXPORT_DIR, NUM_SAMPLES, SEG_CLASSES,
                   SEG_DIR, SEG_FRAMES, SEG_ONNX, WAV)

torch.set_grad_enabled(False)


def load_models():
    from pyannote.audio import Model
    seg = Model.from_pretrained(str(SEG_DIR))
    emb = Model.from_pretrained(str(EMB_DIR))
    seg.eval(); emb.eval()
    return seg, emb


def export_segmentation(seg):
    """Split at the LSTM boundary: Pulsar2's NPU scheduler hits Python's
    recursion limit when the 4-layer BiLSTM is lowered with the long SincNet
    conv chain feeding it. Two axmodels, chained at runtime:
        A  waveform (1,1,160000) -> sincnet features (1,60,589)
        B  features (589,1,60)   -> powerset log-probs (589,1,256)

    The SincNet first conv has a single input channel, which the AX650 NPU
    backend computes WRONG (verified with check=2 xrun: 8% mismatch). Workaround:
    zero-pad the input to 8 channels and zero-pad the conv weights accordingly --
    mathematically identical, all NPU convs become multi-channel.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    # materialized first-conv sinc filters (80, 1, 251) from the filterbank
    w_sinc = seg.sincnet.conv1d[0].filterbank.filters().detach().numpy()
    assert w_sinc.shape == (80, 1, 251), w_sinc.shape

    # The SincNet first conv has a single input channel, which the AX650 NPU
    # backend computes WRONG (check=2 xrun mismatch at any length; C>=2 is fine).
    # Workaround: zero-pad the input to 2 channels and zero-pad the conv weights
    # accordingly -- mathematically identical, minimal MAC overhead.
    PAD_CHANNELS = 2

    class SincNetPart(nn.Module):
        def __init__(self, sincnet):
            super().__init__()
            self.wav_norm1d = sincnet.wav_norm1d
            self.pool1d = sincnet.pool1d
            self.norm1d = sincnet.norm1d
            self.conv1 = sincnet.conv1d[1]
            self.conv2 = sincnet.conv1d[2]
            padded = np.zeros((80, PAD_CHANNELS, 251), dtype=np.float32)
            padded[:, 0, :] = w_sinc[:, 0, :]
            self.conv0 = nn.Conv1d(PAD_CHANNELS, 80, 251, stride=10)
            with torch.no_grad():
                self.conv0.weight.copy_(torch.from_numpy(padded))
                self.conv0.bias.zero_()

        def forward(self, waveform):                    # (1, 1, 160000)
            x = self.wav_norm1d(waveform)
            x = F.pad(x, (0, 0, 0, PAD_CHANNELS - 1))   # (1, 2, 160000)
            x = torch.abs(self.conv0(x))
            x = F.leaky_relu(self.norm1d[0](self.pool1d[0](x)))
            x = F.leaky_relu(self.norm1d[1](self.pool1d[1](self.conv1(x))))
            x = F.leaky_relu(self.norm1d[2](self.pool1d[2](self.conv2(x))))
            return x

    # Pulsar2's ONNX parser rejects LSTM nodes with batch_first layout, so the
    # LSTM is exported seq-first with explicit transposes (identical math).
    hp = dict(seg.hparams.lstm)
    lstm_sf = nn.LSTM(60, hp['hidden_size'], num_layers=hp['num_layers'],
                      bidirectional=hp['bidirectional'], batch_first=False)
    lstm_sf.load_state_dict(seg.lstm.state_dict())

    class LstmPart(nn.Module):
        def __init__(self, lstm, linear, classifier, activation):
            super().__init__()
            self.lstm = lstm
            self.linear = linear
            self.classifier = classifier
            self.activation = activation

        def forward(self, features):            # (589, 1, 60) seq-first layout
            outputs, _ = self.lstm(features)    # (589, 1, 256)
            outputs = outputs.transpose(0, 1)   # (1, 589, 256) batch-first
            for linear in self.linear:
                outputs = torch.nn.functional.leaky_relu(linear(outputs))
            return self.activation(self.classifier(outputs))

    part_a = SincNetPart(seg.sincnet)
    part_b = LstmPart(lstm_sf, seg.linear, seg.classifier, seg.activation)

    x = torch.randn(1, 1, NUM_SAMPLES)
    with torch.inference_mode():
        feat = seg.sincnet(x)
        t_full = seg(x)
    a = part_a(x)
    b = part_b(feat.transpose(1, 2).transpose(0, 1))  # (1,589,60) -> (589,1,60)
    assert torch.allclose(a, feat, atol=1e-6)
    assert torch.allclose(b, t_full, atol=1e-5), (b - t_full).abs().max()
    print(f'seg split: A {tuple(a.shape)}  B {tuple(b.shape)}  '
          f'(vs full maxdiff {(b-t_full).abs().max().item():.2e})')

    torch.onnx.export(part_a, x, str(EXPORT_DIR / 'segmentation_sincnet.onnx'),
                      input_names=['waveform'], output_names=['features'],
                      dynamic_axes=None, opset_version=17, do_constant_folding=True)
    torch.onnx.export(part_b, feat.transpose(1, 2).transpose(0, 1),
                      str(EXPORT_DIR / 'segmentation_lstm.onnx'),
                      input_names=['features'], output_names=['log_probs'],
                      dynamic_axes=None, opset_version=17, do_constant_folding=True)
    print(f'exported export/segmentation_sincnet.onnx + export/segmentation_lstm.onnx')


def export_embedding(emb):
    import torch.nn as nn

    class EmbExport(nn.Module):
        """fbank computed on host (numpy port); NPU runs ResNet only."""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, fbank, weights):
            return self.model.resnet(fbank, weights=weights)[1]

    wrapper = EmbExport(emb)
    fbank = torch.randn(1, 998, 80)
    w = torch.rand(1, SEG_FRAMES)
    y = wrapper(fbank, w)
    assert tuple(y.shape) == (1, EMB_DIM), y.shape
    torch.onnx.export(
        wrapper, (fbank, w), str(EMB_ONNX),
        input_names=['fbank', 'weights'], output_names=['embedding'],
        dynamic_axes=None, opset_version=17,
        do_constant_folding=True,
    )
    print(f'exported {EMB_ONNX}  {tuple(y.shape)}')


def save_fbank_params():
    """Extract mel filterbank + hamming window from torchaudio (bit-exact)."""
    import torch
    from torchaudio.compliance.kaldi import get_mel_banks

    window_size = int(16000 * 25 * 0.001)      # 400
    window_shift = int(16000 * 10 * 0.001)     # 160
    padded = 1
    while padded < window_size:
        padded *= 2
    mel, _ = get_mel_banks(80, padded, 16000, 20.0, 0.0, 100.0, -500.0, 1.0)
    mel = torch.nn.functional.pad(mel, (0, 1), mode='constant', value=0)  # (80, 513)
    window = torch.hamming_window(window_size, periodic=False, alpha=0.54, beta=0.46)
    out = EXPORT_DIR / 'fbank_params.npz'
    np.savez(out,
             mel_banks=mel.numpy(), window=window.numpy(),
             window_size=window_size, window_shift=window_shift,
             padded_window_size=padded, sample_rate=16000)
    print(f'wrote {out}  mel {tuple(mel.shape)} win {window_size}')


def check_onnx():
    import onnxruntime as ort
    import soundfile as sf

    seg_a = ort.InferenceSession(str(EXPORT_DIR / 'segmentation_sincnet.onnx'),
                                 providers=['CPUExecutionProvider'])
    seg_b = ort.InferenceSession(str(EXPORT_DIR / 'segmentation_lstm.onnx'),
                                 providers=['CPUExecutionProvider'])
    emb = ort.InferenceSession(str(EMB_ONNX), providers=['CPUExecutionProvider'])

    wav, sr = sf.read(str(WAV), dtype='float32', always_2d=False)
    assert sr == 16000
    seg_model, emb_model = load_models()

    results = {}
    for name, chunk in (('random', None), ('real', wav[:NUM_SAMPLES])):
        if chunk is None:
            chunk = np.random.randn(NUM_SAMPLES).astype(np.float32)
        weights = np.random.rand(1, SEG_FRAMES).astype(np.float32)
        x = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0)
        with torch.inference_mode():
            t_seg = seg_model(x)
            fbank_t = emb_model.compute_fbank(x)          # (1, 998, 80)
            t_emb = emb_model.resnet(fbank_t, weights=torch.from_numpy(weights))[1]
        o_a = seg_a.run(None, {'waveform': chunk[None, None]})[0]
        o_b = seg_b.run(None, {'features': o_a.transpose(2, 0, 1)})[0]
        o_emb = emb.run(None, {'fbank': fbank_t.numpy(), 'weights': weights})[0]
        results[name] = {
            'seg_cos': float(torch.cosine_similarity(
                torch.from_numpy(o_b.ravel()).float(), t_seg.ravel().float(), dim=0)),
            'seg_maxdiff': float((np.abs(o_b - t_seg.numpy())).max()),
            'emb_cos': float(torch.cosine_similarity(
                torch.from_numpy(o_emb).float(), t_emb.float(), dim=1)[0]),
            'emb_maxdiff': float(np.abs(o_emb - t_emb.numpy()).max()),
        }
        print(name, results[name])
    return results


def check_numpy_fbank():
    """numpy fbank port vs torchaudio: compare resulting embeddings."""
    import onnxruntime as ort
    import soundfile as sf
    import torch
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent
                             / 'python' / 'community1_sdk'))
    from fbank import kaldi_fbank, load_fbank_params

    emb = ort.InferenceSession(str(EMB_ONNX), providers=['CPUExecutionProvider'])
    _, emb_model = load_models()
    params = load_fbank_params(EXPORT_DIR / 'fbank_params.npz')

    wav, sr = sf.read(str(WAV), dtype='float32', always_2d=False)
    chunk = wav[:NUM_SAMPLES]
    x = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0)

    fb_np = kaldi_fbank(chunk, params)
    with torch.inference_mode():
        fb_t = emb_model.compute_fbank(x)

    diff = np.abs(fb_np - fb_t[0].numpy())
    print(f'fbank numpy vs torchaudio: maxdiff {diff.max():.2e}  mean {diff.mean():.2e}')

    w = np.random.rand(1, SEG_FRAMES).astype(np.float32)
    e_np = emb.run(None, {'fbank': fb_np[None], 'weights': w})[0]
    e_t = emb.run(None, {'fbank': fb_t.numpy(), 'weights': w})[0]
    cos = float(torch.cosine_similarity(torch.from_numpy(e_np).float(),
                                        torch.from_numpy(e_t).float(), dim=1)[0])
    print(f'embedding via numpy-fbank vs torch-fbank: cosine {cos:.7f}  '
          f'maxdiff {np.abs(e_np - e_t).max():.2e}')
    return {'fbank_maxdiff': float(diff.max()), 'emb_cos': cos}


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    seg, emb = load_models()
    export_segmentation(seg)
    export_embedding(emb)
    save_fbank_params()
    results = check_onnx()
    results['numpy_fbank'] = check_numpy_fbank()

    meta = {
        'models': [
            {'name': 'segmentation_sincnet', 'onnx': 'export/segmentation_sincnet.onnx',
             'input': [{'name': 'waveform', 'shape': [1, 1, NUM_SAMPLES],
                        'dtype': 'float32', 'range': [-1.0, 1.0]}],
             'output': [{'name': 'features', 'shape': [1, 60, SEG_FRAMES]}],
             'note': 'SincNet front-end only; split from the LSTM because Pulsar2 '
                     'NPU backend overflows Python recursion when lowering the '
                     '4-layer BiLSTM fed by the long conv chain'},
            {'name': 'segmentation_lstm', 'onnx': 'export/segmentation_lstm.onnx',
             'input': [{'name': 'features', 'shape': [SEG_FRAMES, 1, 60],
                        'note': 'seq-first layout (host transposes the sincnet output)'}],
             'output': [{'name': 'log_probs', 'shape': [1, SEG_FRAMES, SEG_CLASSES],
                         'note': 'powerset log-probs (log-softmax); host does argmax + '
                                 'mapping to 3 local-speaker columns'}],
             'frames_per_10s': SEG_FRAMES},
            {'name': 'embedding', 'onnx': 'export/embedding.onnx',
             'input': [{'name': 'fbank', 'shape': [1, 998, 80], 'dtype': 'float32',
                        'note': 'numpy-port kaldi fbank, globally mean-centered '
                                '(see export/fbank_params.npz)'},
                       {'name': 'weights', 'shape': [1, SEG_FRAMES],
                        'dtype': 'float32',
                        'note': 'speaker activity mask (0/1), nearest-interpolated '
                                'internally to fbank frames'}],
             'output': [{'name': 'embedding', 'shape': [1, EMB_DIM]}],
             'dimension': EMB_DIM},
        ],
        'onnx_parity': results,
        'source': 'pyannote/speaker-diarization-community-1',
        'torch_version': torch.__version__,
    }
    with open(EXPORT_DIR / 'model_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print('wrote export/model_meta.json')


if __name__ == '__main__':
    main()
