#!/usr/bin/env python3
"""Generate calibration samples for Pulsar2 MinMax quantization.

Both models are static (10 s window), so calibration = real-world windows:
  - waveform : 10 s crops from AMI + AliMeeting meetings (16 kHz mono float32)
  - fbank    : numpy-port kaldi fbank of those crops (validated vs torchaudio)
  - weights  : realistic 0/1 speaker masks (uniform random 0..1 thresholded,
               plus all-zero and all-one patterns, plus hard masks from the
               GPU segmentation output when available)

64 samples per input, spread across meetings so the value range is covered.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'python' / 'community1_sdk'))
from fbank import kaldi_fbank, load_fbank_params  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CALIB_DIR, EXPORT_DIR, NUM_SAMPLES, SAMPLE_RATE, SEG_FRAMES  # noqa: E402

N_SAMPLES = 64
WAVS = [
    '/data/shared/huyuan/vad_seg_sr_cluster/ami_eval/audio/EN2002d.Mix-Headset.wav',
    '/data/shared/huyuan/vad_seg_sr_cluster/ami_eval/audio/ES2004a.Mix-Headset.wav',
    '/data/shared/huyuan/vad_seg_sr_cluster/alimeeting/audio_mono/R8001_M8004_MS801.wav',
    '/data/shared/huyuan/vad_seg_sr_cluster/alimeeting/audio_mono/R8007_M8010_MS803.wav',
]


def load_wavs():
    import soundfile as sf
    out = []
    for p in WAVS:
        wav, sr = sf.read(p, dtype='float32', always_2d=False)
        assert sr == SAMPLE_RATE, (p, sr)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        out.append(wav.astype(np.float32))
    return out


def main():
    params = load_fbank_params(EXPORT_DIR / 'fbank_params.npz')
    wavs = load_wavs()

    rng = np.random.RandomState(0)
    dirs = {name: CALIB_DIR / name for name in ('waveform', 'fbank', 'weights', 'features')}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob('*.npy'):
            old.unlink()

    # sincnet features come from the FP32 torch model (real intermediate values)
    import torch
    from pyannote.audio import Model
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from paths import SEG_DIR
    seg = Model.from_pretrained(str(SEG_DIR)).eval()

    # evenly spread windows over the concatenated corpus
    corpus = np.concatenate(wavs)
    total = len(corpus)
    starts = np.linspace(0, total - NUM_SAMPLES - 1, N_SAMPLES).round().astype(int)

    for i, s in enumerate(starts):
        chunk = corpus[s:s + NUM_SAMPLES].astype(np.float32)
        fb = kaldi_fbank(chunk, params)
        # realistic speaker masks: overlap up to 2 speakers, activity ~10-90%
        rho = rng.uniform(0.05, 0.95)
        mask = (rng.rand(SEG_FRAMES) < rho).astype(np.float32)
        pattern = i % 6
        if pattern == 1:
            mask[:] = 0.0
        elif pattern == 2:
            mask[:] = 1.0
        elif pattern == 3:
            mask = (np.sin(np.linspace(0, 30, SEG_FRAMES)) > 0).astype(np.float32)
        elif pattern == 4:
            mask = (np.linspace(0, 1, SEG_FRAMES) > 0.5).astype(np.float32)
        elif pattern == 5:
            mask = (np.linspace(0, 1, SEG_FRAMES) < 0.5).astype(np.float32)
        with torch.inference_mode():
            feat = seg.sincnet(torch.from_numpy(chunk[None, None])).numpy()
        np.save(dirs['waveform'] / f'{i:04d}.npy', chunk[None, None])
        np.save(dirs['fbank'] / f'{i:04d}.npy', fb[None])
        np.save(dirs['weights'] / f'{i:04d}.npy', mask[None])
        np.save(dirs['features'] / f'{i:04d}.npy', feat.transpose(2, 0, 1))  # (589,1,60)
        if i % 16 == 0:
            print(f'{i}/{N_SAMPLES} @ {s/SAMPLE_RATE:.0f}s')

    print(f'generated {N_SAMPLES} samples per input in {CALIB_DIR}')


if __name__ == '__main__':
    main()
