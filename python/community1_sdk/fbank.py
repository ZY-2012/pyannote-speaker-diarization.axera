"""NumPy port of torchaudio.compliance.kaldi.fbank (deterministic settings).

Matches the community-1 WeSpeaker frontend exactly:
    num_mel_bins=80, frame_length=25ms, frame_shift=10ms, round_to_power_of_two=True,
    snip_edges=True, dither=0.0, window_type='hamming', use_energy=False,
    sample_frequency=16000, preemphasis=0.97, remove_dc_offset=True,
    raw_energy=True, use_log_fbank=True, use_power=True, htk_compat=False,
    subtract_mean=False + global mean centering (fbank_centering_span=None).

The mel filterbank and hamming window are taken from torchaudio at export time
(export/fbank_params.npz) so they are bit-identical; only the FFT backend
(numpy pocketfft vs torch cuFFT) may differ at ~1e-7.
"""
from pathlib import Path

import numpy as np

EPS = np.float32(np.finfo(np.float32).eps)  # 1.1920929e-07


def load_fbank_params(npz_path):
    z = np.load(npz_path)
    return {
        'mel_banks': z['mel_banks'],     # (80, 513)
        'window': z['window'],           # (400,)
        'window_size': int(z['window_size']),
        'window_shift': int(z['window_shift']),
        'padded_window_size': int(z['padded_window_size']),
        'sample_rate': int(z['sample_rate']),
    }


def kaldi_fbank(wav, params, center=True):
    """wav: float32 (num_samples,) in [-1, 1] -> fbank (m, 80) float32"""
    wav = wav.astype(np.float32)
    wav = wav * (1 << 15)

    window_size = params['window_size']
    window_shift = params['window_shift']
    padded = params['padded_window_size']

    n = wav.shape[0]
    if n < window_size:
        raise ValueError(f'audio too short ({n} < {window_size})')
    m = 1 + (n - window_size) // window_shift

    # framing (torch as_strided equivalent)
    strided = np.lib.stride_tricks.as_strided(
        wav, shape=(m, window_size), strides=(window_shift * 4, 4)).copy()

    # remove_dc_offset
    strided = strided - strided.mean(axis=1, keepdims=True)

    # preemphasis with replicate padding on the left
    shifted = np.concatenate([strided[:, :1], strided[:, :-1]], axis=1)
    strided = strided - 0.97 * shifted

    # hamming window
    strided = strided * params['window'].astype(np.float32)[None, :]

    # zero-pad to padded_window_size
    if padded != window_size:
        strided = np.pad(strided, ((0, 0), (0, padded - window_size)))

    # rfft -> power spectrum (fbank keeps the DC bin as-is)
    fft = np.fft.rfft(strided, axis=1)
    power = np.maximum(np.square(np.abs(fft)), EPS).astype(np.float32)

    # mel filterbank
    mel_energies = power @ params['mel_banks'].T   # (m, 80)
    mel_energies = np.log(np.maximum(mel_energies, EPS)).astype(np.float32)

    if center:
        mel_energies = mel_energies - mel_energies.mean(axis=0, keepdims=True)
    return mel_energies.astype(np.float32)


if __name__ == '__main__':
    import sys
    wav_path, npz_path = sys.argv[1], sys.argv[2]
    import soundfile as sf
    wav, sr = sf.read(wav_path, dtype='float32', always_2d=False)
    params = load_fbank_params(npz_path)
    fb = kaldi_fbank(wav[:sr * 10], params)
    print('fbank shape', fb.shape, fb.dtype)
    print(fb[0, :5], '...', 'min', fb.min(), 'max', fb.max())
