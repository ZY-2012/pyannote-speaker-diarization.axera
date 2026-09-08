"""community-1 speaker diarization pipeline (board version).

Replicates pyannote.audio 4.0.7 SpeakerDiarization.apply exactly, with the
segmentation SincNet and the embedding ResNet running on NPU axmodels and the
LSTM / FFN / fbank / clustering on the CPU (numpy/scipy).

Stages: slide segmentation -> powerset decoding (hard 3-speaker masks)
-> speaker_count -> per-(chunk, local speaker) embeddings -> AHC + PLDA + VBx
clustering -> constrained assignment -> discrete diarization -> RTTM.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .fbank import kaldi_fbank, load_fbank_params
from .vbx import cluster_vbx

SAMPLE_RATE = 16000
WINDOW_SEC = 10.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)      # 160000
STEP_SEC = 1.0
STEP_SAMPLES = int(SAMPLE_RATE * STEP_SEC)          # 16000
SEG_FRAMES = 589
FRAME_STEP = 0.016875
FRAME_DUR = 0.0619375
NUM_LOCAL_SPEAKERS = 3
EMB_DIM = 256
EMB_BATCH = 32
EMB_MIN_SAMPLES = 400
MIN_ACTIVE_RATIO = 0.2


def closest_frame(t):
    return int(np.rint((t - 0.5 * FRAME_DUR) / FRAME_STEP))


class Pipeline:
    def __init__(self, seg_a, seg_b1, seg_b2, host_params, lstm_params, ffn_params,
                 powerset_map, emb_model, emb_tail, fbank_params, transform_npz,
                 plda_npz, threshold=0.6, Fa=0.07, Fb=0.8, num_speakers=None,
                 lstm_cells=None, step_sec=1.0):
        from .axwrap import open_session
        from .lstm import BidirLstm, CellLstm
        from .ffn import Ffn
        from .vbx import PLDA
        self.seg_a = open_session(seg_a)
        self.seg_b1 = open_session(seg_b1)
        self.seg_b2 = open_session(seg_b2)
        self.hp = np.load(host_params)
        if lstm_cells is not None:
            self.lstm = CellLstm(lstm_cells)
        else:
            self.lstm = BidirLstm(lstm_params)
        self.ffn = Ffn(ffn_params, powerset_map)
        self.emb = open_session(emb_model)
        self.emb_tail = np.load(emb_tail)
        self.fbank_params = load_fbank_params(fbank_params)
        self.plda = PLDA(transform_npz, plda_npz)
        self.threshold = threshold
        self.Fa = Fa
        self.Fb = Fb
        self.num_speakers = num_speakers          # None = auto (VBx estimate)
        self.min_num_frames = int(np.ceil(
            SEG_FRAMES * EMB_MIN_SAMPLES / WINDOW_SAMPLES))
        self.step_sec = step_sec
        self.step_samples = int(SAMPLE_RATE * step_sec)

    @staticmethod
    def _inst_norm(x, w, b):
        mu = x.mean(axis=2, keepdims=True)
        sd = np.sqrt(np.square(x - mu).mean(axis=2, keepdims=True) + 1e-5)
        return (x - mu) / sd * w.reshape(1, -1, 1) + b.reshape(1, -1, 1)

    @staticmethod
    def _leaky(x):
        return np.where(x > 0, x, 0.01 * x).astype(np.float32)

    # ---- segmentation ----
    def _segment_chunk(self, wav_chunk):
        """wav_chunk (160000,) float32 -> hard (589, 3) uint8"""
        hp = self.hp
        # host: instance-norm'd waveform + im2col
        xn = ((wav_chunk - wav_chunk.mean())
              / np.sqrt(wav_chunk.var() + 1e-5) * hp['norm_w'] + hp['norm_b'])
        cols = np.lib.stride_tricks.as_strided(
            xn, shape=(15975, 251), strides=(40, 4)).copy()
        # NPU A: matmul + abs + maxpool
        a = self.seg_a.run({'unfolded': cols[None].astype(np.float32)})
        a = a[list(a)[0]]
        a = self._leaky(self._inst_norm(a, hp['n0_w'], hp['n0_b']))
        # NPU B1: conv(80->60) + maxpool
        b1 = self.seg_b1.run({'x': a})
        b1 = b1[list(b1)[0]]
        b1 = self._leaky(self._inst_norm(b1, hp['n1_w'], hp['n1_b']))
        # NPU B2: conv(60->60) + maxpool
        b2 = self.seg_b2.run({'x': b1})
        b2 = b2[list(b2)[0]]
        feat = self._leaky(self._inst_norm(b2, hp['n2_w'], hp['n2_b']))
        x = feat[0].transpose(1, 0)                  # (589, 60)
        x = self.lstm.forward(x)                     # (589, 256)
        return self.ffn.forward(x)                   # (589, 3) uint8

    def get_segmentations(self, wav):
        """wav (N,) float32 -> (num_chunks, 589, 3) uint8 hard masks"""
        num_samples = int(wav.shape[0])
        n_full = (1 + (num_samples - WINDOW_SAMPLES) // self.step_samples
                  if num_samples >= WINDOW_SAMPLES else 0)
        chunks = [wav[c * self.step_samples:c * self.step_samples + WINDOW_SAMPLES]
                  for c in range(n_full)]
        rem = ((num_samples - WINDOW_SAMPLES) % self.step_samples
               if num_samples >= WINDOW_SAMPLES else num_samples)
        if num_samples < WINDOW_SAMPLES or rem > 0:
            ch = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
            part = wav[n_full * self.step_samples:]
            ch[:len(part)] = part
            chunks.append(ch)
        data = np.stack([self._segment_chunk(c) for c in chunks])
        return data

    # ---- speaker count ----
    def speaker_count(self, seg_data):
        C = seg_data.shape[0]
        counts = np.sum(seg_data, axis=2, keepdims=True).astype(np.float32)
        t_end = WINDOW_SEC + (C - 1) * self.step_sec + 0.5 * FRAME_DUR
        T = closest_frame(t_end) + 1
        agg = np.zeros((T, 1), dtype=np.float32)
        n_contrib = np.zeros((T, 1), dtype=np.float32)
        for c in range(C):
            s_frame = closest_frame(c * self.step_sec + 0.5 * FRAME_DUR)
            agg[s_frame:s_frame + SEG_FRAMES] += counts[c]
            n_contrib[s_frame:s_frame + SEG_FRAMES] += 1
        avg = agg / np.maximum(n_contrib, 1e-12)
        return np.rint(avg).astype(np.uint8)         # (T, 1)

    # ---- embeddings ----
    def get_embeddings(self, wav, seg_data):
        """-> (num_chunks, 3, 256) float32"""
        C = seg_data.shape[0]
        clean = (np.sum(seg_data, axis=2, keepdims=True) < 2).astype(np.float32)
        clean_seg = seg_data * clean
        seg1_b = self.emb_tail['seg1_b']                    # constant for all-zero masks

        embs = np.empty((C, NUM_LOCAL_SPEAKERS, EMB_DIM), dtype=np.float32)
        for c in range(C):
            s = c * self.step_samples
            crop = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
            part = wav[s:s + WINDOW_SAMPLES]
            crop[:len(part)] = part
            fbank = None                                     # computed lazily, shared
            for spk in range(NUM_LOCAL_SPEAKERS):
                mask = seg_data[c, :, spk].astype(np.float32)
                cmask = clean_seg[c, :, spk].astype(np.float32)
                used = cmask if cmask.sum() > self.min_num_frames else mask
                if used.sum() == 0.0:
                    embs[c, spk] = seg1_b
                    continue
                if fbank is None:
                    fbank = kaldi_fbank(crop, self.fbank_params)
                out = self.emb.run({'fbank': fbank[None]})
                frames = out[list(out)[0]]                  # (1, 256, 10, 125)
                embs[c, spk] = self._pool_tail(frames, used)
        return embs

    def _pool_tail(self, frames, weights):
        z = self.emb_tail
        n_frames = int(z['n_frames'])
        n_weights = int(z['n_weights'])
        feats = int(z['feats'])
        x = frames.reshape(1, feats, n_frames)              # (1, 2560, 125)
        idx = (np.arange(n_frames) * n_weights / n_frames).astype(np.int64)
        wi = weights[idx][None, None, :]                    # (1, 1, 125)
        v1 = wi.sum(axis=2) + 1e-8
        mean = np.sum(x * wi, axis=2) / v1
        dx2 = np.square(x - mean[:, :, None])
        v2 = np.square(wi).sum(axis=2)
        var = np.sum(dx2 * wi, axis=2) / (v1 - v2 / v1 + 1e-8)
        std = np.sqrt(var)
        stats = np.concatenate([mean, std], axis=1)         # (1, 5120)
        return (stats @ z['seg1_w'].T + z['seg1_b'])[0]

    # ---- clustering (AHC + PLDA + VBx) ----
    def cluster(self, embeddings, seg_data):
        C = seg_data.shape[0]
        train, chunk_idx, speaker_idx = self._filter_embeddings(embeddings, seg_data)

        if train.shape[0] < 2:
            hard = np.zeros((C, NUM_LOCAL_SPEAKERS), dtype=np.int8)
            soft = np.ones((C, NUM_LOCAL_SPEAKERS, 1), dtype=np.float32)
            centroids = np.mean(train, axis=0, keepdims=True) if train.size else \
                np.zeros((1, EMB_DIM), dtype=np.float32)
            return hard, soft, centroids

        train_norm = train / np.linalg.norm(train, axis=1, keepdims=True)
        dendro = linkage(train_norm, method='centroid', metric='euclidean')
        ahc = fcluster(dendro, self.threshold, criterion='distance') - 1
        _, ahc = np.unique(ahc, return_inverse=True)

        fea = self.plda(train)
        q, sp = cluster_vbx(ahc, fea, self.plda.phi, Fa=self.Fa, Fb=self.Fb,
                            maxIters=20)

        W = q[:, sp > 1e-7]
        centroids = W.T @ train / W.sum(0, keepdims=True).T

        auto_k = centroids.shape[0]
        num_clusters = self.num_speakers
        if auto_k < 1:
            num_clusters = 1
        if num_clusters and num_clusters != auto_k:
            from sklearn.cluster import KMeans
            kc = KMeans(n_clusters=num_clusters, n_init=3, random_state=42,
                        copy_x=False).fit_predict(train_norm)
            centroids = np.vstack([np.mean(train[kc == k], axis=0)
                                   for k in range(num_clusters)])

        e2k = cdist(embeddings.reshape(-1, EMB_DIM), centroids, metric='cosine')
        soft = (2 - e2k).reshape(C, NUM_LOCAL_SPEAKERS, -1)

        constrained = True
        if num_clusters and num_clusters != auto_k:
            constrained = False
        if constrained:
            const = soft.min() - 1.0
            inactive = np.sum(seg_data, axis=1) == 0     # (C, 3)
            soft[inactive] = const
            hard = self._constrained_argmax(soft)
        else:
            hard = np.argmax(soft, axis=2)
        return hard.astype(np.int8), soft, centroids

    @staticmethod
    def _filter_embeddings(embeddings, seg_data):
        _, num_frames, _ = seg_data.shape
        single = (np.sum(seg_data, axis=2, keepdims=True) == 1)
        num_clean = np.sum(seg_data * single, axis=1)        # (C, 3)
        active = num_clean >= MIN_ACTIVE_RATIO * num_frames
        valid = ~np.any(np.isnan(embeddings), axis=2)
        chunk_idx, speaker_idx = np.where(active * valid)
        return embeddings[chunk_idx, speaker_idx], chunk_idx, speaker_idx

    @staticmethod
    def _constrained_argmax(soft):
        soft = np.nan_to_num(soft, nan=np.nanmin(soft))
        C, S, K = soft.shape
        hard = -2 * np.ones((C, S), dtype=np.int8)
        for c in range(C):
            speakers, clusters = linear_sum_assignment(soft[c], maximize=True)
            for s, k in zip(speakers, clusters):
                hard[c, s] = k
        return hard

    # ---- reconstruction ----
    def reconstruct(self, seg_data, hard_clusters, count):
        C, num_frames, _ = seg_data.shape
        K = int(np.max(hard_clusters)) + 1
        clustered = np.nan * np.zeros((C, num_frames, K))
        for c in range(C):
            cluster = hard_clusters[c]
            segmentation = seg_data[c]
            for k in np.unique(cluster):
                if k == -2:
                    continue
                clustered[c, :, k] = np.max(segmentation[:, cluster == k], axis=1)
        # to_diarization: aggregate (sum) + top-count threshold
        t_end = WINDOW_SEC + (C - 1) * self.step_sec + 0.5 * FRAME_DUR
        T = closest_frame(t_end) + 1
        act = np.zeros((T, K), dtype=np.float32)
        for c in range(C):
            s_frame = closest_frame(c * self.step_sec + 0.5 * FRAME_DUR)
            act[s_frame:s_frame + num_frames] += np.nan_to_num(clustered[c])
        max_per_frame = int(np.max(count))
        if K < max_per_frame:
            act = np.pad(act, ((0, 0), (0, max_per_frame - K)))
        order = np.argsort(-act, axis=-1)
        binary = np.zeros_like(act)
        for t in range(T):
            for i in range(int(count[t, 0])):
                binary[t, order[t, i]] = 1.0
        return binary, count

    # ---- RTTM ----
    @staticmethod
    def to_rttm(binary, uri):
        """binary (T, K) -> list of (speaker, start, dur) segments"""
        T, K = binary.shape
        timestamps = [i * FRAME_STEP + 0.5 * FRAME_DUR for i in range(T + 1)]
        segments = []
        for k in range(K):
            act = binary[:, k] > 0.5
            starts = [t for t in range(T) if act[t] and (t == 0 or not act[t - 1])]
            ends = [t for t in range(T) if act[t] and (t == T - 1 or not act[t + 1])]
            for a, b in zip(starts, ends):
                segments.append((k, timestamps[a], timestamps[b + 1]))
        segments.sort(key=lambda s: (s[1], s[2]))
        merged = []
        for seg in segments:
            if merged and merged[-1][0] == seg[0] and seg[1] - merged[-1][2] <= 0.0:
                merged[-1] = (merged[-1][0], merged[-1][1],
                              max(merged[-1][2], seg[2]))
            else:
                merged.append(seg)
        return [(f'SPEAKER_{k:02d}', start, end - start)
                for k, start, end in merged if end - start > 0]


def run_pipeline(pipeline, wav, uri='audio'):
    seg_data = pipeline.get_segmentations(wav)
    count = pipeline.speaker_count(seg_data)
    if np.max(count) == 0:
        return []
    embeddings = pipeline.get_embeddings(wav, seg_data)
    hard, soft, centroids = pipeline.cluster(embeddings, seg_data)
    count = count.astype(np.float32)
    count = np.minimum(count, np.inf)
    binary, count = pipeline.reconstruct(seg_data, hard, count)
    return pipeline.to_rttm(binary, uri)
