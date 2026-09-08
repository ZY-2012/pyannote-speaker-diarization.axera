"""LSTM implementations for the PyanNet 4-layer bidirectional LSTM.

Two backends, identical math (validated against each other on real audio):
  * BidirLstm        - pure numpy recurrence (reference)
  * CellLstm         - chunked cells on NPU axmodels (32-step unrolled graph,
                       both directions per call, FP32 state carry on host)
"""
from __future__ import annotations

import numpy as np

CHUNK = 64


class BidirLstm:
    def __init__(self, params_path):
        z = np.load(params_path)
        self.n_layers = int(z['n_layers'])
        self.hidden = None
        for k in z.files:
            if k.startswith('weight_hh_l0') and not k.endswith('reverse'):
                self.hidden = z[k].shape[1]
        assert self.hidden is not None
        self.W = []   # [layer][dir] -> (x@Wih.T + bih, Whh, bhh)
        for l in range(self.n_layers):
            layer = []
            for suffix in ('', '_reverse'):
                Wih = z[f'weight_ih_l{l}{suffix}'].astype(np.float32)   # (512, in)
                bih = z[f'bias_ih_l{l}{suffix}'].astype(np.float32)     # (512,)
                Whh = z[f'weight_hh_l{l}{suffix}'].astype(np.float32)   # (512, 128)
                bhh = z[f'bias_hh_l{l}{suffix}'].astype(np.float32)     # (512,)
                layer.append((Wih, bih, Whh, bhh))
            self.W.append(layer)

    @staticmethod
    def _dir(x, Wih, bih, Whh, bhh, hid):
        """x: (T, in) -> (T, hid) unidirectional LSTM."""
        hx = np.matmul(x, Wih.T) + bih          # (T, 512)
        h = np.zeros(hid, dtype=np.float32)
        c = np.zeros(hid, dtype=np.float32)
        out = np.empty((x.shape[0], hid), dtype=np.float32)
        for t in range(x.shape[0]):
            g = hx[t] + np.matmul(Whh, h) + bhh   # (512,)
            gi = 1.0 / (1.0 + np.exp(-g[0:hid]))
            gf = 1.0 / (1.0 + np.exp(-g[hid:2 * hid]))
            gg = np.tanh(g[2 * hid:3 * hid])
            go = 1.0 / (1.0 + np.exp(-g[3 * hid:]))
            c = gf * c + gi * gg
            h = go * np.tanh(c)
            out[t] = h
        return out

    def forward(self, x):
        """x: (589, 60) sincnet features -> (589, 256) concatenated directions."""
        for l in range(self.n_layers):
            (Wih, bih, Whh, bhh) = self.W[l][0]
            (Wih_r, bih_r, Whh_r, bhh_r) = self.W[l][1]
            # both directions in one batched pass
            T = x.shape[0]
            hx = np.matmul(x, Wih.T) + bih                        # (T, 512)
            hx_r = np.matmul(x[::-1].copy(), Wih_r.T) + bih_r
            W = np.stack([Whh, Whh_r])                            # (2, 512, hid)
            B = np.stack([bhh, bhh_r])                            # (2, 512)
            HX = np.stack([hx, hx_r])                             # (2, T, 512)
            hid = self.hidden
            h = np.zeros((2, hid), dtype=np.float32)
            c = np.zeros((2, hid), dtype=np.float32)
            out = np.empty((2, T, hid), dtype=np.float32)
            for t in range(T):
                g = HX[:, t] + np.matmul(W, h[:, :, None])[:, :, 0] + B   # (2,512)
                gi = 1.0 / (1.0 + np.exp(-g[:, :hid]))
                gf = 1.0 / (1.0 + np.exp(-g[:, hid:2 * hid]))
                gg = np.tanh(g[:, 2 * hid:3 * hid])
                go = 1.0 / (1.0 + np.exp(-g[:, 3 * hid:]))
                c = gf * c + gi * gg
                h = go * np.tanh(c)
                out[:, t] = h
            fwd = out[0]
            bwd = out[1][::-1]
            x = np.concatenate([fwd, bwd], axis=1)
        return x


class CellLstm:
    """Chunked NPU LSTM: 4 axmodels (lstm_cell_l{0..3}), 32-step chunks."""

    def __init__(self, cell_models):
        from .axwrap import open_session
        self.cells = [open_session(p) for p in cell_models]

    def forward(self, x):
        for l in range(4):
            in_dim = x.shape[1]
            T = x.shape[0]
            bwd_seq = x[::-1].copy()
            h = np.zeros((1, 256), dtype=np.float32)
            c = np.zeros((1, 256), dtype=np.float32)
            ys = []
            for s in range(0, T, CHUNK):
                chunk = x[s:s + CHUNK]
                bchunk = bwd_seq[s:s + CHUNK]
                xc = np.zeros((CHUNK, 2 * in_dim), dtype=np.float32)
                xc[:len(chunk), :in_dim] = chunk
                xc[:len(chunk), in_dim:] = bchunk
                out = self.cells[l].run({'x': xc, 'h0': h, 'c0': c})
                y, h, c = out[list(out)[0]], out[list(out)[1]], out[list(out)[2]]
                ys.append(y[:len(chunk)])
            yall = np.concatenate(ys, axis=0)
            x = np.concatenate([yall[:, :128], yall[:, 128:][::-1]], axis=1)
        return x
