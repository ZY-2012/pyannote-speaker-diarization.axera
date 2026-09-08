"""Host-side FFN head + powerset decoding for the segmentation model.

    lstm_out (589, 256) -> leaky_relu(FC128) x2 -> classifier (7)
    -> log-softmax -> argmax -> one-hot -> powerset mapping (7 -> 3)
    -> hard 0/1 multi-label (589, 3)

Weights: export/ffn_params.npz, export/powerset_mapping.npz.
"""
from __future__ import annotations

import numpy as np

LEAKY = 0.01


class Ffn:
    def __init__(self, ffn_path, mapping_path):
        z = np.load(ffn_path)
        self.l0_w = z['l0_w'].astype(np.float32)   # (256, 128)
        self.l0_b = z['l0_b'].astype(np.float32)
        self.l1_w = z['l1_w'].astype(np.float32)   # (128, 128)
        self.l1_b = z['l1_b'].astype(np.float32)
        self.c_w = z['c_w'].astype(np.float32)     # (128, 7)
        self.c_b = z['c_b'].astype(np.float32)
        self.mapping = np.load(mapping_path)['mapping'].astype(np.float32)  # (7, 3)

    def forward(self, x):
        """x: (589, 256) -> hard (589, 3) uint8"""
        h = np.matmul(x, self.l0_w) + self.l0_b
        h = np.where(h > 0, h, h * LEAKY)
        h = np.matmul(h, self.l1_w) + self.l1_b
        h = np.where(h > 0, h, h * LEAKY)
        logits = np.matmul(h, self.c_w) + self.c_b       # (589, 7)
        hard = np.eye(7, dtype=np.float32)[np.argmax(logits, axis=1)]  # (589, 7)
        return np.matmul(hard, self.mapping).astype(np.uint8)          # (589, 3)
