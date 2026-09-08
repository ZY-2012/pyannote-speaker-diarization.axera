"""Inference backend abstraction: axengine on the board, onnxruntime on the host.

Usage:
    from community1_sdk.axwrap import open_session
    sess = open_session('models/segmentation_sincnet.axmodel')
    out = sess.run({'waveform': x})   # dict of numpy arrays
"""
from __future__ import annotations

import numpy as np


class Session:
    def __init__(self, model_path):
        self.model_path = str(model_path)
        self._backend = 'onnxruntime'
        import onnxruntime as ort
        self._sess = ort.InferenceSession(str(model_path),
                                          providers=['CPUExecutionProvider'])

    def run(self, inputs: dict) -> dict:
        out = self._sess.run(None, inputs)
        names = [o.name for o in self._sess.get_outputs()]
        return dict(zip(names, out))

    @property
    def input_names(self):
        return [i.name for i in self._sess.get_inputs()]


class AxSession(Session):
    def __init__(self, model_path):
        self.model_path = str(model_path)
        self._backend = 'axengine'
        import axengine as axe
        self._sess = axe.InferenceSession(str(model_path),
                                          providers=['AxEngineExecutionProvider'])

    def run(self, inputs: dict) -> dict:
        out = self._sess.run(None, {k: np.ascontiguousarray(v, dtype=np.float32)
                                    for k, v in inputs.items()})
        names = [o.name for o in self._sess.get_outputs()]
        return dict(zip(names, out))


def open_session(model_path, prefer_ax=True):
    if prefer_ax and str(model_path).endswith('.axmodel'):
        try:
            return AxSession(model_path)
        except ImportError:
            pass
    return Session(model_path)
