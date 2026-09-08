"""Direct port of pyannote.audio utils/vbx.py + core/plda.py (Apache-2.0/MIT).

Pure numpy/scipy, runs on the board CPU.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.special import logsumexp, softmax


def VBx(X, Phi, Fa=1.0, Fb=1.0, pi=10, gamma=None, maxIters=10, epsilon=1e-4,
        alphaQInit=1.0, ref=None, plot=False, return_model=False,
        alpha=None, invL=None):
    D = X.shape[1]
    if type(pi) is int:
        pi = np.ones(pi) / pi
    if gamma is None:
        gamma = np.random.gamma(alphaQInit, size=(X.shape[0], len(pi)))
        gamma = gamma / gamma.sum(1, keepdims=True)
    assert gamma.shape[1] == len(pi) and gamma.shape[0] == X.shape[0]
    G = -0.5 * (np.sum(X ** 2, axis=1, keepdims=True) + D * np.log(2 * np.pi))
    V = np.sqrt(Phi)
    rho = X * V
    Li = []
    ELBO = None
    for ii in range(maxIters):
        if ii > 0 or alpha is None or invL is None:
            invL = 1.0 / (1 + Fa / Fb * gamma.sum(axis=0, keepdims=True).T * Phi)
            alpha = Fa / Fb * invL * gamma.T.dot(rho)
        log_p_ = Fa * (rho.dot(alpha.T) - 0.5 * (invL + alpha ** 2).dot(Phi) + G)
        eps = 1e-8
        lpi = np.log(pi + eps)
        log_p_x = logsumexp(log_p_ + lpi, axis=-1)
        log_pX_ = np.sum(log_p_x, axis=0)
        gamma = np.exp(log_p_ + lpi - log_p_x[:, None])
        pi = np.sum(gamma, axis=0)
        pi = pi / pi.sum()
        ELBO = log_pX_ + Fb * 0.5 * np.sum(np.log(invL) - invL - alpha ** 2 + 1)
        Li.append([ELBO])
        if ii > 0 and ELBO - Li[-2][0] < epsilon:
            break
    return (gamma, pi, Li) + ((alpha, invL) if return_model else ())


def cluster_vbx(ahc_init, fea, Phi, Fa, Fb, maxIters=20, init_smoothing=7.0):
    qinit = np.zeros((len(ahc_init), ahc_init.max() + 1))
    qinit[range(len(ahc_init)), ahc_init.astype(int)] = 1.0
    qinit = qinit if init_smoothing < 0 else softmax(qinit * init_smoothing, axis=1)
    gamma, pi, _, _, _ = VBx(fea, Phi, Fa=Fa, Fb=Fb, pi=qinit.shape[1],
                             gamma=qinit, maxIters=maxIters, return_model=True)
    return gamma, pi


def l2_norm(vec_or_matrix):
    if len(vec_or_matrix.shape) == 1:
        return vec_or_matrix / np.linalg.norm(vec_or_matrix)
    elif len(vec_or_matrix.shape) == 2:
        return vec_or_matrix / np.linalg.norm(vec_or_matrix, axis=1, ord=2)[:, np.newaxis]
    raise ValueError('Wrong number of dimensions, 1 or 2 is supported, not %i.'
                     % len(vec_or_matrix.shape))


def vbx_setup(transform_npz, plda_npz):
    x = np.load(transform_npz)
    mean1, mean2, lda = x['mean1'], x['mean2'], x['lda']
    p = np.load(plda_npz)
    plda_mu, plda_tr, plda_psi = p['mu'], p['tr'], p['psi']
    W = np.linalg.inv(plda_tr.T.dot(plda_tr))
    B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
    acvar, wccn = eigh(B, W)
    plda_psi = acvar[::-1]
    plda_tr = wccn.T[::-1]

    def xvec_tf(xv):
        return np.sqrt(lda.shape[1]) * l2_norm(
            lda.T.dot(np.sqrt(lda.shape[0]) * l2_norm(xv - mean1).T).T - mean2)

    def plda_tf(x0, lda_dim=lda.shape[1]):
        return (x0 - plda_mu).dot(plda_tr.T)[:, :lda_dim]

    return xvec_tf, plda_tf, plda_psi


class PLDA:
    def __init__(self, transform_npz, plda_npz, lda_dimension=128):
        self._xvec_tf, self._plda_tf, self._plda_psi = vbx_setup(transform_npz, plda_npz)
        self.lda_dimension = lda_dimension

    @property
    def phi(self):
        return self._plda_psi[:self.lda_dimension]

    def __call__(self, embeddings):
        return self._plda_tf(self._xvec_tf(embeddings), lda_dim=self.lda_dimension)
