from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np

from .cluster import Cluster
from .specs import EXEC_TYPES, DOMINANT_TYPES, K

H = W = 64
N_CH = 6


_PLAN_CACHE: Dict[Tuple[int, int, int, int], Tuple[np.ndarray, ...]] = {}


def _plan(r: int, c: int, h: int, w: int):
    key = (r, c, h, w)
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        yi = np.linspace(0, r - 1, h)
        xi = np.linspace(0, c - 1, w)
        y0 = np.floor(yi).astype(np.intp); y1 = np.minimum(y0 + 1, r - 1)
        x0 = np.floor(xi).astype(np.intp); x1 = np.minimum(x0 + 1, c - 1)
        plan = (y0, y1, (yi - y0).astype(np.float32)[:, None],
                x0, x1, (xi - x0).astype(np.float32)[None, :])
        _PLAN_CACHE[key] = plan
    return plan


def _resample(mat: np.ndarray, h: int, w: int) -> np.ndarray:
    out = np.empty((h, w), dtype=np.float32)
    return _resample_into(mat, h, w, out)


def _resample_into(mat: np.ndarray, h: int, w: int, dst: np.ndarray) -> np.ndarray:
    r, c = mat.shape
    if r == h and c == w:
        dst[...] = mat
        return dst
    y0, y1, wy, x0, x1, wx = _plan(r, c, h, w)
    if r == 1:
        row = mat[0]
        dst[...] = row[x0] * (1.0 - wx) + row[x1] * wx
        return dst
    a = mat[np.ix_(y0, x0)]; b = mat[np.ix_(y0, x1)]
    c_ = mat[np.ix_(y1, x0)]; d = mat[np.ix_(y1, x1)]
    a *= (1.0 - wx); b *= wx; a += b
    c_ *= (1.0 - wx); d *= wx; c_ += d
    a *= (1.0 - wy); c_ *= wy; a += c_
    dst[...] = a
    return dst


def resource_heatmap(cluster: Cluster, out: np.ndarray = None) -> np.ndarray:
    rem = np.stack([nd.remaining_rate() for nd in cluster.nodes], axis=0)
    rem = rem.T
    if out is None:
        out = np.empty((K, H, W), dtype=np.float32)
    for k in range(K):
        _resample_into(rem[k:k + 1, :], H, W, out[k])
    return out


def workload_heatmap(cluster: Cluster, out: np.ndarray = None) -> np.ndarray:
    occ = cluster.occupancy_by_class()
    chan = np.zeros((N_CH, occ.shape[0]), dtype=np.float32)

    for ci, e in enumerate(EXEC_TYPES):
        block = occ[:, ci * 7:(ci + 1) * 7]
        chan[0] += block[:, 0]
        chan[1] += block[:, 1] + block[:, 2]
        chan[2] += block[:, 3]
    chan[3] = occ[:, [4, 11, 18]].sum(axis=1)
    chan[4] = occ[:, [5, 12, 19]].sum(axis=1)
    chan[5] = occ.sum(axis=1) * 0.25
    denom = chan.max(axis=1, keepdims=True)
    chan /= np.where(denom > 0, denom, 1.0)
    if out is None:
        out = np.empty((N_CH, H, W), dtype=np.float32)
    for k in range(N_CH):
        _resample_into(chan[k:k + 1, :], H, W, out[k])
    return out


class TS_C2EHM:
    def __init__(self, window: int = 6):
        self.window = window
        self.r_seq: Deque[np.ndarray] = deque(maxlen=window)
        self.w_seq: Deque[np.ndarray] = deque(maxlen=window)

        self._sum_r = np.zeros((N_CH, H, W), dtype=np.float32)
        self._sum_w = np.zeros((N_CH, H, W), dtype=np.float32)
        self._obs = np.empty((2 * N_CH, H, W), dtype=np.float32)
        self._dlt = np.empty((N_CH, H, W), dtype=np.float32)
        self._cached: np.ndarray = None

    def update(self, cluster: Cluster) -> None:
        r = resource_heatmap(cluster)
        w = workload_heatmap(cluster)
        full = len(self.r_seq) >= self.window
        if full:
            self._sum_r -= self.r_seq[0]
            self._sum_w -= self.w_seq[0]
        self.r_seq.append(r)
        self.w_seq.append(w)
        self._sum_r += r
        self._sum_w += w
        self._cached = None

    def observation(self) -> np.ndarray:
        if self._cached is not None:
            return self._cached
        n = len(self.r_seq)
        if n == 0:
            self._obs.fill(0.0)
        else:
            np.divide(self._sum_r, n, out=self._obs[:N_CH])
            np.divide(self._sum_w, n, out=self._obs[N_CH:])

        self._cached = self._obs.copy()
        return self._cached

    def delta(self) -> np.ndarray:
        n = len(self.r_seq)
        if n < 2:
            self._dlt.fill(0.0)
            return self._dlt
        cur = self.r_seq[-1]

        np.subtract(self._sum_r, cur, out=self._dlt)
        self._dlt *= -1.0 / (n - 1)
        self._dlt += cur
        return self._dlt
