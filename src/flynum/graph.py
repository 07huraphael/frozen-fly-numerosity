from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse


@dataclass
class Graph:
    ids: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    count: np.ndarray
    signs: np.ndarray
    input_nodes: np.ndarray
    input_xy: np.ndarray
    cell_types: np.ndarray
    provenance: dict

    def __post_init__(self):
        if len(self.pre) != len(self.post) or len(self.pre) != len(self.count):
            raise ValueError("Mismatched edge arrays")
        if len(self.signs) != len(self.ids) or len(self.cell_types) != len(self.ids):
            raise ValueError("Mismatched neuron arrays")
        if len(self.pre):
            if min(self.pre.min(), self.post.min()) < 0 or max(self.pre.max(), self.post.max()) >= self.n:
                raise ValueError("Edge index outside neuron array")
            code = self.pre.astype(np.int64) * self.n + self.post
            if len(np.unique(code)) != len(code):
                raise ValueError("Parallel edge rows must be coalesced before constructing a simple graph")
        if not np.isfinite(self.count).all() or np.any(self.count <= 0):
            raise ValueError("Weights must be finite and positive before polarity assignment")

    @property
    def n(self):
        return len(self.ids)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: v for k, v in vars(self).items() if k != "provenance"},
                            provenance=json.dumps(self.provenance))

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as d:
            return cls(**{k: d[k] for k in d.files if k != "provenance"},
                       provenance=json.loads(str(d["provenance"])))

    def fingerprint(self):
        h = hashlib.sha256()
        for v in (self.ids, self.pre, self.post, self.count, self.signs, self.input_nodes, self.input_xy):
            h.update(np.ascontiguousarray(v).tobytes())
        return h.hexdigest()

    def operator(self, gain=0.9, recurrent=True):
        # Rows are postsynaptic, columns presynaptic: W @ x is incoming drive.
        weights = self.count.astype(np.float64) * self.signs[self.pre]
        w = sparse.csr_matrix((weights, (self.post, self.pre)), shape=(self.n, self.n))
        strength = np.asarray(abs(w).sum(axis=1)).ravel()
        w = sparse.diags(1 / np.maximum(strength, 1)) @ w
        w = (w * gain).astype(np.float32).tocsr()
        if not recurrent:
            w = sparse.csr_matrix(w.shape, dtype=np.float32)
        return w, {"raw_abs_weight_sum": float(abs(weights).sum()),
                   "raw_weight_mean": float(abs(weights).mean()) if len(weights) else 0,
                   "normalized_abs_row_sum_max": float(np.asarray(abs(w).sum(axis=1)).max()),
                   "normalized_frobenius": float(np.sqrt((w.data ** 2).sum()))}


def rewire(graph: Graph, seed: int, swaps_per_edge=5, min_changed=0.7):
    """Simple directed double swaps, retaining source weights and signed degrees."""
    rng = np.random.default_rng(seed)
    pre, post = graph.pre.copy(), graph.post.copy()
    m, n = len(pre), graph.n
    codes = set((pre.astype(np.int64) * n + post).tolist())
    original = codes.copy()
    target, accepted, attempts = int(swaps_per_edge * m), 0, 0
    limit = max(1000, target * 50)
    while accepted < target and attempts < limit:
        candidates = rng.integers(m, size=(min(100000, max(1000, target - accepted)), 2))
        for i, j in candidates:
            attempts += 1
            u, v, x, y = int(pre[i]), int(post[i]), int(pre[j]), int(post[j])
            if u == x or v == y or u == y or x == v or graph.signs[u] != graph.signs[x]:
                continue
            a, b = u * n + y, x * n + v
            if a in codes or b in codes:
                continue
            codes.remove(u * n + v); codes.remove(x * n + y)
            codes.add(a); codes.add(b)
            post[i], post[j] = y, v
            accepted += 1
            if accepted >= target or attempts >= limit:
                break
    changed = 1 - len(codes & original) / max(m, 1)
    if accepted < target or changed < min_changed:
        raise RuntimeError(f"Insufficient rewiring: {accepted}/{target} swaps, changed={changed:.3f}")
    for values, before in ((pre, graph.pre), (post, graph.post)):
        if not np.array_equal(np.bincount(values, minlength=n), np.bincount(before, minlength=n)):
            raise AssertionError("Rewiring changed a directed degree")
    return replace(graph, pre=pre, post=post), {
        "seed": seed, "accepted_swaps": accepted, "attempted_swaps": attempts,
        "changed_edge_fraction": changed, "directed_degrees_preserved": True,
        "source_signed_weights_preserved": True, "signed_in_degrees_preserved": True,
        "in_strengths_preserved": False,
    }


def random_graph_like(graph: Graph, seed: int):
    rng = np.random.default_rng(seed)
    # Bijective coding of all possible non-self edges.
    code = rng.choice(graph.n * (graph.n - 1), size=len(graph.pre), replace=False)
    pre = code // (graph.n - 1)
    post = code % (graph.n - 1)
    post = post + (post >= pre)
    return replace(graph, pre=pre.astype(np.int32), post=post.astype(np.int32),
                   count=rng.permutation(graph.count))


def fixture(seed=1, n=240):
    """Spatial synthetic graph for software QA only, with no LC11 interpretation."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 1, (n, 2))
    distance = ((xy[:, None] - xy[None]) ** 2).sum(2)
    prob = 0.05 + 0.20 * np.exp(-distance / 0.04)
    mask = rng.random((n, n)) < prob
    np.fill_diagonal(mask, False)
    pre, post = np.nonzero(mask)
    inputs = np.arange(n // 3)
    return Graph(np.arange(n), pre.astype(np.int32), post.astype(np.int32),
                 rng.integers(1, 15, len(pre)).astype(np.float32),
                 np.where(rng.random(n) < 0.2, -1, 1).astype(np.int8), inputs, xy[inputs],
                 np.full(n, "synthetic", dtype="U16"), {"kind": "synthetic_fixture", "seed": seed})
