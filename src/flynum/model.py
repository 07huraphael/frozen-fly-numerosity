from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.linalg import solve
from scipy.spatial import cKDTree

from .graph import Graph


class RetinaEncoder:
    def __init__(self, graph: Graph, size: int):
        centers, self.inverse = np.unique(graph.input_xy, axis=0, return_inverse=True)
        yy, xx = np.mgrid[0:size, 0:size]
        pixels = np.column_stack([(xx.ravel() + 0.5) / size, (yy.ravel() + 0.5) / size])
        _, nearest = cKDTree(centers).query(pixels)
        count = np.bincount(nearest, minlength=len(centers))
        self.pool = sparse.csr_matrix((1 / np.maximum(count[nearest], 1),
                        (nearest, np.arange(size * size))), shape=(len(centers), size * size))
        self.pool = self.pool.astype(np.float32)

    def __call__(self, images):
        pooled = (self.pool @ images.reshape(len(images), -1).T).T
        return pooled[:, self.inverse]


class Reservoir:
    def __init__(self, graph: Graph, dynamics: dict, batch_size=64, recurrent=True):
        self.graph = graph
        self.leak = dynamics["leak"]
        self.input_scale = dynamics["input_scale"]
        self.w, self.weight_diagnostics = graph.operator(dynamics["gain"], recurrent)
        self.batch_size = batch_size
        self.output_nodes = np.setdiff1d(np.arange(graph.n), graph.input_nodes)
        if not len(self.output_nodes):
            raise ValueError("No non-retinal readout neurons")
        self.last_activity = {}

    def states(self, a, b=None, gap=0, duration=4, post_steps=2):
        """Only the final state is returned. No reset between A and B."""
        results = []
        means, saturations = [], []
        for start in range(0, len(a), self.batch_size):
            end = min(start + self.batch_size, len(a))
            state = np.zeros((self.graph.n, end - start), dtype=np.float32)
            drive = np.zeros_like(state)

            def advance(input_values, steps):
                nonlocal state
                drive.fill(0)
                if input_values is not None:
                    drive[self.graph.input_nodes] = input_values.T * self.input_scale
                for _ in range(steps):
                    state = (1 - self.leak) * state + self.leak * np.tanh(self.w @ state + drive)

            advance(a[start:end], duration)
            if b is not None:
                advance(None, gap)
                advance(b[start:end], duration)
            advance(None, post_steps)
            means.append(float(np.mean(abs(state))))
            saturations.append(float(np.mean(abs(state) > 0.95)))
            results.append(state[self.output_nodes].T.copy())
        self.last_activity = {"final_mean_abs_activity": float(np.mean(means)),
                              "final_saturation_fraction": float(np.mean(saturations))}
        return np.concatenate(results)


@dataclass
class LinearReadout:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    alpha: float

    def predict(self, x):
        return ((x - self.mean) / self.scale) @ self.coef + self.intercept

    def fingerprint(self):
        h = hashlib.sha256()
        for x in (self.mean, self.scale, self.coef, np.array([self.intercept, self.alpha])):
            h.update(np.ascontiguousarray(x).tobytes())
        return h.hexdigest()

    def save(self, path):
        np.savez_compressed(path, **vars(self))


def fit_readout(x, y, alpha):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = x.mean(0)
    std = x.std(0)
    scale = np.where(std > 1e-7, std, 1)
    z = (x - mean) / scale
    intercept = float(y.mean())
    centered = y - intercept
    if x.shape[1] > x.shape[0]:
        gram = z @ z.T
        gram.flat[:: len(gram) + 1] += alpha
        coef = z.T @ solve(gram, centered, assume_a="pos", check_finite=False)
    else:
        gram = z.T @ z
        gram.flat[:: len(gram) + 1] += alpha
        coef = solve(gram, z.T @ centered, assume_a="pos", check_finite=False)
    return LinearReadout(mean, scale, coef, intercept, float(alpha))


def select_readout(train, y, validation, vy, alphas):
    models = [fit_readout(train, y, a) for a in alphas]
    losses = [float(np.mean(abs(m.predict(validation) - vy))) for m in models]
    return models[int(np.argmin(losses))], {str(a): v for a, v in zip(alphas, losses)}


def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite prediction")
    return {"n": len(y), "mae": float(np.mean(abs(y - prediction))),
            "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
            "exact_accuracy": float(np.mean(np.rint(prediction) == y)),
            "bias": float(np.mean(prediction - y))}


def integrated_inputs(a, b=None, duration=4, gap=0, post_steps=2, decay=1.0):
    """Generic leaky pixel integrator; uses the same one-set readout protocol."""
    state = np.zeros_like(a)
    for values, steps in ((a, duration), (None, gap if b is not None else 0),
                          (b, duration if b is not None else 0), (None, post_steps)):
        for _ in range(steps):
            state = decay * state
            if values is not None:
                state = state + values
    return state
