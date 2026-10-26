from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np


@dataclass
class Stimuli:
    images: np.ndarray
    labels: np.ndarray
    conditions: np.ndarray
    metadata: list[dict]


@dataclass
class Pairs:
    a: np.ndarray
    b: np.ndarray
    labels_a: np.ndarray
    labels_b: np.ndarray
    conditions: np.ndarray
    groups: np.ndarray
    reverse_index: np.ndarray

    @property
    def labels(self):
        return self.labels_a + self.labels_b


def make_dot_image(n: int, condition: str, rng: np.random.Generator, size: int = 48):
    """Nonoverlapping antialiased discs. Labels are never passed to an encoder."""
    if not 0 <= n <= 10:
        raise ValueError("Numerosity must be in [0, 10]")
    contrast = float(rng.uniform(0.65, 0.95))
    extent = 0.72
    radius = float(rng.uniform(0.035, 0.045))
    if condition == "fixed_area":
        radius = math.sqrt(0.048 / (math.pi * max(n, 1)))
    elif condition == "fixed_extent":
        extent = 0.53  # fixed allowed centre range; realized hull is measured, not asserted equal
    elif condition == "cue_conflict":
        radius = math.sqrt(0.070 / (math.pi * max(n, 1) ** 1.65))
        contrast = 0.85
    elif condition == "ood":
        radius = float(rng.uniform(0.025, 0.030))
        contrast = float(rng.uniform(0.30, 0.48))
        extent = 0.83
    elif condition != "fixed_radius":
        raise ValueError(f"Unknown condition: {condition}")
    low, high = max((1 - extent) / 2, radius + 0.012), min((1 + extent) / 2, 1 - radius - 0.012)
    centers = []
    for i in range(n):
        for _ in range(10000):
            if condition == "ood":
                # Held-out near-grid layout, jittered independently on every trial.
                grid = rng.integers(0, 4, size=2)
                candidate = low + (high - low) * (grid + rng.uniform(0.15, 0.85, 2)) / 4
            else:
                candidate = rng.uniform(low, high, 2)
            if all(np.linalg.norm(candidate - c) > 2 * radius + 0.012 for c in centers):
                centers.append(candidate)
                break
        else:
            raise RuntimeError(f"Could not place {n} discs in {condition}")
    supersample = 4
    ss = size * supersample
    canvas = np.zeros((ss, ss), dtype=np.float32)
    axis = (np.arange(ss) + 0.5) / ss
    for cx, cy in centers:
        xidx = np.flatnonzero(abs(axis - cx) <= radius)
        yidx = np.flatnonzero(abs(axis - cy) <= radius)
        disk = ((axis[yidx, None] - cy) ** 2 + (axis[None, xidx] - cx) ** 2 <= radius ** 2)
        canvas[np.ix_(yidx, xidx)] = np.maximum(canvas[np.ix_(yidx, xidx)], disk * contrast)
    image = canvas.reshape(size, supersample, size, supersample).mean((1, 3))
    occupied = np.argwhere(image > 0)
    bbox = np.ptp(occupied, axis=0) / size if len(occupied) else np.zeros(2)
    meta = {
        "count": n, "condition": condition, "radius": radius, "contrast": contrast,
        "geometric_area": float(n * math.pi * radius ** 2),
        "pixel_mass": float(image.sum()), "occupied_pixels": int((image > 0).sum()),
        "bbox_x": float(bbox[1]), "bbox_y": float(bbox[0]),
        "center_extent": extent, "centers": np.asarray(centers).tolist(),
        "sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }
    return image, meta


def singles(seed: int, conditions: list[str], repeats: int, size: int) -> Stimuli:
    rng = np.random.default_rng(seed)
    images, labels, conds, metadata = [], [], [], []
    for condition in conditions:
        for n in range(11):
            for _ in range(repeats):
                image, meta = make_dot_image(n, condition, rng, size)
                images.append(image)
                labels.append(n)
                conds.append(condition)
                metadata.append(meta)
    order = rng.permutation(len(images))
    return Stimuli(np.stack(images)[order], np.array(labels)[order], np.array(conds)[order],
                   [metadata[i] for i in order])


def pairs(seed: int, conditions: list[str], repeats: int, size: int) -> Pairs:
    rng = np.random.default_rng(seed)
    aa, bb, na, nb, conds, groups, reverse = [], [], [], [], [], [], []
    for condition in conditions:
        for a in range(6):
            for b in range(a, 6):
                for rep in range(repeats):
                    ia, _ = make_dot_image(a, condition, rng, size)
                    ib, _ = make_dot_image(b, condition, rng, size)
                    idx = len(aa)
                    aa.append(ia); bb.append(ib); na.append(a); nb.append(b)
                    conds.append(condition); groups.append(f"{a}+{b}")
                    reverse.append(idx if a == b else idx + 1)
                    if a != b:
                        aa.append(ib); bb.append(ia); na.append(b); nb.append(a)
                        conds.append(condition); groups.append(f"{a}+{b}"); reverse.append(idx)
    return Pairs(np.stack(aa), np.stack(bb), np.array(na), np.array(nb),
                 np.array(conds), np.array(groups), np.array(reverse))


def cue_features(images: np.ndarray) -> np.ndarray:
    """Only measured pixels; no labels, radii, centres, or component counts."""
    n, h, w = images.shape
    mass = images.sum((1, 2))
    safe = np.maximum(mass, 1e-8)
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    cx = (images * xx).sum((1, 2)) / safe
    cy = (images * yy).sum((1, 2)) / safe
    vx = (images * (xx[None] - cx[:, None, None]) ** 2).sum((1, 2)) / safe
    vy = (images * (yy[None] - cy[:, None, None]) ** 2).sum((1, 2)) / safe
    return np.column_stack([mass, (images > 0).sum((1, 2)), images.max((1, 2)),
                            images.std((1, 2)), cx, cy, vx, vy]).astype(np.float32)


def check_leakage(*sets: Stimuli) -> dict:
    nonzero_sets = [{m["sha256"] for m in s.metadata if m["count"] > 0} for s in sets]
    overlap = sum(len(a & b) for i, a in enumerate(nonzero_sets) for b in nonzero_sets[i + 1:])
    if overlap:
        raise ValueError(f"Found {overlap} nonzero images shared across splits")
    return {"nonzero_cross_split_duplicates": overlap,
            "zero_images": "Identical blank images are unavoidable and explicitly exempted."}
