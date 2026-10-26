from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from .graph import Graph


def transmitter_polarity(row, typ):
    nt = (row["known_nt"] or row["top_nt"]).lower()
    transmitters = {s.strip() for s in nt.split(",")}
    if typ.startswith(("R1", "R7", "R8")):
        return -1, "histamine_by_photoreceptor_identity", False
    uncertain = "glutamate" in transmitters or not transmitters.intersection({"gaba", "acetylcholine"})
    if not row["known_nt"] and float(row.get("top_nt_conf") or 0) < .5:
        uncertain = True
    return (-1 if transmitters.intersection({"gaba", "glutamate"}) else 1), nt, uncertain


def coalesce_edges(edges):
    """Sum parallel rows before applying a per-neuron-pair synapse threshold."""
    if not len(edges):
        return edges.copy()
    ordered = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    start = np.r_[True, np.any(ordered[1:, :2] != ordered[:-1, :2], axis=1)]
    locations = np.flatnonzero(start)
    result = ordered[locations].copy()
    result[:, 2] = np.add.reduceat(ordered[:, 2], locations)
    return result


def prepare(raw: Path, out: Path, crop=(0.30, 0.70), min_synapses=5):
    """Anatomical selection only: central retina, its outputs, LC11/10a and their inputs."""
    t = time.perf_counter()
    with gzip.open(raw / "neuron_annotations_flywire_v2.1.0.tsv.gz", "rt", encoding="utf-8") as f:
        annotations = {int(r["root_id"]): r for r in csv.DictReader(f, delimiter="\t")}
    with (raw / "right_visual_positions_all_neurons.csv").open(encoding="utf-8") as f:
        retina = list(csv.DictReader(f))
    coords = np.array([[float(r["x_axis"]), float(r["y_axis"])] for r in retina])
    coord_min, coord_span = coords.min(0), np.ptp(coords, axis=0)
    norm = (coords - coord_min) / coord_span
    inside = np.all((norm >= crop[0]) & (norm <= crop[1]), axis=1)
    input_ids = np.array([int(r["root_id"]) for r, ok in zip(retina, inside) if ok], dtype=np.uint64)
    r7 = np.array([i for i, r in enumerate(retina) if r["cell_type"].startswith("R7")])
    if len(r7) < 10 or len(input_ids) < 10:
        raise ValueError("Incomplete retinal geometry")
    # The deposit's inferred ommatidia use R7 seeds. Use a fixed Voronoi average.
    _, nearest_r7 = cKDTree(norm[r7]).query(norm[inside])
    input_xy = (norm[r7[nearest_r7]] - crop[0]) / (crop[1] - crop[0])
    input_xy = np.clip(input_xy, 0, 1)
    target_ids = np.array([rid for rid, r in annotations.items() if r["side"] == "right"
                           and (r["hemibrain_type"] in ("LC11", "LC10a")
                                or r["cell_type"] in ("LC11", "LC10a"))], dtype=np.uint64)
    optic_ids = np.array([rid for rid, r in annotations.items() if r["side"] == "right"
                         and r["super_class"] in ("optic", "visual_projection", "visual_centrifugal")],
                        dtype=np.uint64)
    print(f"Loading version-pinned graph; inputs={len(input_ids)}, LC targets={len(target_ids)}", flush=True)
    with gzip.open(raw / "connections_biological.csv.gz", "rt") as f:
        edges = np.loadtxt(f, delimiter=",", skiprows=1, dtype=np.uint64)
    if edges.ndim != 2 or edges.shape[1] != 3:
        raise ValueError("Unexpected graph columns")
    pre, post, count = edges.T
    input_edges = coalesce_edges(edges[np.isin(pre, input_ids)])
    target_edges = coalesce_edges(edges[np.isin(post, target_ids)])
    retina_out = np.unique(input_edges[input_edges[:, 2] >= min_synapses, 1])
    target_in = np.unique(target_edges[target_edges[:, 2] >= min_synapses, 0])
    selected = np.unique(np.concatenate([input_ids, target_ids,
                         np.intersect1d(retina_out, optic_ids), np.intersect1d(target_in, optic_ids)]))
    raw_rows = len(edges)
    touches_selection = np.isin(pre, selected) | np.isin(post, selected)
    touching_rows = int(touches_selection.sum())
    relevant = coalesce_edges(edges[touches_selection])
    pre, post, count = relevant.T
    strong = count >= min_synapses
    pre_in, post_in = np.isin(pre, selected), np.isin(post, selected)
    retained = strong & pre_in & post_in & (pre != post)
    ep, eq = pre[retained], post[retained]
    counts = count[retained].astype(np.float32)
    types, signs, uncertain, glutamate_ids = [], [], [], []
    nt_counts = {}
    for rid in selected:
        row = annotations[int(rid)]
        typ = row["cell_type"] or row["hemibrain_type"] or "untyped"
        types.append(typ)
        sign, nt, is_uncertain = transmitter_polarity(row, typ)
        if is_uncertain:
            uncertain.append(str(rid))
        if "glutamate" in nt.split(", "):
            glutamate_ids.append(str(rid))
        signs.append(sign)
        nt_counts[nt] = nt_counts.get(nt, 0) + 1
    source_manifest = json.loads((raw / "download_manifest.json").read_text())
    report = {
        "kind": "empirical_connectome", "connectome": "FlyWire FAFB v783",
        "processed_deposit": "https://doi.org/10.5281/zenodo.21549559",
        "upstream_connectivity": "https://doi.org/10.5281/zenodo.10676866",
        "annotations_version": "v2.1.0", "license": "CC-BY-4.0",
        "selection_rule": "Right-eye central crop inputs, direct optic outputs, all right LC11/LC10a, direct optic inputs to those types; induced graph",
        "crop_normalized_eye_axes": list(crop), "minimum_synapses_per_pair": min_synapses,
        "selection_uses_task_performance": False,
        "neurons": len(selected), "edges": len(ep), "retinal_inputs": len(input_ids),
        "ommatidial_seeds_used": len(np.unique(input_xy, axis=0)),
        "lc11_neurons": types.count("LC11"), "lc10a_neurons": types.count("LC10a"),
        "raw_global_edge_rows": raw_rows,
        "raw_touching_edge_rows": touching_rows,
        "unique_touching_pairs": len(relevant),
        "parallel_rows_merged_touching_selection": touching_rows - len(relevant),
        "parallel_row_policy": "Sum syn_count for each ordered neuron pair before threshold; source describes grouped data but contains parallel rows. Upstream biological reconciliation remains a confirmation-stage check.",
        "excluded_neurons": len(annotations) - len(selected),
        "boundary_in_edges_ge_threshold": int((strong & ~pre_in & post_in).sum()),
        "boundary_out_edges_ge_threshold": int((strong & pre_in & ~post_in).sum()),
        "internal_below_threshold_edges": int((~strong & pre_in & post_in).sum()),
        "removed_internal_self_edges": int((strong & pre_in & post_in & (pre == post)).sum()),
        "transmitter_counts": nt_counts, "uncertain_sign_neuron_ids": uncertain,
        "glutamate_neuron_ids": glutamate_ids,
        "input_mapping": "Fixed pixel Voronoi averages using deposited R7-seed projected coordinates; grayscale drive to all selected retinal types",
        "mapping_limitation": "PCA eye-plane reconstruction, not independently calibrated angular retinotopy; cropped subcircuit omits boundary drive",
        "lc11_biological_interpretation_approved": False,
        "source_files": source_manifest["files"], "preparation_seconds": time.perf_counter() - t,
    }
    graph = Graph(selected, np.searchsorted(selected, ep).astype(np.int32),
                  np.searchsorted(selected, eq).astype(np.int32), counts,
                  np.array(signs, np.int8), np.searchsorted(selected, input_ids).astype(np.int32),
                  input_xy.astype(np.float32), np.array(types), report)
    graph.save(out)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Full inclusion/exclusion ledger makes the anatomical boundary reviewable.
    with out.with_name("neuron_selection.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["root_id", "included", "side", "cell_type", "hemibrain_type"])
        chosen = set(map(int, selected))
        for rid, r in annotations.items():
            w.writerow([rid, int(rid in chosen), r["side"], r["cell_type"], r["hemibrain_type"]])
    print(f"Prepared {graph.n:,} neurons / {len(graph.pre):,} edges in {report['preparation_seconds']:.1f}s", flush=True)
    return graph
