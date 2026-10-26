"""Reconcile the actual model and anatomical selection against canonical FlyWire data."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow
import pyarrow.feather as feather

from flynum.graph import Graph
from flynum.prepare import coalesce_edges


def main():
    raw = Path("data/raw/flywire_783/proofread_connections_783.feather")
    expected_md5 = "f48f972d262323a102aed49af1396b8a"
    h = hashlib.md5()
    with raw.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    assert h.hexdigest() == expected_md5, "Canonical source checksum failed"
    graph = Graph.load(Path("data/processed/flywire_visual.npz"))
    print("Reading canonical pre/post/synapse-count columns", flush=True)
    table = feather.read_table(raw, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"],
                               memory_map=True, use_threads=False)
    rows = []
    for batch in table.to_batches(max_chunksize=500000):
        pre, post, count = [batch.column(i).to_numpy(zero_copy_only=False) for i in range(3)]
        keep = np.isin(pre, graph.ids) | np.isin(post, graph.ids)
        if keep.any():
            # Cast each column first: stacking uint64 IDs with signed counts can promote to float64.
            rows.append(np.column_stack([pre[keep].astype(np.uint64), post[keep].astype(np.uint64),
                                         count[keep].astype(np.uint64)]))
    relevant = coalesce_edges(np.concatenate(rows))
    strong = relevant[:, 2] >= graph.provenance["minimum_synapses_per_pair"]
    induced = relevant[strong & np.isin(relevant[:, 0], graph.ids) & np.isin(relevant[:, 1], graph.ids)
                       & (relevant[:, 0] != relevant[:, 1])]
    np.testing.assert_array_equal(induced[:, 0], graph.ids[graph.pre])
    np.testing.assert_array_equal(induced[:, 1], graph.ids[graph.post])
    np.testing.assert_array_equal(induced[:, 2], graph.count)
    with gzip.open("data/raw/zenodo_21549559/neuron_annotations_flywire_v2.1.0.tsv.gz", "rt", encoding="utf-8") as f:
        annotations = list(csv.DictReader(f, delimiter="\t"))
    optic = np.array([int(r["root_id"]) for r in annotations if r["side"] == "right" and
                     r["super_class"] in ("optic", "visual_projection", "visual_centrifugal")], dtype=np.uint64)
    target = graph.ids[np.isin(graph.cell_types, ["LC11", "LC10a"])]
    retina = graph.ids[graph.input_nodes]
    out = np.unique(relevant[strong & np.isin(relevant[:, 0], retina), 1])
    incoming = np.unique(relevant[strong & np.isin(relevant[:, 1], target), 0])
    selected = np.unique(np.concatenate([retina, target, np.intersect1d(out, optic), np.intersect1d(incoming, optic)]))
    np.testing.assert_array_equal(selected, graph.ids)
    report = {"passed": True, "canonical_record": "https://zenodo.org/records/10676866",
              "canonical_file": raw.name, "md5": h.hexdigest(),
              "sha256": raw.with_suffix(".sha256").read_text().strip(),
              "canonical_table_rows": table.num_rows, "pyarrow_version": pyarrow.__version__,
              "canonical_touching_rows": sum(len(r) for r in rows),
              "canonical_unique_touching_pairs": len(relevant),
              "model_neurons": graph.n, "model_edges": len(graph.pre),
              "all_model_edge_endpoints_match": True, "all_model_synapse_counts_match": True,
              "anatomical_neighbor_selection_matches": True, "graph_sha256": graph.fingerprint(),
              "interpretation": "Coalescing deposited parallel rows reproduces the canonical FlyWire model subgraph and its anatomical selection exactly."}
    Path("data/processed/canonical_reconciliation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if Path("results/pilot").exists():
        Path("results/pilot/canonical_reconciliation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
