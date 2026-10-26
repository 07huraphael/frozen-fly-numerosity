"""Structural feasibility audit independent of any task predictions."""
from collections import deque
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from flynum.graph import Graph


def audit(path=Path("data/processed/flywire_visual.npz")):
    g = Graph.load(path)
    outgoing = sparse.csr_matrix((np.ones(len(g.pre)), (g.pre, g.post)), shape=(g.n, g.n))
    hops = np.full(g.n, -1, dtype=np.int32)
    hops[g.input_nodes] = 0
    queue = deque(map(int, g.input_nodes))
    while queue:
        u = queue.popleft()
        for v in outgoing.indices[outgoing.indptr[u]:outgoing.indptr[u + 1]]:
            if hops[v] < 0:
                hops[v] = hops[u] + 1
                queue.append(int(v))
    ncomp, labels = connected_components(outgoing, directed=True, connection="strong")
    indeg = np.bincount(g.post, minlength=g.n)
    outdeg = np.bincount(g.pre, minlength=g.n)
    result = {"neurons": g.n, "edges": len(g.pre), "graph_sha256": g.fingerprint(),
              "reachable_from_retina": int((hops >= 0).sum()),
              "unreachable_from_retina": int((hops < 0).sum()),
              "zero_indegree": int((indeg == 0).sum()), "zero_outdegree": int((outdeg == 0).sum()),
              "strong_components": ncomp, "largest_strong_component": int(np.bincount(labels).max()),
              "targets": {}}
    for typ in ["LC11", "LC10a"]:
        target_hops = hops[g.cell_types == typ]
        result["targets"][typ] = {
            "total": len(target_hops), "reachable": int((target_hops >= 0).sum()),
            "min_hops": int(target_hops[target_hops >= 0].min()) if (target_hops >= 0).any() else None,
            "max_hops": int(target_hops.max()),
        }
    path.with_name("structural_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    audit()
