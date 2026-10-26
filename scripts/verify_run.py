"""Independently reconcile saved trial CSVs, stimuli, readouts, and reported metrics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from flynum.model import LinearReadout


def verify(root: Path):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    with np.load(root / "stimuli.npz") as data:
        ay = data["test_y"].copy()
        a, b = data["a"].copy(), data["b"].copy()
        conditions = data["pair_conditions"].copy()
        def nonzero_hashes(images, labels):
            return {hashlib.sha256(im.tobytes()).hexdigest() for im, n in zip(images, labels) if n > 0}
        atrain = nonzero_hashes(data["train"], data["train_y"])
        aval = nonzero_hashes(data["validation"], data["validation_y"])
        atest = nonzero_hashes(data["test"], data["test_y"])
        btest = nonzero_hashes(data["pair_a"], a) | nonzero_hashes(data["pair_b"], b)
        assert not atrain & aval and not atrain & atest and not aval & atest
        assert not (atrain | aval | atest) & btest, "Nonzero image shared between A and B"
    if (root / "stimuli_c.npz").exists():
        with np.load(root / "stimuli_c.npz") as cdata:
            cimages = nonzero_hashes(cdata["a"], cdata["labels_a"]) | nonzero_hashes(cdata["b"], cdata["labels_b"])
            assert not cimages & (atrain | aval | atest | btest), "Nonzero image shared between C and A/B"
    expected_b_rows = len(a) * len(config["gaps"]) * len(config["durations"])
    audited = []
    for graph in summary["graphs"]:
        path = root / graph["name"]
        with (path / "predictions.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        aa = [r for r in rows if r["experiment"] == "A"]
        bb = [r for r in rows if r["experiment"] == "B"]
        assert len(aa) == len(ay), (graph["name"], "A row count")
        assert len(bb) == expected_b_rows, (graph["name"], "B row count")
        for row in aa:
            i = int(row["trial"])
            assert float(row["truth"]) == ay[i]
        primary_errors, common_errors = [], []
        for row in bb:
            i = int(row["trial"])
            assert int(row["a"]) == a[i] and int(row["b"]) == b[i]
            assert float(row["truth"]) == a[i] + b[i]
            assert row["condition"] == conditions[i]
            if (int(row["gap"]) == config["primary_gap"]
                    and int(row["duration"]) == config["primary_duration"]
                    and row["condition"] in config["primary_conditions"]):
                primary_errors.append(abs(float(row["prediction"]) - float(row["truth"])))
                common_errors.append(abs(float(row["common_alpha_prediction"]) - float(row["truth"])))
        assert np.isclose(np.mean(primary_errors), graph["primary"]["mae"], rtol=1e-12)
        assert np.isclose(np.mean(common_errors), graph["primary"]["common_alpha_mae"], rtol=1e-12)
        with np.load(path / "single_readout.npz") as saved:
            readout = LinearReadout(saved["mean"], saved["scale"], saved["coef"],
                                    float(saved["intercept"]), float(saved["alpha"]))
        assert readout.fingerprint() == graph["readout_sha256_before_b"] == graph["readout_sha256_after_b"]
        if graph["family"] == "degree_rewire":
            assert graph["rewire_diagnostics"]["changed_edge_fraction"] >= config["min_changed_fraction"]
            assert graph["rewire_diagnostics"]["directed_degrees_preserved"]
            assert graph["rewire_diagnostics"]["signed_in_degrees_preserved"]
        audited.append({"model": graph["name"], "A_trials": len(aa), "B_trials": len(bb),
                        "primary_trials": len(primary_errors), "readout_hash_verified": True,
                        "metrics_recomputed_from_csv": True})
    rewire_count = sum(g["family"] == "degree_rewire" for g in summary["graphs"])
    assert rewire_count == config["rewires"]
    assert json.loads((root / "stimulus_audit.json").read_text())["nonzero_cross_split_duplicates"] == 0
    if (root / "c_split.json").exists():
        split = json.loads((root / "c_split.json").read_text())
        assert not set(split["train_groups"]) & set(split["test_groups"])
    report = {"passed": True, "models": len(audited), "rewires": rewire_count,
              "nonzero_A_B_C_cross_experiment_image_overlap": 0,
              "checks": audited, "scope": "Saved-artifact integrity and independent metric reconciliation"}
    (root / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Verified {len(audited)} models, {rewire_count} rewires, all trial targets and frozen readout hashes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    verify(parser.parse_args().path)
