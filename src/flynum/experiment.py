from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from .graph import Graph, fixture, random_graph_like, rewire
from .model import Reservoir, RetinaEncoder, fit_readout, integrated_inputs, metrics, select_readout
from .stimuli import check_leakage, cue_features, pairs, singles


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def source_digest():
    h = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for p in sorted(root.glob("*.py")):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def canonical_split(groups):
    holdout = {"0+0", "0+3", "1+2", "1+4", "2+3", "2+5", "3+4", "5+5"}
    test = np.isin(groups, sorted(holdout))
    return ~test, test


def discrimination(x, y, vx, vy, tx, ty, conditions, seed):
    """Separate linear greater-than readout, fitted on single-set differences."""
    rng = np.random.default_rng(seed)

    def comparisons(states, labels):
        i, j = rng.integers(len(labels), size=(2, max(300, len(labels) * 2)))
        keep = labels[i] != labels[j]
        i, j = i[keep], j[keep]
        return states[i] - states[j], np.sign(labels[i] - labels[j]), i, j

    dx, dy, _, _ = comparisons(x, y)
    dvx, dvy, _, _ = comparisons(vx, vy)
    model, _ = select_readout(dx, dy, dvx, dvy, [0.01, 1, 100])
    dtx, dty, i, j = comparisons(tx, ty)
    pred = np.sign(model.predict(dtx))
    ratios = np.minimum(ty[i], ty[j]) / np.maximum(ty[i], ty[j])
    rows = []
    for lo, hi in [(0, .25), (.25, .5), (.5, .75), (.75, 1.01)]:
        mask = (ratios >= lo) & (ratios < hi)
        if mask.any():
            rows.append({"ratio_min": lo, "ratio_max": hi, "n": int(mask.sum()),
                         "accuracy": float(np.mean(pred[mask] == dty[mask]))})
    return {"accuracy": float(np.mean(pred == dty)), "n": len(dty), "by_ratio": rows,
            "note": "Zero/nonzero comparisons included in first ratio bin; ties excluded."}


def run(config: dict, output: Path, graph: Graph | None, include_c=False, resume=False):
    started = time.perf_counter()
    if graph is None:
        graph = fixture(config["seed"])
    output.mkdir(parents=True, exist_ok=True)
    signature = {"config": config, "graph_fingerprint": graph.fingerprint(),
                 "source_sha256": source_digest(), "include_c": include_c}
    if (output / "manifest.json").exists():
        old = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if not resume or any(old.get(k) != v for k, v in signature.items()):
            raise ValueError("Output exists: use a new directory, or --resume with identical inputs/code")
    else:
        write_json(output / "manifest.json", {
            **signature, "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "graph_provenance": graph.provenance,
            "status": "exploratory_pilot", "confirmatory": False,
        })
    seed = config["seed"]
    size = config["image_size"]
    train = singles(seed + 1, config["train_conditions"], config["train_per_count_condition"], size)
    val = singles(seed + 2, config["train_conditions"], config["validation_per_count_condition"], size)
    test = singles(seed + 3, config["test_conditions"], config["test_per_count_condition"], size)
    sequence = pairs(seed + 4, config["test_conditions"], config["pair_repeats"], size)
    leakage = check_leakage(train, val, test)
    write_json(output / "stimulus_audit.json", leakage)
    for name, dataset in (("train", train), ("validation", val), ("test", test)):
        with (output / f"stimuli_{name}.jsonl").open("w", encoding="utf-8") as f:
            for m in dataset.metadata:
                f.write(json.dumps(m) + "\n")
    np.savez_compressed(output / "stimuli.npz", train=train.images, train_y=train.labels,
                        validation=val.images, validation_y=val.labels, test=test.images, test_y=test.labels,
                        pair_a=sequence.a, pair_b=sequence.b, a=sequence.labels_a, b=sequence.labels_b,
                        pair_conditions=sequence.conditions, pair_reverse_index=sequence.reverse_index)
    encoder = RetinaEncoder(graph, size)
    encoded = [encoder(s.images) for s in (train, val, test)]
    encoded_pairs = (encoder(sequence.a), encoder(sequence.b))
    cdata = pairs(seed + 5, config["test_conditions"], 3, size) if include_c else None
    if include_c:
        cencoded = (encoder(cdata.a), encoder(cdata.b))
        cmask_train, cmask_test = canonical_split(cdata.groups)
        # Training uses in-distribution stimulus families only.
        cmask_train &= np.isin(cdata.conditions, config["train_conditions"])
        corder = np.flatnonzero(cmask_train)
        np.random.default_rng(seed + 6).shuffle(corder)
        write_json(output / "c_split.json", {
            "train_groups": sorted(set(cdata.groups[cmask_train])),
            "test_groups": sorted(set(cdata.groups[cmask_test])),
            "nested_train_indices": corder.tolist(), "test_indices": np.flatnonzero(cmask_test).tolist(),
            "train_test_group_overlap": 0, "ridge_alpha": 1.0,
        })
        np.savez_compressed(output / "stimuli_c.npz", a=cdata.a, b=cdata.b,
                            labels_a=cdata.labels_a, labels_b=cdata.labels_b,
                            groups=cdata.groups, conditions=cdata.conditions)
    summaries = []
    variants = [("anatomical" if graph.provenance["kind"] == "empirical_connectome" else "synthetic_fixture", "reference", None)]
    variants += [(f"rewire_{i:02}", "degree_rewire", i) for i in range(config["rewires"])]
    variants += [("random_matched", "random", None), ("weight_shuffle", "weight_shuffle", None),
                 ("no_recurrent_edges", "no_recurrence", None)]
    if graph.provenance.get("glutamate_neuron_ids"):
        variants += [("glutamate_excitatory", "sign_sensitivity", None)]
    variants += [("image_cues", "cues", None), ("pixel_integrator", "integrator", 1.0),
                 ("leaky_pixel_integrator", "integrator", 0.8)]
    for ordinal, (name, family, param) in enumerate(variants):
        graph_dir = output / name
        graph_dir.mkdir(exist_ok=True)
        if resume and (graph_dir / "summary.json").exists():
            summaries.append(json.loads((graph_dir / "summary.json").read_text()))
            print(f"Resumed {name}", flush=True)
            continue
        t = time.perf_counter()
        rewire_info = {}
        if family == "degree_rewire":
            variant, rewire_info = rewire(graph, seed + 1000 + param, config["swaps_per_edge"],
                                          config["min_changed_fraction"])
        elif family == "random":
            variant = random_graph_like(graph, seed + 2000)
        elif family == "weight_shuffle":
            variant = replace(graph, count=np.random.default_rng(seed + 2001).permutation(graph.count))
        elif family == "sign_sensitivity":
            sign_variant = graph.signs.copy()
            sign_variant[np.isin(graph.ids, np.array(graph.provenance["glutamate_neuron_ids"], dtype=np.uint64))] = 1
            variant = replace(graph, signs=sign_variant)
        else:
            variant = graph
        if family in ("cues", "integrator"):
            def features(a, b=None, gap=0, duration=4):
                if family == "cues":
                    return cue_features(a if b is None else b)
                aa = a.reshape(len(a), -1)
                bb = None if b is None else b.reshape(len(b), -1)
                return integrated_inputs(aa, bb, duration, gap, config["post_steps"], decay=param)
            x, vx, tx = [features(s.images, duration=config["primary_duration"]) for s in (train, val, test)]
            af, bf = sequence.a, sequence.b
            weight_diag, activity = {}, {}
        else:
            reservoir = Reservoir(variant, config["dynamics"], config["batch_size"], family != "no_recurrence")
            def features(a, b=None, gap=0, duration=4):
                return reservoir.states(a, b, gap, duration, config["post_steps"])
            x, vx, tx = [features(z, duration=config["primary_duration"]) for z in encoded]
            af, bf = encoded_pairs
            weight_diag, activity = reservoir.weight_diagnostics, reservoir.last_activity.copy()
        readout, losses = select_readout(x, train.labels, vx, val.labels, config["ridge_alphas"])
        common = readout if readout.alpha == 1 else fit_readout(x, train.labels, 1)
        readout.save(graph_dir / "single_readout.npz")
        readout_before = readout.fingerprint()
        a_prediction = readout.predict(tx)
        a_rows = [{"experiment": "A", "condition": c, **metrics(test.labels[test.conditions == c],
                                a_prediction[test.conditions == c])} for c in config["test_conditions"]]
        discrim = discrimination(x, train.labels, vx, val.labels, tx, test.labels, test.conditions, seed + 7)
        b_rows, c_rows, primary_prediction = [], [], None
        with (graph_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["experiment", "trial", "condition", "gap", "duration",
                                                  "a", "b", "truth", "prediction", "common_alpha_prediction"])
            writer.writeheader()
            for i, p in enumerate(a_prediction):
                writer.writerow(dict(experiment="A", trial=i, condition=test.conditions[i], gap=0,
                                     duration=config["primary_duration"], a=test.labels[i], b="",
                                     truth=test.labels[i], prediction=p,
                                     common_alpha_prediction=common.predict(tx[i:i+1])[0]))
            for duration in config["durations"]:
                for gap in config["gaps"]:
                    sx = features(af, bf, gap, duration)
                    pred, cpred = readout.predict(sx), common.predict(sx)
                    is_primary_time = gap == config["primary_gap"] and duration == config["primary_duration"]
                    for c in config["test_conditions"] + ["primary_shift"]:
                        mask = (np.isin(sequence.conditions, config["primary_conditions"]) if c == "primary_shift"
                                else sequence.conditions == c)
                        b_rows.append({"experiment": "B", "condition": c, "gap": gap, "duration": duration,
                                       **metrics(sequence.labels[mask], pred[mask]),
                                       "common_alpha_mae": metrics(sequence.labels[mask], cpred[mask])["mae"],
                                       "order_swap_mean_abs_change": float(np.mean(abs(pred[mask] - pred[sequence.reverse_index][mask]))),
                                       "zero_operand_mae": metrics(sequence.labels[mask & ((sequence.labels_a == 0) | (sequence.labels_b == 0))],
                                               pred[mask & ((sequence.labels_a == 0) | (sequence.labels_b == 0))])["mae"],
                                       "both_nonzero_mae": metrics(sequence.labels[mask & (sequence.labels_a > 0) & (sequence.labels_b > 0)],
                                               pred[mask & (sequence.labels_a > 0) & (sequence.labels_b > 0)])["mae"]})
                    if is_primary_time:
                        primary_prediction = pred.copy()
                        if family not in ("cues", "integrator"):
                            activity = reservoir.last_activity.copy()
                    for i, p in enumerate(pred):
                        writer.writerow(dict(experiment="B", trial=i, condition=sequence.conditions[i], gap=gap,
                            duration=duration, a=sequence.labels_a[i], b=sequence.labels_b[i], truth=sequence.labels[i],
                            prediction=p, common_alpha_prediction=cpred[i]))
        if readout.fingerprint() != readout_before:
            raise AssertionError("Single-set readout changed during B")
        if include_c:
            ca, cb = (cdata.a, cdata.b) if family in ("cues", "integrator") else cencoded
            cx = features(ca, cb, config["primary_gap"], config["primary_duration"])
            ctest = np.flatnonzero(cmask_test)
            with (graph_dir / "c_predictions.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["budget", "trial", "condition", "group", "truth", "prediction", "answer_seen", "boundary"])
                writer.writeheader()
                for budget in config["c_budgets"]:
                    if budget > len(corder):
                        raise ValueError("C budget exceeds training pool")
                    idx = corder[:budget]
                    cm = fit_readout(cx[idx], cdata.labels[idx], 1)
                    cp = cm.predict(cx[ctest])
                    seen = np.isin(cdata.labels[ctest], np.unique(cdata.labels[idx]))
                    boundary = np.isin(cdata.labels[ctest], [0, 10])
                    for label, mask in [("all", np.ones(len(ctest), bool)), ("answer_seen", seen),
                                        ("answer_unseen", ~seen), ("boundary", boundary)]:
                        if mask.any():
                            c_rows.append({"budget": budget, "stratum": label,
                                           **metrics(cdata.labels[ctest][mask], cp[mask])})
                    for j, i in enumerate(ctest):
                        writer.writerow(dict(budget=budget, trial=i, condition=cdata.conditions[i], group=cdata.groups[i],
                            truth=cdata.labels[i], prediction=cp[j], answer_seen=bool(seen[j]), boundary=bool(boundary[j])))
        primary = next(r for r in b_rows if r["condition"] == "primary_shift" and
                       r["gap"] == config["primary_gap"] and r["duration"] == config["primary_duration"])
        summary = {"name": name, "family": family, "primary": primary,
                   "readout_alpha": readout.alpha, "validation_mae": losses,
                   "readout_sha256_before_b": readout_before, "readout_sha256_after_b": readout.fingerprint(),
                   "graph_sha256": variant.fingerprint() if family not in ("cues", "integrator") else None,
                   "rewire_diagnostics": rewire_info, "weight_diagnostics": weight_diag,
                   "activity_diagnostics": activity, "feature_count": x.shape[1], "A": a_rows, "B": b_rows,
                   "C": c_rows, "discrimination": discrim, "seconds": time.perf_counter() - t}
        write_json(graph_dir / "summary.json", summary)
        np.save(graph_dir / "primary_predictions.npy", primary_prediction)
        summaries.append(summary)
        print(f"[{ordinal + 1}/{len(variants)}] {name}: B shift MAE={primary['mae']:.4f}, "
              f"A MAE={np.mean([r['mae'] for r in a_rows]):.4f}, {summary['seconds']:.1f}s", flush=True)
    write_json(output / "summary.json", {"graphs": summaries, "total_seconds": time.perf_counter() - started})
    from .report import make_report
    make_report(output)
    return summaries
