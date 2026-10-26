from __future__ import annotations

import base64
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_report(path: Path):
    path = Path(path)
    data = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["config"]
    graphs = data["graphs"]
    ref = graphs[0]
    nulls = [g for g in graphs if g["family"] == "degree_rewire"]
    null_mae = np.array([g["primary"]["mae"] for g in nulls])
    delta = ref["primary"]["mae"] - null_mae
    rng = np.random.default_rng(cfg["seed"] + 99)
    draws = rng.choice(delta, size=(cfg["bootstrap_samples"], len(delta)), replace=True).mean(1)
    low, high = np.quantile(draws, [.025, .975])
    with np.load(path / "stimuli.npz") as stimulus:
        train_mean = float(stimulus["train_y"].mean())
        pair_truth = stimulus["a"] + stimulus["b"]
        pair_shift = np.isin(stimulus["pair_conditions"], cfg["primary_conditions"])
        constant_truth = pair_truth[pair_shift]
        constant_mae = float(np.mean(abs(constant_truth - train_mean)))
        constant_accuracy = float(np.mean(np.rint(train_mean) == constant_truth))
    comparison = {
        "reference_mae": ref["primary"]["mae"], "rewire_mean_mae": float(null_mae.mean()),
        "reference_minus_rewire_mean_mae": float(delta.mean()), "conditional_mean_delta_ci95": [float(low), float(high)],
        "rewire_count": len(nulls),
        "reference_rank_among_rewires_plus_reference": 1 + int((null_mae < ref["primary"]["mae"]).sum()),
        "conditional_on": "One stimulus seed, one input mapping, one empirical graph, fixed dynamics",
        "bootstrap_unit": "rewired graph, not image",
        "constant_train_mean_prediction": train_mean,
        "constant_train_mean_mae": constant_mae,
        "constant_train_mean_exact_accuracy": constant_accuracy,
        "evidence_scope": "Exploratory computational pilot; no confirmatory or biological causal conclusion",
    }
    (path / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    with (path / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "family", "experiment", "condition", "gap", "duration",
                                               "n", "mae", "rmse", "exact_accuracy", "bias"])
        writer.writeheader()
        for g in graphs:
            for row in g["A"] + g["B"]:
                writer.writerow({k: ({"model": g["name"], "family": g["family"], **row}).get(k, "")
                                 for k in writer.fieldnames})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    names = [ref["name"], "degree rewires (mean)"] + [g["name"] for g in graphs if g["family"] not in ("reference", "degree_rewire")]
    values = [ref["primary"]["mae"], null_mae.mean()] + [g["primary"]["mae"] for g in graphs if g["family"] not in ("reference", "degree_rewire")]
    ax.barh(np.arange(len(names)), values, color=["#148a88", "#8d9bab"] + ["#c0c9d1"] * (len(names) - 2))
    ax.scatter(null_mae, np.ones(len(null_mae)), s=12, color="#32455c", alpha=.65)
    ax.set_yticks(np.arange(len(names)), names); ax.invert_yaxis()
    ax.set_xlabel("Unclipped sum MAE (lower is better)"); ax.set_title("B: cue-shifted sequential transfer")
    ax = axes[0, 1]
    conds = cfg["test_conditions"]
    xpos = np.arange(len(conds))
    ref_a = [r["mae"] for r in ref["A"]]
    ref_b = [next(r["mae"] for r in ref["B"] if r["condition"] == c and
             r["gap"] == cfg["primary_gap"] and r["duration"] == cfg["primary_duration"]) for c in conds]
    ax.bar(xpos - .18, ref_a, .36, label="A: single set", color="#148a88")
    ax.bar(xpos + .18, ref_b, .36, label="B: sequential sum", color="#e78a56")
    ax.set_xticks(xpos, [s.replace("_", "\n") for s in conds]); ax.set_ylabel("MAE")
    ax.set_title("Reference graph by stimulus family"); ax.legend(frameon=False)
    ax = axes[1, 0]
    for dur in cfg["durations"]:
        rows = sorted([r for r in ref["B"] if r["condition"] == "primary_shift" and r["duration"] == dur], key=lambda r: r["gap"])
        ax.plot([r["gap"] for r in rows], [r["mae"] for r in rows], "o-", label=f"duration {dur} steps")
    ax.set_xlabel("Gap (model steps)"); ax.set_ylabel("Sum MAE")
    ax.set_title("Temporal perturbation, frozen readout"); ax.legend(frameon=False)
    ax = axes[1, 1]
    if ref["C"]:
        for label in ("all", "answer_seen", "answer_unseen", "boundary"):
            rows = [r for r in ref["C"] if r["stratum"] == label]
            ax.plot([r["budget"] for r in rows], [r["mae"] for r in rows], "o-", label=label.replace("_", " "))
        ax.set_xlabel("Addition training examples"); ax.set_ylabel("Held-out combination MAE")
        ax.set_title("C: new linear readout, frozen circuit"); ax.legend(frameon=False)
    else:
        ax.hist(delta, bins=min(10, len(delta)), color="#8d9bab")
        ax.axvline(0, color="#e78a56", linestyle="--")
        ax.set_xlabel("Reference MAE - rewire MAE"); ax.set_title("Graph-to-graph variation")
    fig.suptitle("Frozen connectome numerosity pilot | exploratory, one stimulus seed", fontsize=14)
    fig.savefig(path / "overview.png"); fig.savefig(path / "overview.svg"); plt.close(fig)
    with np.load(path / "stimuli.npz") as stimulus:
        images, labels = stimulus["test"], stimulus["test_y"]
        conditions = np.array([json.loads(l)["condition"] for l in (path / "stimuli_test.jsonl").read_text().splitlines()])
        fig, axes = plt.subplots(len(conds), 6, figsize=(9, 7.5), constrained_layout=True)
        for j, condition in enumerate(conds):
            for k, n in enumerate([0, 1, 2, 4, 7, 10]):
                i = np.flatnonzero((labels == n) & (conditions == condition))[0]
                axes[j, k].imshow(images[i], cmap="gray", vmin=0, vmax=1)
                axes[j, k].set_xticks([]); axes[j, k].set_yticks([])
                if j == 0: axes[j, k].set_title(f"n={n}")
                if k == 0: axes[j, k].set_ylabel(condition.replace("_", "\n"), rotation=0, ha="right", va="center")
        fig.suptitle("Held-out stimulus examples (fixed display contrast)")
        fig.savefig(path / "stimulus_gallery.png"); plt.close(fig)
    audits = []
    metadata = [json.loads(l) for l in (path / "stimuli_test.jsonl").read_text().splitlines()]
    for c in conds:
        rows = [r for r in metadata if r["condition"] == c and r["count"] > 0]
        counts = [r["count"] for r in rows]
        for field in ("geometric_area", "pixel_mass", "occupied_pixels", "bbox_x", "bbox_y", "contrast", "radius"):
            v = np.array([r[field] for r in rows])
            rho = float(np.corrcoef(counts, v)[0, 1]) if np.ptp(v) > 1e-12 else None
            audits.append({"condition": c, "cue": field, "count_correlation": rho,
                           "min": float(v.min()), "max": float(v.max())})
    (path / "cue_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    empirical = manifest["graph_provenance"]["kind"] == "empirical_connectome"
    scope = "실제 FlyWire 유래 부분 회로" if empirical else "합성 연결망을 사용한 소프트웨어 검증"
    a_shift = np.mean([r["mae"] for r in ref["A"] if r["condition"] in cfg["primary_conditions"]])
    lines = ["# 프로젝트 예비실험 결과", "", f"실험 범위: **{scope}**. 탐색적 결과이며 확인 실험이 아니다.", "",
             f"- A, 단서 변화 조건 수량 MAE: **{a_shift:.4f}**",
             f"- B, 단서 변화 조건 합산 MAE: **{ref['primary']['mae']:.4f}**",
             f"- B, 정확한 합 비율: **{100 * ref['primary']['exact_accuracy']:.2f}%**",
             f"- 재배선 {len(nulls)}개 평균 B MAE: **{null_mae.mean():.4f}**",
             f"- 학습 정답 평균만 내는 상수 예측기 B MAE: **{constant_mae:.4f}** (합 정확도 {100*constant_accuracy:.2f}%)",
             f"- 실제/기준 − 재배선 평균 MAE: **{delta.mean():.4f}** (음수가 기준에 유리)",
             f"- 위 평균 차이의 조건부 bootstrap 95% CI: [{low:.4f}, {high:.4f}]", "",
             "이 신뢰구간은 한 자극 시드와 한 입력 대응을 고정한 재배선 간 변동만 반영한다.",
             "자극·입력 설정·개체 수준의 불확실성은 포함하지 않는다.", "",
             "| 모델 | B MAE | 합 정확도 | 공통 alpha=1 MAE |", "|---|---:|---:|---:|"]
    for g in graphs:
        if g["family"] != "degree_rewire":
            p = g["primary"]
            lines.append(f"| {g['name']} | {p['mae']:.4f} | {100*p['exact_accuracy']:.2f}% | {p['common_alpha_mae']:.4f} |")
    interpretation = ("현재 기준 연결망은 상수 예측기보다 B MAE가 크다. 이 설정에서 유용한 추가 학습 없는 합산 전이는 확인되지 않았다."
                      if ref["primary"]["mae"] >= constant_mae else
                      "현재 기준 연결망은 상수 예측기보다 B MAE가 작다. 단서 의존성과 강한 대조군, 다중 시드 반복을 함께 검토해야 한다.")
    lines += ["", "![실험 결과](overview.png)", "", "![자극 예시](stimulus_gallery.png)", "",
              "## 해석과 미완료 검증", "",
              interpretation, "",
              "A의 수량 판독과 B의 합산 전이는 별개의 결과다. 비교군보다 낮은 오차만으로 수량 표현의 일반화나 덧셈 성공을 단정하지 않는다.",
              "밝기·면적 단서가 바뀐 조건에서 A 자체가 충분히 작동하는지, 단순 적분기가 설명하는지 함께 판단해야 한다.",
              "LC11 억제에 대한 생물학적 해석은 보류했다. 입력 대응의 독립 검증과 일반 시각 과제 대조가 필요하다.",
              "glutamate 부호 민감도는 별도 조건으로 비교했다. 다중 자극/입력 시드, 공간/세포유형 제약 재배선, 반복 제시 적응 검증은 후속 단계다.",
              "가공 원자료에 동일 뉴런쌍의 병렬 행이 있어 시냅스 수를 합친 후 기준을 적용했다. 상위 FlyWire 원본과의 최종 대조는 남아 있다.",
              "최종 확인 실험과 외부 사전등록은 수행하지 않았다.", "",
              "## 재현 자료", "",
              "- `manifest.json`: 설정·코드 해시·환경·데이터 출처", "- `metrics.csv`: 조건별 A/B 지표",
              "- 각 모델의 `predictions.csv`: 모든 A/B trial의 원시 예측",
              "- `single_readout.npz`: A에서 학습하고 B에서 고정한 판독기",
              "- `stimuli.npz`, `stimuli_*.jsonl`: 실제 자극·정답·메타데이터",
              "- `cue_audit.json`: 양수 개수에서 단서와 개수의 상관 및 범위",
              "- `comparison.json`: 재배선 단위의 조건부 효과 비교", "",
              f"전체 실행시간: {data['total_seconds']:.1f}초. 모델 step은 생물학적 시간으로 환산하지 않았다.", "",
              "자료: [가공 데이터](https://zenodo.org/records/21549559), [FlyWire 원자료](https://zenodo.org/records/10676866).",
              "코드와 실험은 사용자 제공 기획안을 바탕으로 이 작업 폴더에서 작성·실행했다."]
    report = "\n".join(lines) + "\n"
    (path / "REPORT.md").write_text(report, encoding="utf-8")
    rows = "".join(f"<tr><td>{html.escape(g['name'])}</td><td>{g['primary']['mae']:.4f}</td><td>{100*g['primary']['exact_accuracy']:.2f}%</td></tr>" for g in graphs)
    img = base64.b64encode((path / "overview.png").read_bytes()).decode()
    doc = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>고정 초파리 연결망 · 예비실험</title><style>body{{font:16px/1.65 system-ui,sans-serif;background:#f4f7fa;color:#182b3b;margin:0}}main{{max-width:1100px;margin:40px auto;padding:32px;background:white;border-radius:16px}}h1{{font-size:30px}}.tag{{color:#087c7a;font-weight:700}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px 12px;text-align:left;border-bottom:1px solid #dbe2e8}}.note{{padding:16px;background:#edf5f5}}a{{color:#087c7a}}</style>
<main><p class="tag">EXPLORATORY PILOT · {html.escape(scope)}</p><h1>고정 초파리 연결망의 수량→합산 전이</h1>
<p>B 단서 변화 조건 MAE <b>{ref['primary']['mae']:.4f}</b> · 재배선 {len(nulls)}개 평균 <b>{null_mae.mean():.4f}</b></p>
<p class="note">{html.escape(interpretation)} 한 자극 시드와 한 입력 대응을 사용한 예비실험이며, LC11의 생물학적 기능이나 기호적 덧셈 습득을 입증한 결과가 아닙니다.</p>
<img src="data:image/png;base64,{img}" alt="실험 A, B 및 대조군 결과"><h2>모델별 합산 전이</h2><table><tr><th>모델</th><th>MAE</th><th>합 정확도</th></tr>{rows}</table>
<p>기준−재배선 평균 MAE {delta.mean():.4f}, 재배선 단위 조건부 95% CI [{low:.4f}, {high:.4f}].</p>
<p>상세 결과: <a href="REPORT.md">REPORT.md</a> · <a href="metrics.csv">지표 CSV</a> · <a href="manifest.json">재현 명세</a></p></main></html>'''
    (path / "report.html").write_text(doc, encoding="utf-8")
    print(f"Report written: {path / 'REPORT.md'}", flush=True)
