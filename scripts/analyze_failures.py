"""Descriptive diagnostics only: does not fit or change any experimental model."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def analyze(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["config"]
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    ref = summary["graphs"][0]
    with (root / ref["name"] / "predictions.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ar = [r for r in rows if r["experiment"] == "A"]
    br = [r for r in rows if r["experiment"] == "B" and int(r["gap"]) == cfg["primary_gap"]
          and int(r["duration"]) == cfg["primary_duration"]]
    shifted = [r for r in br if r["condition"] in cfg["primary_conditions"]]
    p = np.array([float(r["prediction"]) for r in shifted])
    diagnostic = {
        "model": ref["name"], "primary_prediction_min": float(p.min()),
        "primary_prediction_max": float(p.max()),
        "primary_prediction_outside_0_10_fraction": float(np.mean((p < 0) | (p > 10))),
        "primary_mae": ref["primary"]["mae"], "A_conditions": ref["A"],
        "neural_state_diagnostics": ref["activity_diagnostics"],
        "interpretation_scope": "Descriptive analysis of frozen model outputs; no test-based retuning",
    }
    (root / "failure_diagnostics.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    ax = axes[0]
    for condition in cfg["test_conditions"]:
        means = [np.mean([float(r["prediction"]) for r in ar if r["condition"] == condition and int(r["truth"]) == n]) for n in range(11)]
        ax.plot(range(11), means, "o-", markersize=3, label=condition)
    ax.plot([0, 10], [0, 10], "k--", alpha=.5, label="ideal")
    ax.set(xlabel="True single-set numerosity", ylabel="Mean raw prediction", title="A: cue dependence of numerosity readout")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for condition in cfg["primary_conditions"]:
        rr = [r for r in shifted if r["condition"] == condition]
        ax.scatter([int(r["truth"]) for r in rr], [float(r["prediction"]) for r in rr], alpha=.45, s=16, label=condition)
    ax.plot([0, 10], [0, 10], "k--", alpha=.6)
    ax.axhspan(0, 10, color="#148a88", alpha=.09)
    ax.set(xlabel="True sequential sum", ylabel="Raw frozen-readout prediction", title="B: extrapolation beyond the trained answer range")
    ax.legend(frameon=False)
    fig.savefig(root / "failure_diagnostics.png", dpi=160)
    fig.savefig(root / "failure_diagnostics.svg"); plt.close(fig)
    # A logarithmic comparison keeps extreme failures visible without hiding smaller baselines.
    nulls = [g for g in summary["graphs"] if g["family"] == "degree_rewire"]
    others = [g for g in summary["graphs"] if g["family"] not in ("reference", "degree_rewire")]
    names = [ref["name"], "degree rewires (mean)"] + [g["name"] for g in others]
    null_mae = np.array([g["primary"]["mae"] for g in nulls])
    constant_mae = json.loads((root / "comparison.json").read_text())["constant_train_mean_mae"]
    values = [ref["primary"]["mae"], null_mae.mean()] + [g["primary"]["mae"] for g in others]
    fig, ax = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    ax.barh(range(len(names)), values, color=["#148a88", "#8d9bab"] + ["#c0c9d1"] * len(others))
    ax.scatter(null_mae, np.ones(len(null_mae)), s=18, color="#32455c", alpha=.7)
    ax.set_xscale("log"); ax.set_xlim(1, max(values) * 2)
    ax.set_yticks(range(len(names)), names); ax.invert_yaxis()
    for i, value in enumerate(values):
        ax.text(value * 1.08, i, f"{value:.2f}", va="center", fontsize=9)
    ax.axvline(constant_mae, color="#d95f42", linestyle="--", alpha=.6)
    ax.set_xlabel("Unclipped sum MAE (logarithmic axis; lower is better)")
    ax.set_title("B primary endpoint: cue conflict + held-out visual conditions\nDots are 20 degree-preserving rewires; dashed line is constant prediction")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(root / "primary_comparison.png", dpi=160)
    fig.savefig(root / "primary_comparison.svg"); plt.close(fig)
    lines = ["# 1차 예비실험 해석", "", "수치가 큰 합산 오차를 숨기기 위해 예측을 clipping하거나 설정을 재선택하지 않았다.", "",
             f"B의 주 조건에서 원시 예측 범위는 {p.min():.2f}~{p.max():.2f}이다.",
             f"예측의 {100*diagnostic['primary_prediction_outside_0_10_fraction']:.1f}%가 학습한 정답 범위 0~10 밖에 있다.", "",
             "이것은 선형 출력의 외삽 실패를 보여준다. 이를 신경망 활동의 수치적 발산과 동일시할 수는 없다.",
             "모델의 tanh 상태는 유계이며, 저장한 활동량 및 포화율 진단을 함께 확인해야 한다.", "",
             "A의 수량 오차도 단서가 바뀌면 증가했다. 따라서 현재 설정에서는 단서에 강한 수량 표현과",
             "추가 학습 없는 합산 전이 중 어느 것도 충분히 확보했다고 볼 수 없다.", "",
             "이 결과는 한 부분 회로, 한 입력 대응, 한 자극 시드와 정해진 동역학의 결과이다.",
             "초파리 연결망 일반의 능력 또는 실제 초파리의 행동에 대한 부정적 결론으로 확대하지 않는다.", "",
             "![수량 표현과 합산 외삽 진단](failure_diagnostics.png)", "",
             "![주 지표 대조군 비교, 로그 축](primary_comparison.png)", "",
             "C의 조합 분할에서는 0, 3, 7, 10이라는 정답 자체를 학습하지 않는 문제와",
             "정답 5의 새로운 피연산자 조합을 구분한다. 현재 C의 `answer_seen` 층은 정답 5에 한정되므로",
             "폭넓은 조합 일반화 검증으로 해석하기에는 범위가 좁다.", "",
             "후속 실험은 입력 대비/공간 대응과 수량 표현을 먼저 검증하고, 사전에 정한 A validation 탐색으로",
             "설정을 선택한 뒤 새로운 test에서 재현해야 한다. 현재 test 성능을 올리기 위한 재조정은 하지 않았다."]
    (root / "INTERPRETATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(diagnostic, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    analyze(parser.parse_args().path)
