"""Tables and plots for the LR search and the ablations.

    python analysis/extra_analysis.py

Writes:
    analysis/lr_search_table.md      dev accuracy per method, task and LR
    analysis/ablation_table.md       RTE full-data results per ablation setting
    analysis/ablation_table.csv
    analysis/figures/ablations.png   accuracy vs setting, one panel per method

The default setting of every method (and the LR used for the final runs) is
not re-run: it is taken from results/final/ (RTE / MRPC full data, same seeds).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.results import load_results

# Defaults of the final runs, read from their JSONs so this script can run
# on any CPU analysis machine without needing PyTorch or a GPU installed.
_final = load_results("final", root=ROOT / "results")
LEARNING_RATES = _final.groupby("method")["hyperparams.learning_rate"].first().to_dict()
METHOD_CONFIGS = {"lora": {"r": int(_final.loc[_final["method"] == "lora", "method_config.r"].iloc[0])},
                  "adapter": {"reduction_factor": int(_final.loc[_final["method"] == "adapter",
                                                                 "method_config.reduction_factor"].iloc[0])},
                  "prompt_tuning": {"num_virtual_tokens": int(_final.loc[_final["method"] == "prompt_tuning",
                                                                         "method_config.num_virtual_tokens"].iloc[0])}}

OUT = ROOT / "analysis"
FIG = OUT / "figures"
NAMES = {"full_ft": "Full FT", "lora": "LoRA", "adapter": "Adapters",
         "prompt_tuning": "Prompt tuning"}
TASK_NAMES = {"rte": "RTE", "mrpc": "MRPC"}

# ablation name -> (label, value of the default setting used in final runs)
DEFAULTS = {
    "full_ft": [("frozen_layers", "frozen bottom layers", 0)],
    "lora": [("lora_rank", "rank r", METHOD_CONFIGS["lora"]["r"])],
    "adapter": [("reduction_factor", "reduction factor (Houlsby)",
                 METHOD_CONFIGS["adapter"]["reduction_factor"]),
                ("config", "adapter type (reduction 16)", "double_seq_bn")],
    "prompt_tuning": [("num_virtual_tokens", "virtual tokens",
                       METHOD_CONFIGS["prompt_tuning"]["num_virtual_tokens"]),
                      ("init", "prompt init (20 tokens)", "RANDOM")],
}
VALUE_LABELS = {"double_seq_bn": "Houlsby", "seq_bn": "Pfeiffer",
                "RANDOM": "random", "TEXT": "text"}


def pm(s: pd.Series) -> str:
    s = s.dropna() * 100
    if len(s) == 0:
        return "-"
    if len(s) == 1:
        return f"{s.iloc[0]:.1f}"
    return f"{s.mean():.1f} ± {s.std(ddof=1):.1f}"


def lr_search(final: pd.DataFrame) -> None:
    # Summarize dev split accuracies across candidate learning rates to justify final LR choices
    lr = load_results("lr_search", root=ROOT / "results")
    base = final[(final["regime"] == "full") & (final["seed"] == 1)
                 & final["task"].isin(TASK_NAMES)]
    df = pd.concat([lr, base], ignore_index=True) if len(lr) else base
    lines = ["LR search: dev accuracy (our 10% split of the training set) x100, "
             "seed 1, full data. **Bold** = best LR on that task; "
             "\\* = LR used for all final runs. GLUE validation accuracy in brackets.",
             "", "| Method | LR | RTE dev (val) | MRPC dev (val) |",
             "| --- | --- | --- | --- |"]
    picks = []
    for m in NAMES:
        d = df[df["method"] == m]
        best = {t: d[d["task"] == t]["metrics.dev_accuracy"].max() for t in TASK_NAMES}
        for rate in sorted(d["hyperparams.learning_rate"].unique()):
            cells = []
            for t in TASK_NAMES:
                x = d[(d["task"] == t) & (d["hyperparams.learning_rate"] == rate)]
                if not len(x):
                    cells.append("-")
                    continue
                dev, val = x["metrics.dev_accuracy"].iloc[0], x["metrics.accuracy"].iloc[0]
                cell = f"{dev * 100:.1f} ({val * 100:.1f})"
                cells.append(f"**{cell}**" if dev == best[t] else cell)
            star = "\\*" if rate == LEARNING_RATES[m] else ""
            lines.append(f"| {NAMES[m]} | {rate:g}{star} | " + " | ".join(cells) + " |")
        mean_dev = (d.groupby("hyperparams.learning_rate")["metrics.dev_accuracy"]
                    .mean().sort_values(ascending=False))
        if len(mean_dev):
            picks.append(f"- {NAMES[m]}: best mean dev accuracy at LR {mean_dev.index[0]:g} "
                         f"(final runs used {LEARNING_RATES[m]:g}).")
    text = "\n".join(lines) + "\n\n" + "\n".join(picks) + "\n"
    (OUT / "lr_search_table.md").write_text(text, encoding="utf-8")
    print(text)


def ablations(final: pd.DataFrame) -> None:
    # Compile ablation results on RTE across 3 seeds and generate multi-panel errorbar plots
    abl = load_results("ablation", root=ROOT / "results")
    rte_final = final[(final["task"] == "rte") & (final["regime"] == "full")]
    rows, panels = [], []
    for m, specs in DEFAULTS.items():
        for name, label, default in specs:
            groups = [(default, rte_final[rte_final["method"] == m])]
            if len(abl):
                a = abl[(abl["method"] == m) & (abl["ablation.name"] == name)]
                for v in a["ablation.value"].unique():
                    groups.append((v, a[a["ablation.value"] == v]))
            if all(not isinstance(v, str) for v, _ in groups):
                groups.sort(key=lambda g: g[0])
            pts = []
            for v, d in groups:
                acc = d["metrics.accuracy"].dropna() * 100
                rows.append({
                    "Method": NAMES[m], "Ablation": label,
                    "Setting": VALUE_LABELS.get(v, v) if isinstance(v, str) else v,
                    "Default": "yes" if v == default else "",
                    "Seeds": len(acc),
                    "RTE val acc": pm(d["metrics.accuracy"]),
                    "RTE dev acc": pm(d["metrics.dev_accuracy"]),
                    "Trainable params": f"{int(d['efficiency.trainable_params'].median()):,}",
                    "Peak mem (MiB)": f"{d['efficiency.peak_gpu_mem_mb'].max():.0f}",
                    "Time (s)": f"{d['efficiency.train_time_s'].mean():.0f}",
                })
                pts.append((v, acc.mean(), acc.std(ddof=1) if len(acc) > 1 else 0))
            panels.append((m, label, pts))

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "ablation_table.csv", index=False)
    (OUT / "ablation_table.md").write_text(
        "Ablations on RTE, full training data, GLUE validation accuracy x100, "
        "mean ± std over seeds 1-3. One setting changed at a time; the default "
        "row is the final run. LoRA keeps alpha = 2r. Tesla T4, fp16, batch 32.\n\n"
        + table.to_markdown(index=False) + "\n", encoding="utf-8")
    print(table.to_markdown(index=False))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (m, label, pts) in zip(axes.flat, panels):
        numeric = all(not isinstance(v, str) for v, _, _ in pts)
        pts = sorted(pts, key=lambda p: p[0]) if numeric else pts
        xs = [VALUE_LABELS.get(v, str(v)) for v, _, _ in pts]
        means = [p[1] for p in pts]
        stds = [p[2] for p in pts]
        if numeric:
            ax.errorbar(range(len(xs)), means, yerr=stds, marker="o", capsize=4)
        else:
            ax.bar(range(len(xs)), means, yerr=stds, capsize=4, color="tab:gray")
        ax.set_xticks(range(len(xs)), xs)
        ax.set_title(f"{NAMES[m]}: {label}")
        ax.set_ylabel("RTE val acc (%)")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle("Ablations on RTE (full data), mean ± std over 3 seeds")
    fig.tight_layout()
    fig.savefig(FIG / "ablations.png", dpi=150)


if __name__ == "__main__":
    lr_search(_final)
    ablations(_final)
