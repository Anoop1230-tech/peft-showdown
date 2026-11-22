"""Tables and plots for the LR search and the ablations.

    python analysis/extra_analysis.py

Writes:
    analysis/lr_search_table.md      dev accuracy per method, task and LR
    analysis/ablation_table.md       RTE full-data results per ablation setting
    analysis/ablation_table.csv
    analysis/figures/ablations.png   accuracy vs setting, one panel per method

The default setting of every method (and the LR used for the final runs) is
not re-run: it is taken from results/final/ (RTE / MRPC full data, same seeds).
"""  # file description; no effect on running the code

from __future__ import annotations  # allows modern type hints on older Python

import sys  # to change the import search path
from pathlib import Path  # easy file paths

ROOT = Path(__file__).resolve().parent.parent  # this file is analysis/extra_analysis.py, so 2 levels up = repo root
sys.path.insert(0, str(ROOT))  # let Python find the "src" package; without it "from src.results import" fails

import matplotlib  # plotting library
matplotlib.use("Agg")  # draw into files only (no window); without it the script can crash on a machine with no screen
import matplotlib.pyplot as plt  # plotting functions
import pandas as pd  # tables (DataFrames)

from src.results import load_results  # loads result JSONs into a table

_final = load_results("final", root=ROOT / "results")  # all final runs; the default settings are read from here because src.train / src.models need torch
LEARNING_RATES = _final.groupby("method")["hyperparams.learning_rate"].first().to_dict()  # LR used in the final runs per method, e.g. {"lora": 0.0005}
METHOD_CONFIGS = {"lora": {"r": int(_final.loc[_final["method"] == "lora", "method_config.r"].iloc[0])},  # default LoRA rank (8) from the final runs
                  "adapter": {"reduction_factor": int(_final.loc[_final["method"] == "adapter",  # default adapter reduction factor ...
                                                                 "method_config.reduction_factor"].iloc[0])},  # ... (16)
                  "prompt_tuning": {"num_virtual_tokens": int(_final.loc[_final["method"] == "prompt_tuning",  # default number of soft tokens ...
                                                                         "method_config.num_virtual_tokens"].iloc[0])}}  # ... (20)

OUT = ROOT / "analysis"  # where tables are written
FIG = OUT / "figures"  # where figures are written
NAMES = {"full_ft": "Full FT", "lora": "LoRA", "adapter": "Adapters",  # nice display names for the tables ...
         "prompt_tuning": "Prompt tuning"}  # ... and plots (also sets the row order)
TASK_NAMES = {"rte": "RTE", "mrpc": "MRPC"}  # tasks used in the LR search, with display names

DEFAULTS = {  # for each method: (ablation name, label for table/plot, default value used in the final runs)
    "full_ft": [("frozen_layers", "frozen bottom layers", 0)],  # default: 0 layers frozen
    "lora": [("lora_rank", "rank r", METHOD_CONFIGS["lora"]["r"])],  # default rank 8
    "adapter": [("reduction_factor", "reduction factor (Houlsby)",  # adapter bottleneck size ...
                 METHOD_CONFIGS["adapter"]["reduction_factor"]),  # ... default 16
                ("config", "adapter type (reduction 16)", "double_seq_bn")],  # adapter type, default Houlsby
    "prompt_tuning": [("num_virtual_tokens", "virtual tokens",  # number of soft tokens ...
                       METHOD_CONFIGS["prompt_tuning"]["num_virtual_tokens"]),  # ... default 20
                      ("init", "prompt init (20 tokens)", "RANDOM")],  # prompt init, default random
}  # end of DEFAULTS
VALUE_LABELS = {"double_seq_bn": "Houlsby", "seq_bn": "Pfeiffer",  # readable names for text settings ...
                "RANDOM": "random", "TEXT": "text"}  # ... shown in tables and plots


def pm(s: pd.Series) -> str:  # formats scores as "mean ± std" in percent, e.g. "74.0 ± 2.0"
    s = s.dropna() * 100  # drop missing runs, turn fractions into percent
    if len(s) == 0:  # no runs at all
        return "-"  # show a dash instead of "nan"
    if len(s) == 1:  # only one run (e.g. SST-2 full)
        return f"{s.iloc[0]:.1f}"  # just the value; std of one value would be nan
    return f"{s.mean():.1f} ± {s.std(ddof=1):.1f}"  # mean ± sample std (ddof=1 = divide by n-1)


def lr_search(final: pd.DataFrame) -> None:  # builds the LR-search table and writes lr_search_table.md
    lr = load_results("lr_search", root=ROOT / "results")  # the extra LR-search runs
    base = final[(final["regime"] == "full") & (final["seed"] == 1)  # the final LR was not re-run, so take its seed-1 full-data result ...
                 & final["task"].isin(TASK_NAMES)]  # ... on RTE and MRPC from the final runs
    df = pd.concat([lr, base], ignore_index=True) if len(lr) else base  # join both; if there are no LR-search runs, use only the final ones
    lines = ["LR search: dev accuracy (our 10% split of the training set) x100, "  # table caption ...
             "seed 1, full data. **Bold** = best LR on that task; "  # ...
             "\\* = LR used for all final runs. GLUE validation accuracy in brackets.",  # ... (\\* = a literal * in Markdown)
             "", "| Method | LR | RTE dev (val) | MRPC dev (val) |",  # blank line, then the table header
             "| --- | --- | --- | --- |"]  # Markdown header separator
    picks = []  # one summary sentence per method
    for m in NAMES:  # each method in display order
        d = df[df["method"] == m]  # rows of this method
        best = {t: d[d["task"] == t]["metrics.dev_accuracy"].max() for t in TASK_NAMES}  # best dev accuracy per task, to show it in bold
        for rate in sorted(d["hyperparams.learning_rate"].unique()):  # each LR tried, smallest first
            cells = []  # one cell per task for this row
            for t in TASK_NAMES:  # RTE, then MRPC
                x = d[(d["task"] == t) & (d["hyperparams.learning_rate"] == rate)]  # the run for this task and LR
                if not len(x):  # run missing
                    cells.append("-")  # show a dash
                    continue  # go to the next task
                dev, val = x["metrics.dev_accuracy"].iloc[0], x["metrics.accuracy"].iloc[0]  # its dev and validation accuracy
                cell = f"{dev * 100:.1f} ({val * 100:.1f})"  # e.g. "74.0 (72.5)"
                cells.append(f"**{cell}**" if dev == best[t] else cell)  # bold if it is the best LR for this task
            star = "\\*" if rate == LEARNING_RATES[m] else ""  # star marks the LR used in the final runs
            lines.append(f"| {NAMES[m]} | {rate:g}{star} | " + " | ".join(cells) + " |")  # add the table row
        mean_dev = (d.groupby("hyperparams.learning_rate")["metrics.dev_accuracy"]  # dev accuracy averaged over RTE and MRPC per LR ...
                    .mean().sort_values(ascending=False))  # ... best first
        if len(mean_dev):  # if there is any data
            picks.append(f"- {NAMES[m]}: best mean dev accuracy at LR {mean_dev.index[0]:g} "  # say which LR was best ...
                         f"(final runs used {LEARNING_RATES[m]:g}).")  # ... and which one the final runs used
    text = "\n".join(lines) + "\n\n" + "\n".join(picks) + "\n"  # table + summary as one text
    (OUT / "lr_search_table.md").write_text(text, encoding="utf-8")  # save it (utf-8 so special characters are kept)
    print(text)  # also show it in the terminal


def ablations(final: pd.DataFrame) -> None:  # builds the ablation table (csv + md) and the ablations figure
    abl = load_results("ablation", root=ROOT / "results")  # all ablation runs
    rte_final = final[(final["task"] == "rte") & (final["regime"] == "full")]  # RTE full-data final runs = the default setting of each ablation
    rows, panels = [], []  # table rows, and one plot panel per ablation
    for m, specs in DEFAULTS.items():  # each method
        for name, label, default in specs:  # each of its ablations
            groups = [(default, rte_final[rte_final["method"] == m])]  # (setting, runs) list; start with the default setting from the final runs
            if len(abl):  # if there are ablation runs
                a = abl[(abl["method"] == m) & (abl["ablation.name"] == name)]  # runs of this method and this ablation
                for v in a["ablation.value"].unique():  # each tested value
                    groups.append((v, a[a["ablation.value"] == v]))  # add (value, its runs)
            if all(not isinstance(v, str) for v, _ in groups):  # all values are numbers
                groups.sort(key=lambda g: g[0])  # sort them ascending (1, 4, 8, ...); text values keep their order
            pts = []  # (value, mean, std) points for the plot
            for v, d in groups:  # each setting with its runs
                acc = d["metrics.accuracy"].dropna() * 100  # validation accuracies in percent
                rows.append({  # one table row
                    "Method": NAMES[m], "Ablation": label,  # method and what was changed
                    "Setting": VALUE_LABELS.get(v, v) if isinstance(v, str) else v,  # readable setting name
                    "Default": "yes" if v == default else "",  # mark the default setting
                    "Seeds": len(acc),  # how many seeds finished
                    "RTE val acc": pm(d["metrics.accuracy"]),  # validation accuracy mean ± std
                    "RTE dev acc": pm(d["metrics.dev_accuracy"]),  # dev accuracy mean ± std
                    "Trainable params": f"{int(d['efficiency.trainable_params'].median()):,}",  # trainable params with thousands separators
                    "Peak mem (MiB)": f"{d['efficiency.peak_gpu_mem_mb'].max():.0f}",  # highest memory over the seeds
                    "Time (s)": f"{d['efficiency.train_time_s'].mean():.0f}",  # average training time
                })  # end of row
                pts.append((v, acc.mean(), acc.std(ddof=1) if len(acc) > 1 else 0))  # point for the plot; std 0 if only one run (avoids nan)
            panels.append((m, label, pts))  # one panel for this ablation

    table = pd.DataFrame(rows)  # all rows as a table
    table.to_csv(OUT / "ablation_table.csv", index=False)  # save as CSV (index=False = no extra row-number column)
    (OUT / "ablation_table.md").write_text(  # save as Markdown ...
        "Ablations on RTE, full training data, GLUE validation accuracy x100, "  # ... with a caption
        "mean ± std over seeds 1-3. One setting changed at a time; the default "  # ...
        "row is the final run. LoRA keeps alpha = 2r. Tesla T4, fp16, batch 32.\n\n"  # ...
        + table.to_markdown(index=False) + "\n", encoding="utf-8")  # the table in Markdown format (needs the "tabulate" package)
    print(table.to_markdown(index=False))  # show it in the terminal

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))  # a 2 x 3 grid of plots, 14 x 8 inches
    for ax, (m, label, pts) in zip(axes.flat, panels):  # one panel per ablation
        numeric = all(not isinstance(v, str) for v, _, _ in pts)  # numbers (ranks, tokens) or text (Houlsby / Pfeiffer)?
        pts = sorted(pts, key=lambda p: p[0]) if numeric else pts  # sort numeric settings left to right
        xs = [VALUE_LABELS.get(v, str(v)) for v, _, _ in pts]  # x-axis labels
        means = [p[1] for p in pts]  # mean accuracies
        stds = [p[2] for p in pts]  # standard deviations (error bars)
        if numeric:  # numeric settings ...
            ax.errorbar(range(len(xs)), means, yerr=stds, marker="o", capsize=4)  # ... line plot with error bars
        else:  # text settings ...
            ax.bar(range(len(xs)), means, yerr=stds, capsize=4, color="tab:gray")  # ... bar chart with error bars
        ax.set_xticks(range(len(xs)), xs)  # put the labels under the points
        ax.set_title(f"{NAMES[m]}: {label}")  # panel title
        ax.set_ylabel("RTE val acc (%)")  # y-axis label
        ax.grid(alpha=0.3)  # light grid to read values
    for ax in list(axes.flat)[len(panels):]:  # panels that are not used
        ax.axis("off")  # hide them; otherwise empty boxes appear
    fig.suptitle("Ablations on RTE (full data), mean ± std over 3 seeds")  # title for the whole figure
    fig.tight_layout()  # stop labels overlapping
    fig.savefig(FIG / "ablations.png", dpi=150)  # save the figure (dpi 150 = sharp enough for the report)


if __name__ == "__main__":  # only when run directly
    lr_search(_final)  # build the LR-search table
    ablations(_final)  # build the ablation table and figure
