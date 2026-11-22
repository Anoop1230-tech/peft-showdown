"""Tables and plots for the LR search and the ablations.

    python analysis/extra_analysis.py

Writes:
    analysis/lr_search_table.md      dev accuracy per method, task and LR
    analysis/ablation_table.md       RTE full-data results per ablation setting
    analysis/ablation_table.csv
    analysis/figures/ablations.png   accuracy vs setting, one panel per method

The default setting of every method (and the LR used for the final runs) is
not re-run: it is taken from results/final/ (RTE / MRPC full data, same seeds).
"""  # DOCSTRING: description of this whole file; it does not run anything. This file's job: read the LR-search and ablation result JSONs and turn them into the tables and the figure used in the report (Sections 4.1 and 5.3).

from __future__ import annotations  # Makes Python treat type hints as plain text, so modern hint syntax works on older Python versions. Does not change how the code runs.

import sys  # Standard library; used to add the repo folder to Python's import search path (sys.path).
from pathlib import Path  # Object-style file paths that work on Windows and Linux.

ROOT = Path(__file__).resolve().parent.parent  # Absolute path of the project folder. This file is analysis/extra_analysis.py: .parent = analysis/, .parent again = repo root.
sys.path.insert(0, str(ROOT))  # Put the repo root at the FRONT of Python's import search path. This file lives in analysis/, so without this "from src.results import ..." would fail with "No module named src".

import matplotlib  # Plotting library.
matplotlib.use("Agg")  # Select the "Agg" backend, which draws straight into image files and never opens a window. Must be set BEFORE importing pyplot. Without it the script can crash on a machine without a screen (e.g. a server or Kaggle).
import matplotlib.pyplot as plt  # The usual plotting interface (subplots, errorbar, bar, savefig ...), imported as plt by convention.
import pandas as pd  # Table library: results are loaded as a DataFrame (a table) and filtered, grouped and averaged with it.

from src.results import load_results  # Our function (src/results.py) that reads result JSON files into one pandas table, one row per run, with columns like "metrics.accuracy".

_final = load_results("final", root=ROOT / "results")  # Load all 64 FINAL runs. They are needed because the default setting of every ablation, and the LR used in the final runs, were not re-run: their numbers come from these final results.
LEARNING_RATES = _final.groupby("method")["hyperparams.learning_rate"].first().to_dict()  # The LR each method used in its final runs, read from the result files: group rows by method, take the first LR, turn into a dict, e.g. {"lora": 0.0005, ...}. Read from the data (not imported from train.py) because train.py needs torch, which the analysis machine may not have.
METHOD_CONFIGS = {"lora": {"r": int(_final.loc[_final["method"] == "lora", "method_config.r"].iloc[0])},  # Default LoRA rank used in the final runs (8), read from the first LoRA result. Used below to mark the "default" row in the rank ablation.
                  "adapter": {"reduction_factor": int(_final.loc[_final["method"] == "adapter",  # Default adapter reduction factor used in the final runs ...
                                                                 "method_config.reduction_factor"].iloc[0])},  # ... (16), read from the first adapter result.
                  "prompt_tuning": {"num_virtual_tokens": int(_final.loc[_final["method"] == "prompt_tuning",  # Default number of virtual tokens used in the final runs ...
                                                                         "method_config.num_virtual_tokens"].iloc[0])}}  # ... (20), read from the first prompt-tuning result.

OUT = ROOT / "analysis"  # Folder where the tables (.md / .csv) are written.
FIG = OUT / "figures"  # Folder where the figure (.png) is written.
NAMES = {"full_ft": "Full FT", "lora": "LoRA", "adapter": "Adapters",  # Readable display names for tables and plots (instead of code names like "full_ft") ...
         "prompt_tuning": "Prompt tuning"}  # ... the order of this dict is also the order of the rows in the tables.
TASK_NAMES = {"rte": "RTE", "mrpc": "MRPC"}  # The two tasks used in the LR search, with display names. (SST-2 was not included: one SST-2 full run takes about an hour.)

DEFAULTS = {  # For each method, the list of ablations: (ablation name as saved in the results, label for the table/plot, the DEFAULT value used in the final runs). The default value is needed because that row is taken from the final runs instead of the ablation runs.
    "full_ft": [("frozen_layers", "frozen bottom layers", 0)],  # Full FT: number of frozen bottom layers; default 0 (nothing frozen).
    "lora": [("lora_rank", "rank r", METHOD_CONFIGS["lora"]["r"])],  # LoRA: rank r; default 8.
    "adapter": [("reduction_factor", "reduction factor (Houlsby)",  # Adapters ablation 1: bottleneck reduction factor ...
                 METHOD_CONFIGS["adapter"]["reduction_factor"]),  # ... default 16.
                ("config", "adapter type (reduction 16)", "double_seq_bn")],  # Adapters ablation 2: adapter type; default Houlsby (double_seq_bn).
    "prompt_tuning": [("num_virtual_tokens", "virtual tokens",  # Prompt tuning ablation 1: number of virtual tokens ...
                       METHOD_CONFIGS["prompt_tuning"]["num_virtual_tokens"]),  # ... default 20.
                      ("init", "prompt init (20 tokens)", "RANDOM")],  # Prompt tuning ablation 2: initialisation; default RANDOM.
}  # End of DEFAULTS. Total: 6 ablations, which become the 6 panels of the figure.
VALUE_LABELS = {"double_seq_bn": "Houlsby", "seq_bn": "Pfeiffer",  # Readable names for text-valued settings, shown in tables and on the plot axis (code name -> paper name) ...
                "RANDOM": "random", "TEXT": "text"}  # ... and lower-case labels for the prompt init options.


def pm(s: pd.Series) -> str:  # HELPER pm ("plus-minus"): turns a column of scores (fractions like 0.761) into the text "mean ± std" in percent, e.g. "76.1 ± 2.2", the format used in all report tables.
    s = s.dropna() * 100  # Remove missing values (runs without that score), then convert fractions to percentages.
    if len(s) == 0:  # No values at all ...
        return "-"  # ... show a dash instead of "nan".
    if len(s) == 1:  # Only one value (e.g. a single seed) ...
        return f"{s.iloc[0]:.1f}"  # ... show just that number with 1 decimal; the standard deviation of one value is undefined (nan).
    return f"{s.mean():.1f} ± {s.std(ddof=1):.1f}"  # Otherwise: mean ± SAMPLE standard deviation (ddof=1 divides by n-1, the correct choice for a small sample like 3 seeds), both with 1 decimal.


def lr_search(final: pd.DataFrame) -> None:  # FUNCTION lr_search: builds the learning-rate search table (report Section 4.1): for each method and each of its 3 LRs, dev accuracy (and validation accuracy in brackets) on RTE and MRPC, seed 1, full data. Writes analysis/lr_search_table.md and prints it.
    lr = load_results("lr_search", root=ROOT / "results")  # Load the 16 LR-search runs (2 extra LRs x 4 methods x 2 tasks).
    base = final[(final["regime"] == "full") & (final["seed"] == 1)  # The third LR of each method (the one used in the final runs) was NOT re-run, so take those runs from the final results: full data, seed 1 ...
                 & final["task"].isin(TASK_NAMES)]  # ... and only RTE and MRPC.
    df = pd.concat([lr, base], ignore_index=True) if len(lr) else base  # Stack both tables into one, so each method now has all 3 LRs. If there are no LR-search results yet, use only the final ones.
    lines = ["LR search: dev accuracy (our 10% split of the training set) x100, "  # Build the Markdown file line by line. First the caption explaining the table ...
             "seed 1, full data. **Bold** = best LR on that task; "  # ... what bold means ...
             "\\* = LR used for all final runs. GLUE validation accuracy in brackets.",  # ... what the star means ("\\*" is an escaped star, so Markdown shows a literal * instead of starting italics) ...
             "", "| Method | LR | RTE dev (val) | MRPC dev (val) |",  # ... an empty line, then the table header row ...
             "| --- | --- | --- | --- |"]  # ... and the separator row that Markdown requires under a header.
    picks = []  # Summary sentences, one per method, saying which LR was best on average.
    for m in NAMES:  # For each method, in display order:
        d = df[df["method"] == m]  # keep only this method's rows.
        best = {t: d[d["task"] == t]["metrics.dev_accuracy"].max() for t in TASK_NAMES}  # For each task, the highest dev accuracy among this method's LRs, so that cell can be shown in bold.
        for rate in sorted(d["hyperparams.learning_rate"].unique()):  # For each distinct LR of this method, smallest first (one table row each):
            cells = []  # The cells of this row, one per task.
            for t in TASK_NAMES:  # For RTE, then MRPC:
                x = d[(d["task"] == t) & (d["hyperparams.learning_rate"] == rate)]  # find the run for this task and LR.
                if not len(x):  # If that run doesn't exist (not done yet) ...
                    cells.append("-")  # ... show a dash ...
                    continue  # ... and move on to the next task.
                dev, val = x["metrics.dev_accuracy"].iloc[0], x["metrics.accuracy"].iloc[0]  # Read its dev accuracy (used for choosing) and validation accuracy (shown for information).
                cell = f"{dev * 100:.1f} ({val * 100:.1f})"  # Format as "dev (val)" in percent, e.g. "75.9 (76.2)".
                cells.append(f"**{cell}**" if dev == best[t] else cell)  # Make it bold (**...** in Markdown) if it is the best dev accuracy for this task.
            star = "\\*" if rate == LEARNING_RATES[m] else ""  # Add a star after the LR if it is the one used in the final runs.
            lines.append(f"| {NAMES[m]} | {rate:g}{star} | " + " | ".join(cells) + " |")  # Add the finished row, e.g. "| LoRA | 0.0005\* | 75.9 (76.2) | 88.3 (86.5) |". ":g" prints the LR in short form.
        mean_dev = (d.groupby("hyperparams.learning_rate")["metrics.dev_accuracy"]  # For each LR, average the dev accuracy over RTE and MRPC ...
                    .mean().sort_values(ascending=False))  # ... and sort from best to worst.
        if len(mean_dev):  # If there is any data for this method:
            picks.append(f"- {NAMES[m]}: best mean dev accuracy at LR {mean_dev.index[0]:g} "  # write which LR had the best average dev accuracy ...
                         f"(final runs used {LEARNING_RATES[m]:g}).")  # ... next to the LR the final runs used, so you can see at a glance whether they match (they did for every method).
    text = "\n".join(lines) + "\n\n" + "\n".join(picks) + "\n"  # Join the table lines and the summary sentences into one text, with a blank line between them.
    (OUT / "lr_search_table.md").write_text(text, encoding="utf-8")  # Save it as analysis/lr_search_table.md. utf-8 keeps special characters like "±".
    print(text)  # Also print it in the terminal so you see the result immediately.


def ablations(final: pd.DataFrame) -> None:  # FUNCTION ablations: builds the ablation table (report Section 5.3) and the 6-panel ablation figure. All on RTE full data, mean ± std over seeds 1-3. Writes ablation_table.csv, ablation_table.md and figures/ablations.png.
    abl = load_results("ablation", root=ROOT / "results")  # Load the 42 ablation runs (14 settings x 3 seeds).
    rte_final = final[(final["task"] == "rte") & (final["regime"] == "full")]  # The final RTE full-data runs. These are the DEFAULT setting of every ablation (e.g. LoRA r=8), so they are reused instead of re-running them.
    rows, panels = [], []  # rows = one dict per table row; panels = one entry per plot panel (one per ablation).
    for m, specs in DEFAULTS.items():  # For each method and its list of ablations:
        for name, label, default in specs:  # For each ablation of that method:
            groups = [(default, rte_final[rte_final["method"] == m])]  # Start a list of (setting value, runs with that setting). The first entry is the DEFAULT value with this method's final RTE runs.
            if len(abl):  # If ablation results exist:
                a = abl[(abl["method"] == m) & (abl["ablation.name"] == name)]  # keep only the runs of this method and this ablation ...
                for v in a["ablation.value"].unique():  # ... and for each distinct value tested (e.g. r = 1, 4, 16, 64):
                    groups.append((v, a[a["ablation.value"] == v]))  # add (value, its 3 seed runs) to the list.
            if all(not isinstance(v, str) for v, _ in groups):  # If ALL values are numbers (ranks, token counts, layers) ...
                groups.sort(key=lambda g: g[0])  # ... sort them from small to large, so the table reads 1, 4, 8, 16, 64. Text values (Houlsby/Pfeiffer) keep their order: default first.
            pts = []  # Points for this ablation's plot panel: (value, mean accuracy, std).
            for v, d in groups:  # For each setting and its runs:
                acc = d["metrics.accuracy"].dropna() * 100  # Validation accuracies of its seeds, in percent.
                rows.append({  # Add one table row with these columns:
                    "Method": NAMES[m], "Ablation": label,  # method name and what was changed,
                    "Setting": VALUE_LABELS.get(v, v) if isinstance(v, str) else v,  # the setting value, using the readable name for text values (e.g. "Houlsby"),
                    "Default": "yes" if v == default else "",  # "yes" marks the default row (taken from the final runs),
                    "Seeds": len(acc),  # how many seeds actually finished (should be 3),
                    "RTE val acc": pm(d["metrics.accuracy"]),  # validation accuracy as "mean ± std",
                    "RTE dev acc": pm(d["metrics.dev_accuracy"]),  # dev accuracy as "mean ± std",
                    "Trainable params": f"{int(d['efficiency.trainable_params'].median()):,}",  # trainable parameters (same for every seed; median just picks it), with thousands separators,
                    "Peak mem (MiB)": f"{d['efficiency.peak_gpu_mem_mb'].max():.0f}",  # highest peak memory over the seeds, no decimals,
                    "Time (s)": f"{d['efficiency.train_time_s'].mean():.0f}",  # average training time over the seeds, no decimals.
                })  # End of row.
                pts.append((v, acc.mean(), acc.std(ddof=1) if len(acc) > 1 else 0))  # Add the plot point: mean accuracy and sample std; std is set to 0 when there is only one run (otherwise it would be nan and break the error bar).
            panels.append((m, label, pts))  # Store this ablation as one panel of the figure.

    table = pd.DataFrame(rows)  # Turn the list of row dicts into a pandas table.
    table.to_csv(OUT / "ablation_table.csv", index=False)  # Save as CSV (easy to open in Excel). index=False leaves out pandas' row numbers.
    (OUT / "ablation_table.md").write_text(  # Save as Markdown for the report, starting with a caption that explains the setup:
        "Ablations on RTE, full training data, GLUE validation accuracy x100, "  # what is measured,
        "mean ± std over seeds 1-3. One setting changed at a time; the default "  # how it is averaged and that only one setting changes at a time,
        "row is the final run. LoRA keeps alpha = 2r. Tesla T4, fp16, batch 32.\n\n"  # where the default row comes from and the hardware,
        + table.to_markdown(index=False) + "\n", encoding="utf-8")  # followed by the table itself in Markdown format (to_markdown needs the "tabulate" package).
    print(table.to_markdown(index=False))  # Also print the table in the terminal.

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))  # Create one figure with a grid of 2 rows x 3 columns = 6 small plots (one per ablation), 14 x 8 inches in size.
    for ax, (m, label, pts) in zip(axes.flat, panels):  # Pair each plot area with one ablation panel (axes.flat walks through the grid left-to-right, top-to-bottom):
        numeric = all(not isinstance(v, str) for v, _, _ in pts)  # Are the settings numbers (ranks, tokens, layers) or text (Houlsby/Pfeiffer, random/text)?
        pts = sorted(pts, key=lambda p: p[0]) if numeric else pts  # Sort numeric settings from small to large so the line goes left to right.
        xs = [VALUE_LABELS.get(v, str(v)) for v, _, _ in pts]  # Labels for the x-axis: readable names for text values, the number as text otherwise.
        means = [p[1] for p in pts]  # Mean accuracy of each setting (the height of each point/bar).
        stds = [p[2] for p in pts]  # Standard deviation of each setting (the size of each error bar).
        if numeric:  # Numeric settings ...
            ax.errorbar(range(len(xs)), means, yerr=stds, marker="o", capsize=4)  # ... line plot with dots and error bars, so you see the trend as the setting grows (e.g. LoRA rank). Points are evenly spaced (0, 1, 2 ...) because values like 1 and 64 are far apart.
        else:  # Text settings (categories) ...
            ax.bar(range(len(xs)), means, yerr=stds, capsize=4, color="tab:gray")  # ... grey bar chart with error bars; a line would wrongly suggest an order between categories.
        ax.set_xticks(range(len(xs)), xs)  # Put the setting labels under the points/bars.
        ax.set_title(f"{NAMES[m]}: {label}")  # Panel title, e.g. "LoRA: rank r".
        ax.set_ylabel("RTE val acc (%)")  # Y-axis label.
        ax.grid(alpha=0.3)  # Light, semi-transparent grid lines to make values easier to read.
    for ax in list(axes.flat)[len(panels):]:  # Any plot areas left over in the grid (if there were fewer than 6 panels) ...
        ax.axis("off")  # ... are hidden, so no empty boxes appear.
    fig.suptitle("Ablations on RTE (full data), mean ± std over 3 seeds")  # Title for the whole figure.
    fig.tight_layout()  # Automatically adjust spacing so titles and labels don't overlap.
    fig.savefig(FIG / "ablations.png", dpi=150)  # Save the figure as analysis/figures/ablations.png at 150 dots per inch (sharp enough for the report).


if __name__ == "__main__":  # True only when this file is run directly (python analysis/extra_analysis.py), not when imported.
    lr_search(_final)  # Build and save the LR-search table, using the final runs loaded at the top.
    ablations(_final)  # Build and save the ablation tables and figure.
