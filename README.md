# PEFT Showdown

Team 18 project comparing full fine-tuning, LoRA, adapters and prompt tuning on `roberta-base` across three GLUE tasks (SST-2, MRPC, RTE), with full-data and 1k-example regimes.

We measure accuracy (and F1 on MRPC), trainable parameters, peak GPU memory, training time and checkpoint size.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full plan, experiment grid and division of work.

**Team:** Deepak Mandal, Anoop Kumar, Chiranjeet Biswas, Byagari Praveen Kumar

## Code layout

```
src/data.py      load GLUE, 10% dev split, 1k subset (same examples for every method)
src/metrics.py   accuracy, plus F1 for MRPC
src/models.py    build_model(method, num_labels)
src/train.py     one run -> results/final/{method}_{task}_{regime}_seed{seed}.json
src/results.py   results JSON format, validation, loading (docs/results_schema.md)
scripts/run_method.sh   all 16 final runs for one method
scripts/run_extra.sh    LR search (lr) and ablations (ablation), see below
analysis/analysis.ipynb     main tables and plots (results/final/)
analysis/extra_analysis.py  LR-search and ablation tables + plot
configs/hyperparams.yaml    record of every setting used
report/final_report.md      project report (PDF and DOCX next to it)
```

Results: `analysis/unified_table.md`, `analysis/low_data_table.md`,
`analysis/lr_search_table.md`, `analysis/ablation_table.md`, figures in `analysis/figures/`.
Report: `report/final_report.pdf`. Slides: `report/slides.md`.

LR search and ablations (16 + 42 runs on Kaggle, about 40 + 90 min):

    python scripts/kaggle_push.py --extra lr
    python scripts/kaggle_push.py --extra ablation
    python analysis/extra_analysis.py

One run (from the repo root, not from `src/`):

```
python -m src.train --method full_ft --task rte --regime 1k --seed 1
```

If the result JSON already exists the run is skipped. Add `--overwrite` to redo it.

## How to run on Kaggle

Each method has 16 final runs:
- RTE and MRPC, full data and 1k, seeds 1, 2 and 3 (12 runs)
- SST-2 1k, seeds 1, 2 and 3 (3 runs)
- SST-2 full, seed 1 only (1 run, about an hour on a T4)

### One-time setup

1. Create a GitHub fine-grained token: GitHub → Settings → Developer settings →
   Fine-grained tokens → Generate. Repository access: only `Anoop1230-tech/peft-showdown`,
   permission **Contents: Read-only**.
2. On Kaggle, verify your phone number (Settings), otherwise the notebook has no internet.

### Each time

1. Kaggle → Create → New Notebook.
2. In the right panel, **Session options**:
   - Accelerator: **GPU T4 x1**. Not T4 x2: two GPUs change the batch split and break the memory numbers.
   - Internet: **On**.
3. **Add-ons → Secrets**: add `GITHUB_TOKEN` with your token and tick it for this notebook.
4. **Cell 1, setup.** Paste and run it:

   ```python
   from kaggle_secrets import UserSecretsClient
   t = UserSecretsClient().get_secret("GITHUB_TOKEN")
   !git clone -q https://{t}@github.com/Anoop1230-tech/peft-showdown.git
   # remove the token from .git/config so it doesn't end up in the notebook output
   !git -C peft-showdown remote set-url origin https://github.com/Anoop1230-tech/peft-showdown.git
   %cd peft-showdown
   !pip install -q -r requirements.txt
   # Kaggle's torchao 0.10 makes peft refuse to build LoRA / prompt tuning; we don't use it
   !pip uninstall -y -q torchao
   ```

5. **Smoke test, before your first full launch.** Run one short run in a temporary cell
   (RTE 1k, a few minutes on a T4; replace `full_ft` with your method):

   ```python
   !python -m src.train --method full_ft --task rte --regime 1k --seed 1
   !python src/results.py results/final/*.json
   ```

   It should print `ok`, and `trainable params` should match your issue. Then delete
   this cell; the commit in step 7 starts from a fresh machine and redoes the run anyway.
6. **Cell 2, all runs** (replace `full_ft` with your method):

   ```python
   !bash scripts/run_method.sh full_ft
   ```

   Don't run it interactively, the commit in the next step does that.
7. Click **Save Version → Save & Run All (Commit) → Save**. It keeps running with the
   browser closed, for up to 12 hours. You can follow the log under the version's **Logs**.
8. When it finishes, the end of the log shows `ok` for every result file and lists any
   failed runs. Open the version → **Output** → download the files in
   `peft-showdown/results/final/`.
9. On your own machine:

   ```
   git pull
   cp ~/Downloads/<method>_*.json results/final/
   python src/results.py results/final/*.json
   git add results/final/*.json
   git commit -m "<method> results"
   git push
   ```

If some runs failed or the session stopped, start a new version. Finished runs are
skipped only if their JSONs are in the repo, so push what you have first.

### Alternative: launch from your own machine (no browser, no GitHub token)

With the Kaggle CLI (`pip install kaggle`) and `~/.kaggle/kaggle.json` from
kaggle.com/settings → API (phone verification needed):

```
python scripts/kaggle_push.py full_ft --smoke    # RTE 1k seed 1 only
python scripts/kaggle_push.py full_ft            # all 16 runs
kaggle kernels status <kaggle-user>/peft-showdown-full-ft
kaggle kernels output <kaggle-user>/peft-showdown-full-ft -p out --file-pattern '\.json$'
```

It uploads your local repo files (including any result JSONs already there, so those
runs are skipped) as a private Kaggle script on a GPU. Commit before launching real
runs, otherwise `meta.git_commit` ends in `-dirty`.
