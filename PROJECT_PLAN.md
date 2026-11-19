# P18: Parameter-Efficient Fine-Tuning Showdown
## Full Fine-Tuning vs LoRA vs Adapters vs Prompt Tuning

**Team 18:** Deepak Mandal, Anoop Kumar, Chiranjeet Biswas, Byagari Praveen Kumar

---

## 1. What the project is asking for

| Item | What it means for us |
|---|---|
| **Problem** | Compare 4 ways of adapting a pretrained model to a task. Measure accuracy, number of trainable parameters, GPU memory, and training time. |
| **Model** | One encoder model for every method. Recommended: `roberta-base` (125M params). `bert-base-uncased` is an acceptable fallback. |
| **Datasets** | 3 GLUE tasks from Hugging Face: **SST-2** (sentiment, 67k train), **MRPC** (paraphrase, 3.7k train), **RTE** (entailment, 2.5k train). |
| **Methods** | 1. Full fine-tuning (baseline, all weights trained) 2. LoRA 3. Adapters (bottleneck layers) 4. Prompt tuning (soft prompts) |
| **"Same budget"** | Every method gets the same number of epochs, same max sequence length, same seeds, same hyperparameter-search size. Only method-specific settings change. |
| **Research angle 1** | One unified comparison table (all methods × all tasks × all metrics). |
| **Research angle 2** | Low-data regime: repeat everything with only **1,000 training examples** per task and see which method holds up best. |
| **Papers** | [1] LoRA (Hu et al., 2022) [2] Adapters (Houlsby et al., 2019) [3] Prompt tuning (Lester et al., 2021) |

### Key ideas in one paragraph each

- **Full fine-tuning:** update all 125M weights. Usually the most accurate, but uses the most memory and each task needs its own full copy of the model (~500 MB).
- **LoRA:** freeze the model. For chosen weight matrices W (attention query/value), learn a low-rank update `ΔW = B·A` where A is `r × d` and B is `d × r` with small r (e.g. 8). Only A and B are trained (~0.3M params).
- **Adapters:** freeze the model. Insert small bottleneck MLPs (down-project → nonlinearity → up-project, with residual) inside every transformer layer. Only adapters are trained (~1M params).
- **Prompt tuning:** freeze the model. Prepend N learnable "virtual token" embeddings (e.g. 20) to the input. Only those embeddings are trained (~15k params). Lester et al. show it closes the gap with full FT only at very large model sizes, so we **expect it to be weakest on roberta-base**. That is a valid finding we should explain, not a bug.

> In all PEFT methods the **classification head** (the final linear layer) is also trained. Count it in trainable parameters for every method so the comparison is fair.

### One tooling note

The Hugging Face **PEFT** library supports LoRA and Prompt Tuning. It does **not** provide Houlsby-style bottleneck adapters. For adapters use the **AdapterHub `adapters` library** (same Hugging Face ecosystem). Mention this in the report.

---

## 2. Experiment grid

```
4 methods × 3 tasks × 2 data regimes (full, 1k) × 3 seeds = 72 final runs
```

Plus a small learning-rate search per method (Section 4, Step 5).

**Metrics recorded for every run**

| Metric | How |
|---|---|
| Accuracy | SST-2, RTE, MRPC |
| F1 | MRPC (official GLUE metric) |
| Trainable params (count and % of total) | `sum(p.numel() for p in model.parameters() if p.requires_grad)` |
| Peak GPU memory (MB) | `torch.cuda.reset_peak_memory_stats()` before training, `torch.cuda.max_memory_allocated()` after |
| Training time (s) | wall-clock around `trainer.train()` |
| Saved checkpoint size (MB) | size of the saved trainable weights only |

Report every number as **mean ± std over 3 seeds**. RTE and MRPC are small and noisy, so single-seed results are not trustworthy.

**Evaluation split:** GLUE test labels are hidden, so report on the official **validation** split. For hyperparameter tuning, hold out 10% of the training set as our own dev set. Never tune on the validation split we report.

---

## 3. Compute plan

- Free options: Google Colab (T4, 16 GB) or Kaggle (T4 ×2 / P100, 30 h/week each). With 4 members that is ~120 GPU-hours/week on Kaggle alone.
- Use `fp16=True`, `max_length=128`, batch size 32.
- Rough cost on a T4 with roberta-base: SST-2 full data, full FT, 3 epochs ≈ 45-60 min. MRPC/RTE and all 1k runs take a few minutes each. PEFT runs are faster than full FT on the same data.
- SST-2 full-data runs are the expensive part. Start them early.
- Save every run's results to a JSON file and push to the shared repo, so Colab disconnects don't lose results.

---

## 4. Step-by-step procedure (everyone should read this)

### Step 1. Set up the shared repository (Week 1)

```
P18_PEFT_Showdown/
├── configs/            # one YAML per method (hyperparameters)
├── src/
│   ├── data.py         # load + tokenize GLUE, 1k subsampling
│   ├── models.py       # build_model(method, config) -> model
│   ├── metrics.py      # accuracy/F1, param count, memory, time
│   └── train.py        # one entry point: --method --task --regime --seed
├── results/            # one JSON per run
├── analysis/           # notebook: tables + plots
├── report/
└── requirements.txt
```

`requirements.txt`:
```
torch
transformers
datasets
evaluate
peft
adapters
accelerate
scikit-learn
pandas
matplotlib
```

Fix library versions with `pip freeze > requirements.txt` once everything works, so all 4 members get identical results.

### Step 2. Data pipeline (`src/data.py`)

```python
from datasets import load_dataset

TASK_KEYS = {"sst2": ("sentence", None),
             "mrpc": ("sentence1", "sentence2"),
             "rte":  ("sentence1", "sentence2")}

def load_task(task, tokenizer, regime="full", seed=42, max_len=128):
    ds = load_dataset("glue", task)
    k1, k2 = TASK_KEYS[task]

    split = ds["train"].train_test_split(test_size=0.1, seed=seed)  # our tuning dev set
    train, dev = split["train"], split["test"]
    if regime == "1k":
        train = train.shuffle(seed=seed).select(range(1000))

    def tok(b):
        return tokenizer(b[k1], b[k2] if k2 else None,
                         truncation=True, max_length=max_len)
    return (train.map(tok, batched=True),
            dev.map(tok, batched=True),
            ds["validation"].map(tok, batched=True))
```

The 1k subset must be **identical for every method** (same seed → same examples).

### Step 3. Model builders (`src/models.py`)

```python
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model

MODEL = "roberta-base"

def build_full_ft(num_labels):
    return AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)

def build_lora(num_labels, r=8, alpha=16):
    m = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)
    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=r, lora_alpha=alpha,
                     lora_dropout=0.1, target_modules=["query", "value"])
    return get_peft_model(m, cfg)          # classifier head stays trainable

def build_prompt_tuning(num_labels, n_tokens=20):
    m = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)
    cfg = PromptTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=n_tokens,
                             prompt_tuning_init=PromptTuningInit.RANDOM)
    return get_peft_model(m, cfg)

def build_adapter(num_labels, config="double_seq_bn"):  # double_seq_bn = Houlsby, seq_bn = Pfeiffer
    from adapters import AutoAdapterModel
    m = AutoAdapterModel.from_pretrained(MODEL)
    m.add_adapter("glue", config=config)
    m.add_classification_head("glue", num_labels=num_labels)
    m.train_adapter("glue")                # freezes base, trains adapter + head
    m.set_active_adapters("glue")
    return m                               # train with adapters.AdapterTrainer
```

### Step 4. Training script (`src/train.py`)

One script, called like:
```
python src/train.py --method lora --task rte --regime 1k --seed 1 --lr 5e-4
```

It must:
1. Set the seed (`transformers.set_seed(seed)`).
2. Build data and model.
3. Count trainable params.
4. Reset peak memory, start timer, train, stop timer, read peak memory.
5. Evaluate on GLUE validation.
6. Write `results/{method}_{task}_{regime}_seed{seed}.json` with all metrics and hyperparameters.

Shared training settings (the "same budget"):

| Setting | Full data | 1k regime |
|---|---|---|
| Epochs | 3 (SST-2), 10 (MRPC, RTE) | 20 |
| Batch size | 32 | 32 |
| Max length | 128 | 128 |
| Warmup | 6% of steps | 6% |
| Optimizer | AdamW, weight decay 0.01 | same |
| Seeds | 1, 2, 3 | 1, 2, 3 |
| Model selection | best checkpoint on our dev split | same |

### Step 5. Learning-rate search (same size for every method)

Try 3 learning rates per method, seed 1 only, on MRPC/RTE full data and our dev split. Pick the best and use it for all final runs.

| Method | LRs to try | Method-specific settings |
|---|---|---|
| Full FT | 1e-5, 2e-5, 3e-5 | none |
| LoRA | 1e-4, 3e-4, 5e-4 | r = 8, alpha = 16, targets = query, value |
| Adapters | 1e-4, 3e-4, 1e-3 | Houlsby (`double_seq_bn`), reduction factor 16 |
| Prompt tuning | 1e-3, 1e-2, 3e-2 | 20 virtual tokens, random init |

Prompt tuning needs much higher LRs than the others. This is expected.

### Step 6. Run the final grid

72 runs (Section 2). Each member runs the runs for their own method (Section 5), all using the same `train.py`.

### Step 7. Extra ablations (one per member, small)

- LoRA: rank r ∈ {1, 4, 8, 16, 64} on RTE/MRPC.
- Adapters: Houlsby vs Pfeiffer, reduction factor ∈ {8, 16, 64}.
- Prompt tuning: virtual tokens ∈ {5, 20, 50, 100}; random vs text init.
- Full FT: freeze bottom k layers (cheap "partial FT" baseline) or compare roberta-base vs bert-base.

### Step 8. Analysis

1. **Unified table**: rows = methods; columns = SST-2 / MRPC / RTE scores (full and 1k), trainable params, % params, peak memory, time, checkpoint size.
2. **Plots**:
   - Accuracy vs trainable parameters (log x-axis), one point per method per task.
   - Bar chart: full data vs 1k accuracy per method (shows low-data behaviour).
   - Bar charts: peak memory and training time.
   - Ablation curves (LoRA rank, adapter size, prompt length).
3. **Questions to answer in the report**:
   - How close does each PEFT method get to full FT, and at what parameter cost?
   - Which method loses the least accuracy when going from full data to 1k examples?
   - Why is prompt tuning weak at this model size (link to Lester et al.'s scale result)?
   - Which method would you recommend under a memory limit, a time limit, or very little data?

### Step 9. Report and presentation

Sections: Introduction → Background (3 papers) → Methods → Experimental setup → Results (unified table + plots) → Low-data analysis → Ablations → Discussion/limitations → Conclusion. Each member writes the parts about their own method and ablation.

---

## 5. Division of work among the 4 members

Each member **owns one method end-to-end** (implementation, LR search, 18 final runs, ablation, report section). Each member also has **one shared responsibility** so the common work is split evenly.

| Member | Method owned | Shared responsibility |
|---|---|---|
| **Deepak Mandal** (su-24039) | **Full fine-tuning** (baseline) | Repo setup, `data.py`, `train.py`, `metrics.py`. The common pipeline everyone else uses. |
| **Anoop Kumar** (su-24017) | **LoRA** | Experiment tracking: results JSON format, run checklist, checks that everyone uses identical settings and seeds. |
| **Chiranjeet Biswas** (su-24037) | **Adapters** (AdapterHub) | Literature/background section: summaries of the 3 papers, related work. |
| **Byagari Praveen Kumar** (su-24034) | **Prompt tuning** | Analysis notebook: unified table, all plots, presentation slides. |

Why this split is fair: full FT is the most expensive in GPU time, so the shared pipeline work is front-loaded and finishes in Week 1. Prompt tuning runs are the cheapest, which leaves time for the analysis/plots work. The names are only a suggestion, so swap roles if someone prefers a different method.

### Each member's checklist

**Deepak (Full FT + pipeline)**
- [x] Create repo, folder structure, `requirements.txt`
- [x] Write `data.py` (incl. identical 1k subsampling) and `metrics.py`
- [x] Write `train.py` with `--method` switch; test it end-to-end with full FT on RTE
- [x] LR search for full FT
- [x] 18 final runs (start SST-2 full-data runs early, they are the slowest)
- [ ] Ablation: freeze bottom-k layers, or BERT vs RoBERTa
- [x] Report: Methods → Full FT, Experimental setup

**Anoop (LoRA + tracking)**
- [x] `build_lora` in `models.py`; verify trainable params ≈ 0.3M + head
- [x] Define results JSON schema and a shared run tracker (Google Sheet with all 72 runs)
- [x] LR search for LoRA
- [x] 18 final runs
- [ ] Ablation: LoRA rank r ∈ {1, 4, 8, 16, 64}
- [x] Report: Methods → LoRA, rank ablation results

**Chiranjeet (Adapters + literature)**
- [x] `build_adapter` in `models.py` using `adapters` + `AdapterTrainer`; make sure `train.py` handles it
- [x] Verify base model is frozen and head is trainable
- [x] LR search for adapters
- [x] 18 final runs
- [ ] Ablation: Houlsby vs Pfeiffer, reduction factor ∈ {8, 16, 64}
- [x] Report: Background (3 papers), Methods → Adapters

**Praveen (Prompt tuning + analysis)**
- [x] `build_prompt_tuning` in `models.py`; verify only prompt embeddings + head train
- [x] LR search for prompt tuning
- [x] 18 final runs
- [ ] Ablation: virtual tokens ∈ {5, 20, 50, 100}, random vs text init
- [x] `analysis/` notebook: load all JSONs → unified table + all plots
- [x] Report: Methods → Prompt tuning, Results section, slides

---

## 6. Timeline (6 weeks, adjust to your deadline)

| Week | Goal | Who |
|---|---|---|
| 1 | Read the 3 papers. Repo + pipeline working with full FT on RTE. | All read; Deepak builds pipeline |
| 2 | Each method plugged into `models.py` and running on RTE 1k as a smoke test. | Each member for own method |
| 3 | LR search done; best LRs written into `configs/`. Start SST-2 full-data runs. | All |
| 4 | All 72 final runs finished; results in `results/`. | All |
| 5 | Ablations; analysis notebook produces table and plots. | All; Praveen leads analysis |
| 6 | Report, slides, rehearsal. | All |

**Weekly sync (30 min):** each person reports runs done, blockers, and any number that looks wrong.

---

## 7. Common mistakes to avoid

1. **Different 1k subsets per method.** Always sample with the same seed in `data.py`.
2. **Tuning on the validation split you report.** Use our 10% dev split for tuning.
3. **Forgetting the classifier head** in trainable-param counts, or accidentally freezing it in the adapter setup.
4. **Single-seed conclusions** on RTE/MRPC. Always use 3 seeds and report std.
5. **Measuring memory wrongly.** Reset peak stats before each run, and run one method at a time per GPU.
6. **Comparing times across different GPUs.** Record the GPU name in each result JSON; compare times only on the same GPU type.
7. **Using a low LR for prompt tuning.** It needs about 100× the full-FT LR.

---

## 8. Final deliverables

- [x] GitHub repo with code, configs, and `results/` JSONs
- [x] Unified comparison table (full data + 1k)
- [x] Plots: accuracy vs params, full vs 1k, memory, time, ablations
- [x] Written report (with citations to [1], [2], [3])
- [x] Presentation slides

## References

1. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022. https://arxiv.org/abs/2106.09685
2. Houlsby et al., *Parameter-Efficient Transfer Learning for NLP*, ICML 2019. https://arxiv.org/abs/1902.00751
3. Lester et al., *The Power of Scale for Parameter-Efficient Prompt Tuning*, EMNLP 2021. https://arxiv.org/abs/2104.08691
