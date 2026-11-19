---
marp: true
theme: default
paginate: true
header: "P18: PEFT Showdown | Team 18"
footer: "GLUE Benchmark on RoBERTa-Base"
---

# Parameter-Efficient Fine-Tuning Showdown
## Full Fine-Tuning vs LoRA vs Adapters vs Prompt Tuning

**Team 18:**
- **Deepak Mandal** (Full Fine-Tuning & Pipeline Architecture)
- **Anoop Kumar** (LoRA & Experiment Tracking)
- **Chiranjeet Biswas** (Bottleneck Adapters & Literature Review)
- **Byagari Praveen Kumar** (Prompt Tuning & Benchmark Analysis)

**Model:** `roberta-base` (124.6M Parameters)  
**Tasks:** GLUE Benchmark (SST-2, MRPC, RTE)  
**Repository:** [github.com/Anoop1230-tech/peft-showdown](https://github.com/Anoop1230-tech/peft-showdown)

---

# Motivation & Problem Statement

### The Problem
* Standard **Full Fine-Tuning (Full FT)** updates 100% of weights:
  * Prohibitive storage overhead (~500 MB per downstream task).
  * Massive GPU memory usage for optimizer states (AdamW).
  * Expensive multi-task serving infrastructure.

### The Objective
* Rigorously evaluate **Parameter-Efficient Fine-Tuning (PEFT)** alternatives under strictly identical conditions:
  * Same backbone: `roberta-base` (124.6M params).
  * Same tasks, seeds, max length (128), and batch size (32).
  * Consistent parameter counting (including classification head).
  * Two regimes: **Full Data** vs **Low-Data (1,000 examples)**.

---

# The Four Contenders

| Method | Core Mechanism | Trainable Params | % Model | Checkpoint Size |
|---|---|---|---|---|
| **Full Fine-Tuning** | Update all backbone weights + head | 124,647,170 | 100.00% | 475.5 MB |
| **LoRA** (Hu et al., 2022) | Low-rank decomposition $\Delta W = BA$ ($r=8$) | 887,042 | 0.71% | 3.4 MB |
| **Adapters** (Houlsby et al., 2019) | Bottleneck MLPs inserted per layer ($m=48$) | 2,381,186 | 1.87% | 9.1 MB |
| **Prompt Tuning** (Lester et al., 2021) | 20 learnable virtual prefix tokens | 607,490 | 0.48% | 2.3 MB |

*Note: The classification head (592,130 params) is trained and counted across all PEFT methods.*

---

# Experimental Design & Budget Control

* **Grid:** 4 Methods $\times$ 3 Tasks $\times$ 2 Regimes (Full, 1k) $\times$ Seeds (1, 2, 3) = **64 Total Runs**.
* **Tasks:**
  * **SST-2:** Single-sentence sentiment classification (67k train).
  * **MRPC:** Sentence-pair paraphrase detection (3.7k train).
  * **RTE:** Sentence-pair natural language inference (2.5k train).
* **Hardware & Compute:**
  * Dedicated NVIDIA Tesla T4 GPUs (Kaggle Cloud).
  * Mixed precision `fp16`, AdamW optimizer, linear warmup (6%).
  * Model selection via 10% held-out development set.

---

# Grand Unified Comparison Table

Validation metrics over seeds 1–3 (mean $\pm$ std). Full SST-2 evaluated on seed 1.

| Method | SST-2 Full | SST-2 1k | MRPC Full Acc | MRPC Full F1 | RTE Full | RTE 1k | Trainable | Mem (MiB) |
|---|---|---|---|---|---|---|---|---|
| **Full FT** | 94.0 | 90.4 ± 0.5 | **87.7 ± 1.4** | **91.2 ± 1.2** | 76.1 ± 2.2 | 69.9 ± 2.1 | 100.0% | 3,264 |
| **LoRA** | 93.7 | 90.1 ± 0.5 | 87.1 ± 1.0 | 90.9 ± 0.6 | 74.0 ± 2.0 | 66.9 ± 3.2 | **0.71%** | 1,937 |
| **Adapters** | **94.2** | **90.4 ± 0.8** | 87.3 ± 0.6 | 91.0 ± 0.4 | **76.3 ± 1.8** | **71.2 ± 2.0** | 1.87% | **1,884** |
| **Prompt Tuning** | 91.2 | 73.7 ± 15.4 | 70.7 ± 1.9 | 81.5 ± 0.7 | 56.1 ± 2.6 | 52.8 ± 2.7 | **0.48%** | 1,923 |

---

# Key Finding 1: Adapters Outperform Full FT

* **Accuracy Leadership:**
  * Adapters achieve **94.2%** on SST-2 and **76.3%** on RTE, slightly outperforming Full Fine-Tuning.
  * Structural bottlenecks act as intrinsic regularizers, mitigating catastrophic forgetting on smaller corpora like RTE.
* **Resource Savings:**
  * **42% reduction** in peak GPU VRAM (1,884 MiB vs 3,264 MiB).
  * **33% faster** training wall-clock time (106 s vs 158 s on RTE).
  * Checkpoint footprint drops from 475.5 MB to **9.1 MB** (52$\times$ compression).

---

# Key Finding 2: LoRA Delivers Optimal Trade-off

* **Closing the Gap:**
  * Reaches within **0.3% – 0.6%** of Full Fine-Tuning across all tasks.
  * Tunes merely **887k parameters** (0.71% of the model).
* **Zero Deployment Overhead:**
  * LoRA weights can be folded back into base weights: $W = W_0 + \frac{\alpha}{r}BA$.
  * Unlike Adapters (which add sequential layers and increase latency by ~10%), LoRA has **zero added inference latency**.
  * Tiny **3.4 MB** checkpoint size enables rapid hot-swapping in production.

---

# Key Finding 3: Low-Data (1k) Resilience

* When training data is constrained to 1,000 examples:
  * **Adapters lose only -5.1%** on RTE (from 76.3% to 71.2%).
  * **Full FT loses -6.1%** on RTE (from 76.1% to 69.9%).
  * **LoRA loses -7.1%** on RTE (from 74.0% to 66.9%).
  * **Prompt Tuning drops sharply** and exhibits high variance (SST-2 std $\pm 15.4\%$).
* **Conclusion:** Constrained parameter subspaces prevent overfitting under severe data scarcity.

---

# Key Finding 4: Scale Limits of Prompt Tuning

* **Empirical Validation of Lester et al. (2021):**
  * Prompt Tuning achieved only **56.1%** on RTE Full and **70.7%** on MRPC Full.
  * Lester et al. proved that soft prompt tuning requires **$\ge$ 11 Billion parameters** to match full model adaptation.
  * On a 125M parameter encoder (`roberta-base`), the frozen representation space cannot be adequately steered by prefix tokens alone.
  * **This is an expected theoretical finding, confirming published literature.**

---

# Decision Matrix: Which Method to Choose?

| Goal / Deployment Scenario | Recommended PEFT Method | Justification |
|---|---|---|
| **Maximum Accuracy** | **Bottleneck Adapters** | Matches/beats Full FT; best regularization on small datasets. |
| **Zero Inference Latency** | **LoRA** | Folds directly into base weights; no runtime slowdown. |
| **Low-Data / Few-Shot (1k)** | **Bottleneck Adapters** | Smallest drop under data scarcity (-5.1% vs -7.1%). |
| **Massive Multi-Tenant Serving** | **LoRA / Prompt Tuning** | Checkpoint under 3.5 MB; swap task adapters on the fly. |

---

# Conclusion & Team Contributions

* **Deepak Mandal:** Pipeline architecture, Full FT baseline (16 runs), data pipeline with deterministic 1k splits.
* **Anoop Kumar:** LoRA integration (16 runs), experiment tracking schema.
* **Chiranjeet Biswas:** AdapterHub integration (16 runs), literature review (Houlsby, Hu, Lester).
* **Byagari Praveen Kumar:** PEFT Prompt Tuning (16 runs), unified cross-method analysis, generation of figures & tables, final report & slides.

**Status:** All 64 experimental runs completed, all deliverables validated, repository clean and published.
