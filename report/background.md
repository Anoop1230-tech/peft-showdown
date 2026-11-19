# Background and Related Work

Parameter-Efficient Fine-Tuning (PEFT) methods have emerged as an essential alternative to full fine-tuning for adapting large pretrained language models to downstream tasks. While traditional fine-tuning adjusts all model weights—demanding massive GPU memory for optimizer states and requiring a separate copy of the full model checkpoint (~500 MB to tens of gigabytes per task)—PEFT methods freeze the vast majority of the pretrained parameters and train only a small fraction (typically 0.1% to 3%) of parameters.

This project empirically benchmarks four foundational adaptation strategies across the PEFT taxonomy: **Full Fine-Tuning**, **Adapters** (Houlsby et al., 2019), **LoRA** (Hu et al., 2022), and **Prompt Tuning** (Lester et al., 2021).

---

## 1. Primary Literature Review

### 1.1 Bottleneck Adapters (Houlsby et al., 2019)
- **Paper:** *Parameter-Efficient Transfer Learning for NLP* (ICML 2019).
- **Core Concept:** Houlsby et al. introduced the adapter paradigm to transformer architectures. Instead of modifying pretrained weights, adapters insert compact, modular bottleneck sub-layers sequentially after transformer blocks.
- **Architecture:** Each adapter consists of a two-layer feed-forward network with a bottleneck projection:
  $$\text{Adapter}(x) = f(x W_{\text{down}} + b_{\text{down}}) W_{\text{up}} + b_{\text{up}} + x$$
  where $W_{\text{down}} \in \mathbb{R}^{d \times m}$ projects the $d$-dimensional representation to a lower dimension $m \ll d$ (controlled by a reduction factor $r = d/m$), $f(\cdot)$ is a non-linear activation (e.g., GeLU or Swish), and $W_{\text{up}} \in \mathbb{R}^{m \times d}$ restores dimension $d$. A residual skip-connection is maintained around the bottleneck projection.
- **Placement:** The original Houlsby configuration (`double_seq_bn`) inserts two adapters per transformer layer: one immediately following multi-head attention and projection, and a second following the feed-forward layer.
- **Key Findings:** On the GLUE benchmark, adapters achieved performance within 0.4–0.8% of full fine-tuning on BERT while updating only 0.5%–3.6% of the parameters. Subsequent work by Pfeiffer et al. (2020) proposed a single adapter per layer (`seq_bn`), reducing parameters further with comparable accuracy.
- **Limitations:** Because adapters add sequential operations to the computation graph, they introduce inference latency (typically 5%–20% slowdown) that cannot be folded back into the base weights.

### 1.2 Low-Rank Adaptation — LoRA (Hu et al., 2022)
- **Paper:** *LoRA: Low-Rank Adaptation of Large Language Models* (ICLR 2022).
- **Core Concept:** Hypothesizing that weight updates during downstream adaptation have a low "intrinsic dimension" (Aghajanyan et al., 2020), Hu et al. proposed parameterizing the weight update $\Delta W$ directly as the product of two low-rank matrices:
  $$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
  where $W_0 \in \mathbb{R}^{d \times k}$ is frozen, $A \in \mathbb{R}^{r \times k}$ is initialized with Gaussian noise $\mathcal{N}(0, \sigma^2)$, $B \in \mathbb{R}^{d \times r}$ is initialized with zeros, and $r \ll \min(d, k)$ is the rank.
- **Key Characteristics:**
  - **Zero Inference Overhead:** Because $W = W_0 + \frac{\alpha}{r} B A$ is purely linear, $\Delta W$ can be explicitly computed and added into $W_0$ during inference, introducing zero latency or memory overhead at runtime.
  - **Selective Application:** Typically applied to attention projection matrices ($W_q, W_v$ or all four $W_q, W_k, W_v, W_o$).
  - **Memory Efficiency:** Eliminates the need to maintain Adam optimizer states (first and second moments) for the frozen weights, drastically decreasing peak VRAM usage.
- **Key Findings:** On RoBERTa, DeBERTa, and GPT-2/3, LoRA matched or exceeded full fine-tuning performance while requiring orders of magnitude fewer parameters (~0.1%–0.8%) and 3× lower training memory.

### 1.3 Prompt Tuning (Lester et al., 2021)
- **Paper:** *The Power of Scale for Parameter-Efficient Prompt Tuning* (EMNLP 2021).
- **Core Concept:** Rather than modifying model parameters or inserting intermediate layers, prompt tuning prepends a small sequence of $p$ learnable continuous embedding vectors ("virtual tokens") to the input sequence:
  $$X_{\text{prompt}} = [P_1, P_2, \dots, P_p; E(x_1), E(x_2), \dots, E(x_n)]$$
  Only the virtual token embeddings $P \in \mathbb{R}^{p \times d}$ are optimized via backpropagation; the entire transformer backbone remains completely frozen.
- **Key Findings & Scaling Law:**
  - At small model scales (e.g., T5-Small ~60M, Base ~220M), prompt tuning lags significantly behind full fine-tuning (often by 5–15 accuracy points).
  - Crucially, Lester et al. demonstrated that **as model scale increases to billions of parameters (T5-XXL, 11B), prompt tuning closes the gap and matches full fine-tuning**.
  - On moderate models like `roberta-base` (125M parameters), prompt tuning is expected to be the least competitive method, providing a valuable validation of scale dependency.
- **Parameter Footprint:** Extreme parameter efficiency—updating only $p \times d$ parameters (e.g., 20 tokens $\times$ 768 dims = 15,360 parameters, <0.02% of the model).

---

## 2. Structural Comparison of PEFT Paradigms

| Dimension | Full Fine-Tuning | Houlsby Adapters | LoRA ($r=8$) | Prompt Tuning ($p=20$) |
|---|---|---|---|---|
| **Mechanism** | Update all $W$ | Bottleneck sub-layers | Low-rank additive $\Delta W = BA$ | Virtual token embeddings |
| **Trainable Params (approx)** | 124.6M (100%) | 2.38M (1.87%) | 0.89M (0.71%) | ~0.60M (0.48%) |
| **Trainable Modules** | Base model + head | Bottleneck MLPs + head | $A, B$ matrices + head | Virtual embeddings + head |
| **Inference Latency Overhead**| None | +5% to +15% (sequential MLPs) | None (merged at runtime) | Minimal (+20 tokens) |
| **Checkpoint Storage** | ~475 MB | ~9.1 MB | ~3.4 MB | ~2.3 MB |
| **Peak GPU Memory** | High (full optimizer state) | Moderate (frozen backbone) | Low (frozen backbone) | Lowest (frozen backbone) |
| **Scale Dependency** | Strong at all sizes | Strong at all sizes | Strong at all sizes | Requires large scale (>1B) |

---

## 3. Tooling and Library Ecosystem

While Hugging Face's `peft` library supports reparameterization methods (LoRA) and prompt-based methods (Prompt Tuning, Prefix Tuning), it does not provide native implementations of classical bottleneck adapters (Houlsby / Pfeiffer).

To maintain strict methodological fidelity:
- **Adapters** are implemented using the **AdapterHub `adapters` library** (formerly `adapter-transformers`), which provides modular support for `double_seq_bn` (Houlsby), `seq_bn` (Pfeiffer), and the specialized `AdapterTrainer`.
- **LoRA and Prompt Tuning** leverage Hugging Face `peft` with `LoraConfig` and `PromptTuningConfig`.
- All methods utilize identical base checkpoints (`roberta-base`), preprocessing splits, and evaluation metrics.
