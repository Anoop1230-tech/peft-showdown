# Methods

All four methods adapt the same pretrained encoder, `roberta-base` (12 layers, hidden size 768, 124.6M parameters), to each GLUE task by adding a classification head on top of the `<s>` token representation. The head is RoBERTa's standard two-layer classifier (a 768×768 dense layer with tanh, then a 768×2 output layer, 592,130 parameters). It is trained in every method, and it is counted in every trainable-parameter number we report, so that the counts are comparable.

All methods share the same training budget: AdamW with weight decay 0.01, linear warmup over 6% of steps, batch size 32, maximum sequence length 128, fp16, and the same number of epochs (3 for SST-2 full data, 10 for MRPC and RTE full data, 20 for every 1k-example run). The checkpoint with the best accuracy on our held-out 10% dev split is kept and evaluated on the GLUE validation split. Only the learning rate and the method-specific settings below differ between methods.

## Full fine-tuning

Full fine-tuning is the baseline. Every weight of RoBERTa and the classification head is updated, which gives **124,647,170 trainable parameters (100%)**. We use a learning rate of 2e-5, a standard value for fine-tuning `roberta-base` on GLUE.

Because every weight changes, each task needs its own full copy of the model. The saved checkpoint is **475.5 MB** per task. Full fine-tuning also needs the most GPU memory, because the optimizer keeps two AdamW moment estimates for each of the 124.6M parameters, in addition to their gradients.

## LoRA

LoRA (Hu et al., 2022) freezes the pretrained weights and learns a low-rank update for selected weight matrices. For a frozen weight matrix W ∈ ℝ^(d×k), the adapted layer computes

h = W x + (α / r) · B A x,

where A ∈ ℝ^(r×k) and B ∈ ℝ^(d×r) are the only trained parameters of that layer, and the rank r is much smaller than d and k. A is initialised randomly and B with zeros, so training starts exactly from the pretrained model. After training, B A can be merged into W, so LoRA adds no inference latency.

Following the original paper, we apply LoRA to the **query and value projections** of the self-attention block in all 12 layers, with **rank r = 8, scaling α = 16 and dropout 0.1** on the LoRA input. Each adapted 768×768 matrix gets 2 × 8 × 768 = 12,288 new parameters. Over 24 matrices that is 294,912 LoRA parameters. With the classification head the total is **887,042 trainable parameters (0.71% of the 125.5M parameters in the model)**. Two-thirds of the trainable parameters are in the head, not in LoRA itself.

We use a learning rate of 5e-4, in the range used by Hu et al. for `roberta-base`. LoRA needs a much larger learning rate than full fine-tuning, because B starts at zero and the update must grow from nothing. We implement LoRA with the Hugging Face `peft` library (`LoraConfig`, `task_type=SEQ_CLS`, which keeps the classification head trainable).

Only the LoRA matrices and the head need to be stored per task. The saved checkpoint is **3.4 MB**, about 140 times smaller than a full fine-tuned model. Peak GPU memory is also lower than for full fine-tuning (1,937 MiB vs 3,264 MiB on RTE), and training is about 30% faster (112 s vs 158 s for RTE full data on a T4), because gradients and optimizer states are only kept for the 0.9M trainable parameters. The frozen weights still need activations stored for the backward pass, so the memory and time savings are much smaller than the 140× parameter saving.

## Adapters

Adapters (Houlsby et al., 2019) freeze the pretrained transformer weights and inject small feed-forward bottleneck modules directly inside each transformer layer. Unlike LoRA—which modifies the existing attention projection weights through additive low-rank matrices—adapters introduce new sequential sub-layers into the model graph.

We follow the standard Houlsby architecture (`double_seq_bn`), which places **two bottleneck adapters per layer**:
1. After the multi-head self-attention projection and before the residual add and layer norm.
2. After the two-layer feed-forward network and before the second residual add and layer norm.

Each adapter module consists of:
- A down-projection matrix $W_{\text{down}} \in \mathbb{R}^{d \times m}$ reducing dimension from hidden size $d$ to bottleneck dimension $m$.
- A non-linear activation function $f(\cdot)$ (Swish/GeLU).
- An up-projection matrix $W_{\text{up}} \in \mathbb{R}^{m \times d}$ projecting back to dimension $d$.
- An internal residual skip-connection around the bottleneck:

$$h = x + f(x W_{\text{down}} + b_{\text{down}}) W_{\text{up}} + b_{\text{up}}$$

With hidden size $d = 768$ and reduction factor 16, the bottleneck dimension is $m = 768 / 16 = 48$. Each bottleneck module contains $768 \times 48 + 48 = 36,912$ down-projection parameters and $48 \times 768 + 768 = 37,632$ up-projection parameters (total 74,544 parameters per adapter). With two adapters per layer across all 12 transformer layers, this yields **1,789,056 adapter parameters**. Adding the classification head (592,130 parameters) results in **2,381,186 trainable parameters (1.87% of the 127.0M total parameters in the adapted model)**.

We train adapters using a learning rate of 3e-4, as recommended by Houlsby et al. and the AdapterHub benchmarks. Because bottleneck adapters are not supported in Hugging Face's `peft` library, we use AdapterHub's `adapters` library (`AutoAdapterModel`, `DoubleSeqBnConfig`, and `AdapterTrainer`).

At inference time, only the trained adapter weights and classification head need to be distributed per task. The checkpoint size on disk is **9.1 MB**, approximately 52× smaller than a full fine-tuning checkpoint (475.5 MB), enabling efficient modular multi-task deployment while sharing a single frozen copy of `roberta-base`.

## Prompt tuning

Prompt tuning (Lester et al., 2021) freezes the entirety of the pretrained transformer and prepends a small sequence of $p$ continuous, learnable virtual token embeddings directly to the input representation:

$$X_{\text{prompt}} = [P_1, P_2, \dots, P_p; E(x_1), E(x_2), \dots, E(x_n)]$$

where $P \in \mathbb{R}^{p \times d}$ are the only model parameters updated alongside the classification head. The input embedding layer, all 12 self-attention layers, and feed-forward layers remain completely frozen.

We adopt standard prompt tuning settings with **$p = 20$ virtual tokens** initialized randomly (`PromptTuningInit.RANDOM`) from a normal distribution. With hidden size $d = 768$, the virtual prompt comprises $20 \times 768 = 15,360$ parameters. Combined with the classification head (592,130 parameters), this results in **607,490 trainable parameters (0.48% of the 125.3M total parameters)**. Over 97% of the trainable parameters reside in the classification head itself, making prompt tuning the most parameter-lean adaptation strategy in our benchmark.

We use a learning rate of **1e-2** (0.01), approximately 500× larger than full fine-tuning. As shown by Lester et al., soft prompt embeddings require large gradient steps because they must learn task-guiding trajectories in embedding space from scratch while propagating through a completely static transformer backbone. We implement prompt tuning using Hugging Face's `peft` library (`PromptTuningConfig`, `task_type=SEQ_CLS`).

The saved checkpoint per task is **2.3 MB**, about 200× smaller than a full fine-tuning model. Because only 15k prompt weights receive gradients through the backbone, training memory and compute overhead are minimal.

### Scale-dependence and expectations on `roberta-base`

Lester et al. demonstrated that prompt tuning exhibits strong scale-dependence: while it matches full fine-tuning on massive 11-billion parameter models (T5-XXL), it lags significantly on smaller models (~100M–300M parameters). Because `roberta-base` possesses only 125M parameters, the model lacks the excess capacity needed to steer complex reasoning purely through 20 prompt tokens without adjusting internal attention weights. Consequently, we expect prompt tuning to be the least accurate of the four adaptation methods on GLUE. In our report, this is analyzed as empirical confirmation of Lester et al.'s scaling law rather than a defect in the implementation.
