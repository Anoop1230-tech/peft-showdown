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

Only the LoRA matrices and the head need to be stored per task. The saved checkpoint is **3.4 MB**, about 140 times smaller than a full fine-tuned model. Peak GPU memory is also lower than for full fine-tuning (about 1.9 GB vs 3.3 GB on RTE with 1k examples), because gradients and optimizer states are only kept for the 0.9M trainable parameters. The frozen weights still need activations stored for the backward pass, so the memory saving is smaller than the parameter saving.
