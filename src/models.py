"""Model builders, one per method. All use roberta-base.

    model = build_model("full_ft", num_labels=2)

Every builder must leave the classification head trainable, so that
trainable-parameter counts are comparable across methods.

Owners: full_ft = Deepak, lora = Anoop (#26), adapter = Chiranjeet (#28),
prompt_tuning = Praveen (#30).
"""

from __future__ import annotations

from transformers import AutoModelForSequenceClassification

MODEL_NAME = "roberta-base"


def build_full_ft(num_labels: int, frozen_layers: int = 0):
    """Full fine-tuning: every weight is trainable.

    frozen_layers > 0 is the partial-FT ablation: the embeddings and the
    bottom `frozen_layers` transformer layers are frozen.
    """
    # Instantiates roberta-base with a standard two-layer classification head:
    # 768x768 dense projection + tanh + 768x2 linear head (592,130 parameters).
    # All 124.6M parameters across all layers are updated during training.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels)
    if frozen_layers > 0:
        for p in model.roberta.embeddings.parameters():
            p.requires_grad = False
        for layer in model.roberta.encoder.layer[:frozen_layers]:
            for p in layer.parameters():
                p.requires_grad = False
    return model


# Hu et al. (2022) found that adapting attention query and value matrices provides
# the best parameter-to-accuracy ratio. Rank r=8 with alpha=16 yields scaling alpha/r=2.0.
LORA_CONFIG = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.1,
               "target_modules": ["query", "value"]}


def build_lora(num_labels: int, r: int = 8):
    """LoRA (Hu et al., 2022) on the query and value projections.

    The base model is frozen. task_type=SEQ_CLS makes PEFT keep the
    classifier head trainable (modules_to_save), so trainable params are
    LoRA A/B matrices (294,912) + head (592,130) = 887,042.

    For the rank ablation, lora_alpha is kept at 2 * r so the scaling
    alpha / r stays 2 for every rank.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels)
    lora_cfg = dict(LORA_CONFIG)
    if r != LORA_CONFIG["r"]:
        lora_cfg.update(r=r, lora_alpha=2 * r)
    # TaskType.SEQ_CLS automatically retains gradients for the classifier head
    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, **lora_cfg)
    return get_peft_model(model, cfg)


# Reduction factor 16 down-projects hidden dimension 768 to bottleneck dimension 48 (768/16 = 48).
ADAPTER_CONFIG = {"config": "double_seq_bn", "reduction_factor": 16}


def build_adapter(num_labels: int, config: str = "double_seq_bn",
                  reduction_factor: int = 16):
    """Bottleneck adapter (Houlsby et al., 2019) via AdapterHub `adapters`.

    Inserts bottleneck MLPs after self-attention and feed-forward layers in
    every transformer layer. The base model is frozen.
    Trainable params: 1,789,056 adapter weights + 592,130 classification head = 2,381,186 (1.87%).
    """
    from adapters import AutoAdapterModel, DoubleSeqBnConfig, SeqBnConfig

    # Using AdapterHub since standard peft does not implement Houlsby bottleneck adapters.
    model = AutoAdapterModel.from_pretrained(
        MODEL_NAME, num_labels=num_labels)

    # Remove the pre-instantiated default head to prevent conflict with our custom task head.
    if "default" in model.heads:
        model.delete_head("default")

    if config == "double_seq_bn":
        adapter_cfg = DoubleSeqBnConfig(reduction_factor=reduction_factor)
    elif config == "seq_bn":
        adapter_cfg = SeqBnConfig(reduction_factor=reduction_factor)
    else:
        adapter_cfg = config

    # Houlsby architecture adds bottleneck MLPs after self-attention and feed-forward layers.
    model.add_adapter("glue", config=adapter_cfg)
    model.add_classification_head("glue", num_labels=num_labels)
    model.train_adapter("glue")  # Freezes base model, trains adapter + classifier head
    return model


# used for the random-vs-text init ablation
PROMPT_INIT_TEXT = "Classify the sentiment or relation of the following text:"

PROMPT_TUNING_CONFIG = {
    "num_virtual_tokens": 20,
    "prompt_tuning_init": "RANDOM",
}


def build_prompt_tuning(num_labels: int, num_virtual_tokens: int = 20,
                        prompt_tuning_init: str = "RANDOM"):
    """Prompt tuning (Lester et al., 2021) with soft virtual token embeddings.

    Prepends learnable continuous embedding vectors to the input sequence.
    The pretrained base model is frozen. Only the prompt embeddings (15,360)
    and the classifier head (592,130) are trained, totaling 607,490 params (0.48%).
    """
    from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels)
    if prompt_tuning_init == "RANDOM":
        extra = {"prompt_tuning_init": PromptTuningInit.RANDOM}
    else:
        # initialise the prompt from the embeddings of a short text
        extra = {"prompt_tuning_init": PromptTuningInit.TEXT,
                 "prompt_tuning_init_text": PROMPT_INIT_TEXT,
                 "tokenizer_name_or_path": MODEL_NAME}
    # Prepend virtual tokens; SEQ_CLS keeps classifier head trainable alongside prompt weights.
    cfg = PromptTuningConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=num_virtual_tokens,
        **extra,
    )
    return get_peft_model(model, cfg)


BUILDERS = {
    "full_ft": build_full_ft,
    "lora": build_lora,
    "adapter": build_adapter,
    "prompt_tuning": build_prompt_tuning,
}

# method-specific settings written to result["method_config"];
# each owner fills in theirs alongside the builder
METHOD_CONFIGS = {
    "full_ft": {},
    "lora": LORA_CONFIG,
    "adapter": ADAPTER_CONFIG,
    "prompt_tuning": PROMPT_TUNING_CONFIG,
}


def build_model(method: str, num_labels: int, **kwargs):
    if method not in BUILDERS:
        raise ValueError(f"unknown method {method!r}, expected one of {list(BUILDERS)}")
    return BUILDERS[method](num_labels, **kwargs)
