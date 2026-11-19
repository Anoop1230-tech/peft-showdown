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


def build_full_ft(num_labels: int):
    """Full fine-tuning: every weight is trainable."""
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels)


LORA_CONFIG = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.1,
               "target_modules": ["query", "value"]}


def build_lora(num_labels: int):
    """LoRA (Hu et al., 2022) on the query and value projections.

    The base model is frozen. task_type=SEQ_CLS makes PEFT keep the
    classifier head trainable (modules_to_save), so trainable params are
    LoRA A/B matrices (294,912) + head (592,130) = 887,042.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels)
    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, **LORA_CONFIG)
    return get_peft_model(model, cfg)


ADAPTER_CONFIG = {"config": "double_seq_bn", "reduction_factor": 16}


def build_adapter(num_labels: int, config: str = "double_seq_bn",
                  reduction_factor: int = 16):
    """Bottleneck adapter (Houlsby et al., 2019) via AdapterHub `adapters`.

    Inserts bottleneck MLPs after self-attention and feed-forward layers in
    every transformer layer. The base model is frozen.
    Trainable params: 1,789,056 adapter weights + 592,130 classification head = 2,381,186 (1.87%).
    """
    from adapters import AutoAdapterModel, DoubleSeqBnConfig, SeqBnConfig

    model = AutoAdapterModel.from_pretrained(
        MODEL_NAME, num_labels=num_labels)
    if "default" in model.heads:
        model.delete_head("default")
    if config == "double_seq_bn":
        adapter_cfg = DoubleSeqBnConfig(reduction_factor=reduction_factor)
    elif config == "seq_bn":
        adapter_cfg = SeqBnConfig(reduction_factor=reduction_factor)
    else:
        adapter_cfg = config
    model.add_adapter("glue", config=adapter_cfg)
    model.add_classification_head("glue", num_labels=num_labels)
    model.train_adapter("glue")
    return model


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
    init_mode = (PromptTuningInit.RANDOM if prompt_tuning_init == "RANDOM"
                 else PromptTuningInit.TEXT)
    cfg = PromptTuningConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=num_virtual_tokens,
        prompt_tuning_init=init_mode,
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
