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


def build_lora(num_labels: int):
    raise NotImplementedError("LoRA builder: Anoop, issue #26")


def build_adapter(num_labels: int):
    raise NotImplementedError("Adapter builder: Chiranjeet, issue #28")


def build_prompt_tuning(num_labels: int):
    raise NotImplementedError("Prompt tuning builder: Praveen, issue #30")


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
    "lora": {},
    "adapter": {},
    "prompt_tuning": {},
}


def build_model(method: str, num_labels: int):
    if method not in BUILDERS:
        raise ValueError(f"unknown method {method!r}, expected one of {list(BUILDERS)}")
    return BUILDERS[method](num_labels)
