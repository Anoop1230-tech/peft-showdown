"""Model builders, one per method. All use roberta-base.

    model = build_model("full_ft", num_labels=2)

Every builder must leave the classification head trainable, so that
trainable-parameter counts are comparable across methods.

Owners: full_ft = Deepak, lora = Anoop (#26), adapter = Chiranjeet (#28),
prompt_tuning = Praveen (#30).
"""  # file description; no effect on running the code

from __future__ import annotations  # allows modern type hints on older Python; without it old Python may crash on hints

from transformers import AutoModelForSequenceClassification  # loads a pretrained model with a classifier head on top; without it no model can be built

MODEL_NAME = "roberta-base"  # 125M-parameter pretrained model shared by all 4 methods; changing it changes every experiment


def build_full_ft(num_labels: int, frozen_layers: int = 0):  # full fine-tuning: every weight is trained; frozen_layers > 0 only in the ablation
    """Full fine-tuning: every weight is trainable.

    frozen_layers > 0 is the partial-FT ablation: the embeddings and the
    bottom `frozen_layers` transformer layers are frozen.
    """  # help text
    model = AutoModelForSequenceClassification.from_pretrained(  # download RoBERTa weights + add a new random classifier head
        MODEL_NAME, num_labels=num_labels)  # num_labels sets the head output size (2); wrong value -> wrong output size
    if frozen_layers > 0:  # only for the "freeze bottom layers" ablation; normal runs skip this
        for p in model.roberta.embeddings.parameters():  # every weight of the word / position embeddings
            p.requires_grad = False  # freeze it (no gradient, not updated); without this the embeddings would still train
        for layer in model.roberta.encoder.layer[:frozen_layers]:  # the lowest N of the 12 transformer layers
            for p in layer.parameters():  # every weight inside that layer
                p.requires_grad = False  # freeze it; without this the "partial FT" ablation would be the same as full FT
    return model  # give the model to train.py; without return build_model gives None and training crashes


LORA_CONFIG = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.1,  # rank 8, scale alpha/r = 2, 10% dropout on the LoRA path (reduces overfitting)
               "target_modules": ["query", "value"]}  # add LoRA only to attention Q and V layers (as in the LoRA paper); more modules = more params


def build_lora(num_labels: int, r: int = 8):  # LoRA: freeze RoBERTa, learn small matrices B@A added to Q and V; r changes only in the rank ablation
    """LoRA (Hu et al., 2022) on the query and value projections.

    The base model is frozen. task_type=SEQ_CLS makes PEFT keep the
    classifier head trainable (modules_to_save), so trainable params are
    LoRA A/B matrices (294,912) + head (592,130) = 887,042.

    For the rank ablation, lora_alpha is kept at 2 * r so the scaling
    alpha / r stays 2 for every rank.
    """  # help text
    from peft import LoraConfig, TaskType, get_peft_model  # imported here, not at the top, so other methods work even if peft is missing

    model = AutoModelForSequenceClassification.from_pretrained(  # same pretrained RoBERTa + head as full FT
        MODEL_NAME, num_labels=num_labels)  # 2 output classes
    lora_cfg = dict(LORA_CONFIG)  # make a COPY; without copying, update() below would change the global default for later calls
    if r != LORA_CONFIG["r"]:  # rank ablation run (r = 1, 4, 16, 64)
        lora_cfg.update(r=r, lora_alpha=2 * r)  # keep alpha / r = 2; otherwise changing r would also change the update scale and mix two effects
    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, **lora_cfg)  # SEQ_CLS keeps the classifier head trainable; without it the random head stays frozen and accuracy is near chance
    return get_peft_model(model, cfg)  # insert LoRA layers and freeze the base; without this the whole model would train like full FT


ADAPTER_CONFIG = {"config": "double_seq_bn", "reduction_factor": 16}  # Houlsby adapters (2 per layer), bottleneck size 768/16 = 48


def build_adapter(num_labels: int, config: str = "double_seq_bn",  # adapters: freeze RoBERTa, insert small down->up MLPs in every layer
                  reduction_factor: int = 16):  # bigger reduction_factor = smaller bottleneck = fewer params
    """Bottleneck adapter (Houlsby et al., 2019) via AdapterHub `adapters`.

    Inserts bottleneck MLPs after self-attention and feed-forward layers in
    every transformer layer. The base model is frozen.
    Trainable params: 1,789,056 adapter weights + 592,130 classification head = 2,381,186 (1.87%).
    """  # help text
    from adapters import AutoAdapterModel, DoubleSeqBnConfig, SeqBnConfig  # imported here so other methods work even without the adapters library

    model = AutoAdapterModel.from_pretrained(  # AdapterHub's version of RoBERTa (supports adding adapters and named heads)
        MODEL_NAME, num_labels=num_labels)  # 2 output classes
    if "default" in model.heads:  # some versions create a head called "default" automatically
        model.delete_head("default")  # remove it; otherwise there would be an extra unused head and the param count would be wrong
    if config == "double_seq_bn":  # Houlsby type (default)
        adapter_cfg = DoubleSeqBnConfig(reduction_factor=reduction_factor)  # adapter after attention AND after feed-forward
    elif config == "seq_bn":  # Pfeiffer type (ablation)
        adapter_cfg = SeqBnConfig(reduction_factor=reduction_factor)  # only one adapter, after feed-forward -> about half the params
    else:  # any other name
        adapter_cfg = config  # pass the string to AdapterHub as-is; an invalid name makes add_adapter fail
    model.add_adapter("glue", config=adapter_cfg)  # insert the adapter layers, named "glue"
    model.add_classification_head("glue", num_labels=num_labels)  # new classifier head with the same name; without it the model has no output for our labels
    model.train_adapter("glue")  # freeze everything except adapter "glue" and its head; without it all of RoBERTa would train too
    return model  # give the model to train.py


PROMPT_INIT_TEXT = "Classify the sentiment or relation of the following text:"  # text for the TEXT-init ablation; its token embeddings become the starting prompt

PROMPT_TUNING_CONFIG = {  # default prompt-tuning settings (saved in the result JSON)
    "num_virtual_tokens": 20,  # 20 learnable "soft tokens" put in front of every input (20 x 768 = 15,360 params)
    "prompt_tuning_init": "RANDOM",  # start the soft tokens from random vectors
}  # end of PROMPT_TUNING_CONFIG


def build_prompt_tuning(num_labels: int, num_virtual_tokens: int = 20,  # prompt tuning: freeze RoBERTa, learn only soft tokens + head
                        prompt_tuning_init: str = "RANDOM"):  # both arguments change only in ablations
    """Prompt tuning (Lester et al., 2021) with soft virtual token embeddings.

    Prepends learnable continuous embedding vectors to the input sequence.
    The pretrained base model is frozen. Only the prompt embeddings (15,360)
    and the classifier head (592,130) are trained, totaling 607,490 params (0.48%).
    """  # help text
    from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model  # imported here so other methods work even without peft

    model = AutoModelForSequenceClassification.from_pretrained(  # same pretrained RoBERTa + head
        MODEL_NAME, num_labels=num_labels)  # 2 output classes
    if prompt_tuning_init == "RANDOM":  # default setting
        extra = {"prompt_tuning_init": PromptTuningInit.RANDOM}  # random starting vectors for the soft tokens
    else:  # TEXT init ablation
        extra = {"prompt_tuning_init": PromptTuningInit.TEXT,  # start the soft tokens from the embeddings of a real sentence
                 "prompt_tuning_init_text": PROMPT_INIT_TEXT,  # the sentence to use
                 "tokenizer_name_or_path": MODEL_NAME}  # tokenizer to turn that sentence into tokens; without it PEFT raises an error
    cfg = PromptTuningConfig(  # describe the prompt-tuning setup
        task_type=TaskType.SEQ_CLS,  # keeps the classifier head trainable; without it only the prompt trains and the random head stays random
        num_virtual_tokens=num_virtual_tokens,  # how many soft tokens; more tokens = more params and longer inputs
        **extra,  # add the init settings chosen above
    )  # end of config
    return get_peft_model(model, cfg)  # add the soft-prompt layer and freeze the base; without it nothing is frozen (= full FT)


BUILDERS = {  # method name -> builder function; build_model uses this to pick the right one
    "full_ft": build_full_ft,  # full fine-tuning
    "lora": build_lora,  # LoRA
    "adapter": build_adapter,  # adapters
    "prompt_tuning": build_prompt_tuning,  # prompt tuning
}  # end of BUILDERS

METHOD_CONFIGS = {  # default settings of each method, written into result["method_config"] for the record
    "full_ft": {},  # full FT has no extra settings
    "lora": LORA_CONFIG,  # LoRA settings
    "adapter": ADAPTER_CONFIG,  # adapter settings
    "prompt_tuning": PROMPT_TUNING_CONFIG,  # prompt-tuning settings
}  # end of METHOD_CONFIGS


def build_model(method: str, num_labels: int, **kwargs):  # single entry point used by train.py; kwargs carry ablation settings like r=4
    if method not in BUILDERS:  # unknown method name?
        raise ValueError(f"unknown method {method!r}, expected one of {list(BUILDERS)}")  # clear error; without it you get a confusing KeyError
    return BUILDERS[method](num_labels, **kwargs)  # call the matching builder and return its model
