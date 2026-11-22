"""Model builders, one per method. All use roberta-base.

    model = build_model("full_ft", num_labels=2)

Every builder must leave the classification head trainable, so that
trainable-parameter counts are comparable across methods.

Owners: full_ft = Deepak, lora = Anoop (#26), adapter = Chiranjeet (#28),
prompt_tuning = Praveen (#30).
"""  # DOCSTRING: description of this whole file; it does not run anything. This file's job: build the model for each of the 4 methods (Full FT, LoRA, Adapters, Prompt tuning), all starting from the same pretrained RoBERTa, and decide which weights are trained and which are frozen.

from __future__ import annotations  # Makes Python treat type hints as plain text, so modern hint syntax works on older Python versions. Does not change how the code runs.

from transformers import AutoModelForSequenceClassification  # Hugging Face class that downloads a pretrained model AND puts a classification head on top (RoBERTa's head: 768->768 dense + tanh, then 768->2). Used by full FT, LoRA and prompt tuning. Without it no model can be built.

MODEL_NAME = "roberta-base"  # The pretrained model used by ALL methods: 12 transformer layers, hidden size 768, about 125M parameters, already trained on lots of English text. Using the same base for every method is what makes the comparison fair. Changing this name changes every experiment.


def build_full_ft(num_labels: int, frozen_layers: int = 0):  # FUNCTION build_full_ft: builds the Full Fine-Tuning model, where EVERY weight is trained (124.6M parameters). This is the baseline the PEFT methods are compared against. frozen_layers > 0 is used only in the "partial fine-tuning" ablation.
    """Full fine-tuning: every weight is trainable.

    frozen_layers > 0 is the partial-FT ablation: the embeddings and the
    bottom `frozen_layers` transformer layers are frozen.
    """  # Help text.
    model = AutoModelForSequenceClassification.from_pretrained(  # Download (or load from cache) the pretrained RoBERTa weights and add a NEW classification head with random starting weights. By default every weight has requires_grad=True, i.e. everything will be trained.
        MODEL_NAME, num_labels=num_labels)  # num_labels sets the size of the head's output layer (2 = yes/no). A wrong value would give the wrong number of outputs.
    if frozen_layers > 0:  # Only in the ablation runs (3, 6 or 9 frozen layers). Normal full-FT runs have frozen_layers = 0 and skip this whole block.
        for p in model.roberta.embeddings.parameters():  # Loop over every weight tensor in the embedding layer (word embeddings + position embeddings + layer norm) ...
            p.requires_grad = False  # ... and FREEZE it: requires_grad=False means no gradient is computed and the optimizer never changes it. Freezing the embeddings makes sense because the bottom layers are frozen too.
        for layer in model.roberta.encoder.layer[:frozen_layers]:  # Loop over the LOWEST N transformer layers (layer[:3] = layers 0, 1, 2). Lower layers hold general language knowledge; upper layers are more task-specific.
            for p in layer.parameters():  # Every weight tensor inside that layer (attention + feed-forward) ...
                p.requires_grad = False  # ... freeze it. Without this, the "partial FT" ablation would be identical to full FT. Result: freezing 9 layers dropped RTE accuracy to 65.3%, much worse than LoRA with far fewer trained parameters.
    return model  # Give the model back to build_model / train.py. Without return, train.py would get None and crash.


LORA_CONFIG = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.1,  # Default LoRA settings. r = 8: rank of the small matrices (bigger r = more capacity and more parameters). lora_alpha = 16: scaling, the update is multiplied by alpha/r = 2. lora_dropout = 0.1: randomly drops 10% of the LoRA input during training to reduce overfitting.
               "target_modules": ["query", "value"]}  # Which layers get LoRA: the Query and Value projections inside every attention block (as recommended in the LoRA paper). Adding more modules (key, dense...) would add parameters.


def build_lora(num_labels: int, r: int = 8):  # FUNCTION build_lora: builds the LoRA model. RoBERTa is frozen; for each Query/Value weight W, LoRA learns two small matrices B (768 x r) and A (r x 768) and the layer computes W·x + (alpha/r)·B·A·x. Only A, B and the head are trained (887,042 params, 0.71%). r is changed only in the rank ablation.
    """LoRA (Hu et al., 2022) on the query and value projections.

    The base model is frozen. task_type=SEQ_CLS makes PEFT keep the
    classifier head trainable (modules_to_save), so trainable params are
    LoRA A/B matrices (294,912) + head (592,130) = 887,042.

    For the rank ablation, lora_alpha is kept at 2 * r so the scaling
    alpha / r stays 2 for every rank.
    """  # Help text with the parameter count.
    from peft import LoraConfig, TaskType, get_peft_model  # Import from the Hugging Face PEFT library HERE (inside the function) rather than at the top, so the other methods still work on a machine where peft is not installed.

    model = AutoModelForSequenceClassification.from_pretrained(  # Load the same pretrained RoBERTa + new classification head as full FT. At this point everything is still trainable; get_peft_model below freezes the base.
        MODEL_NAME, num_labels=num_labels)  # 2 output classes.
    lora_cfg = dict(LORA_CONFIG)  # Make a COPY of the default settings. Without copying, the .update() below would change the global LORA_CONFIG itself, and every later model (and METHOD_CONFIGS, which points to it) would silently get the wrong rank.
    if r != LORA_CONFIG["r"]:  # Rank ablation run (r = 1, 4, 16 or 64 instead of 8):
        lora_cfg.update(r=r, lora_alpha=2 * r)  # set the new rank AND alpha = 2r, so the scale alpha/r stays 2 for every rank. Otherwise changing r would also change how strong the update is, and we couldn't tell which of the two effects changed the accuracy.
    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, **lora_cfg)  # Build the PEFT config. task_type=SEQ_CLS (sequence classification) tells PEFT to keep the classifier head TRAINABLE. Without it, the randomly initialised head would stay frozen and accuracy would be close to random guessing.
    return get_peft_model(model, cfg)  # Wrap the model: PEFT inserts the A/B matrices into every query/value layer and freezes all original RoBERTa weights. B starts at zero, so at step 0 the model behaves exactly like pretrained RoBERTa. Without this call, nothing is frozen and it would just be full FT.


ADAPTER_CONFIG = {"config": "double_seq_bn", "reduction_factor": 16}  # Default adapter settings. "double_seq_bn" = Houlsby adapters: TWO per layer (after attention and after feed-forward) = 24 in total. reduction_factor 16 = the bottleneck has 768/16 = 48 units.


def build_adapter(num_labels: int, config: str = "double_seq_bn",  # FUNCTION build_adapter: builds the Adapters model. RoBERTa is frozen; small bottleneck networks (768 -> 48 -> 768 with a nonlinearity and a skip connection) are inserted inside every layer. Only these adapters and the head are trained (2,381,186 params, 1.87%).
                  reduction_factor: int = 16):  # reduction_factor controls the bottleneck size: larger factor = smaller bottleneck = fewer parameters. The ablation tried 8 (77.6% on RTE, best in the study) and 64 (70.2%).
    """Bottleneck adapter (Houlsby et al., 2019) via AdapterHub `adapters`.

    Inserts bottleneck MLPs after self-attention and feed-forward layers in
    every transformer layer. The base model is frozen.
    Trainable params: 1,789,056 adapter weights + 592,130 classification head = 2,381,186 (1.87%).
    """  # Help text with the parameter count.
    from adapters import AutoAdapterModel, DoubleSeqBnConfig, SeqBnConfig  # Import from the AdapterHub "adapters" library HERE (not at the top) so other methods work without it. We use AdapterHub because HF PEFT does not offer Houlsby bottleneck adapters.

    model = AutoAdapterModel.from_pretrained(  # Load AdapterHub's version of RoBERTa. It has the same pretrained weights, but supports adding named adapters and named classification heads.
        MODEL_NAME, num_labels=num_labels)  # 2 output classes.
    if "default" in model.heads:  # Some library versions automatically create a head named "default" ...
        model.delete_head("default")  # ... delete it, because we add our own head below. If we kept it there would be an extra unused head and the trainable-parameter count would be wrong.
    if config == "double_seq_bn":  # Houlsby type (the default):
        adapter_cfg = DoubleSeqBnConfig(reduction_factor=reduction_factor)  # adapters after BOTH the attention block and the feed-forward block of every layer (2 per layer).
    elif config == "seq_bn":  # Pfeiffer type (used in the ablation):
        adapter_cfg = SeqBnConfig(reduction_factor=reduction_factor)  # only ONE adapter per layer, after the feed-forward block. About 38% fewer parameters; in our ablation it was 2.2 points worse than Houlsby on RTE.
    else:  # Any other config name ...
        adapter_cfg = config  # ... pass the string to AdapterHub unchanged; it knows many named configs. An invalid name makes add_adapter below raise an error.
    model.add_adapter("glue", config=adapter_cfg)  # Insert the adapter layers into every transformer layer, under the name "glue". The name is used to select, train and save this adapter.
    model.add_classification_head("glue", num_labels=num_labels)  # Add a classification head with the SAME name, so it is linked to the adapter (trained and saved together). Without a head the model would have no output for our 2 classes.
    model.train_adapter("glue")  # Freeze EVERYTHING except adapter "glue" and its head, and mark that adapter as active. Without this call, all of RoBERTa would be trained too (= full FT plus adapters).
    return model  # Give the model back to train.py.


PROMPT_INIT_TEXT = "Classify the sentiment or relation of the following text:"  # Sentence used only in the "TEXT init" ablation: the soft prompt starts from the embeddings of these words instead of random vectors. Result: 53.8% on RTE vs 56.1% for random init, so it did not help.

PROMPT_TUNING_CONFIG = {  # Default prompt-tuning settings (also written into each result's method_config).
    "num_virtual_tokens": 20,  # 20 learnable "virtual tokens" are put in front of every input. Each is a 768-number vector, so 20 x 768 = 15,360 trainable prompt parameters.
    "prompt_tuning_init": "RANDOM",  # The virtual tokens start as random vectors.
}  # End of PROMPT_TUNING_CONFIG.


def build_prompt_tuning(num_labels: int, num_virtual_tokens: int = 20,  # FUNCTION build_prompt_tuning: builds the Prompt Tuning model (Praveen's method). RoBERTa is completely frozen; only 20 learnable vectors added in front of the input, plus the head, are trained (607,490 params, 0.48%, the fewest of all methods).
                        prompt_tuning_init: str = "RANDOM"):  # Both arguments are changed only in ablations: num_virtual_tokens = 5 / 50 / 100, prompt_tuning_init = "TEXT".
    """Prompt tuning (Lester et al., 2021) with soft virtual token embeddings.

    Prepends learnable continuous embedding vectors to the input sequence.
    The pretrained base model is frozen. Only the prompt embeddings (15,360)
    and the classifier head (592,130) are trained, totaling 607,490 params (0.48%).
    """  # Help text with the parameter count.
    from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model  # Import from HF PEFT HERE (not at the top) so other methods work without peft installed.

    model = AutoModelForSequenceClassification.from_pretrained(  # Load the same pretrained RoBERTa + new classification head as the other methods.
        MODEL_NAME, num_labels=num_labels)  # 2 output classes.
    if prompt_tuning_init == "RANDOM":  # Default setting:
        extra = {"prompt_tuning_init": PromptTuningInit.RANDOM}  # start the virtual tokens from random vectors.
    else:  # TEXT init (ablation):
        extra = {"prompt_tuning_init": PromptTuningInit.TEXT,  # start the virtual tokens from the embeddings of real words ...
                 "prompt_tuning_init_text": PROMPT_INIT_TEXT,  # ... namely the words of this sentence ...
                 "tokenizer_name_or_path": MODEL_NAME}  # ... using RoBERTa's tokenizer to turn the sentence into tokens. Without the tokenizer name, PEFT raises an error for TEXT init.
    cfg = PromptTuningConfig(  # Build the PEFT config for prompt tuning.
        task_type=TaskType.SEQ_CLS,  # Sequence classification: keeps the classifier head TRAINABLE. Without it, only the prompt would train and the random head would stay random, so accuracy would be near chance.
        num_virtual_tokens=num_virtual_tokens,  # How many virtual tokens. More tokens = more parameters AND longer inputs (more memory and time: 100 tokens used 2,626 MiB and 175 s) but no accuracy gain in our ablation.
        **extra,  # Add the initialisation settings chosen above.
    )  # End of config.
    return get_peft_model(model, cfg)  # Wrap the model: PEFT adds the trainable prompt embedding table, prepends it to every input, and freezes all RoBERTa weights. Without this call nothing is frozen and it would just be full FT.


BUILDERS = {  # Lookup table: method name -> the function that builds it. build_model() uses this instead of a long if/elif chain; adding a new method only needs one new line here.
    "full_ft": build_full_ft,  # Full fine-tuning (Deepak).
    "lora": build_lora,  # LoRA (Anoop).
    "adapter": build_adapter,  # Adapters (Chiranjeet).
    "prompt_tuning": build_prompt_tuning,  # Prompt tuning (Praveen).
}  # End of BUILDERS.

METHOD_CONFIGS = {  # Default settings of each method. train.py copies the right one into result["method_config"], so every result file records exactly which settings the method used.
    "full_ft": {},  # Full FT has no extra method settings (everything is trained).
    "lora": LORA_CONFIG,  # r, alpha, dropout, target modules.
    "adapter": ADAPTER_CONFIG,  # adapter type and reduction factor.
    "prompt_tuning": PROMPT_TUNING_CONFIG,  # number of virtual tokens and init.
}  # End of METHOD_CONFIGS.


def build_model(method: str, num_labels: int, **kwargs):  # FUNCTION build_model: the single entry point train.py uses to get a model, e.g. build_model("lora", 2) or build_model("lora", 2, r=4) for an ablation. **kwargs collects any extra settings and passes them on to the builder.
    if method not in BUILDERS:  # Safety check: is this a known method?
        raise ValueError(f"unknown method {method!r}, expected one of {list(BUILDERS)}")  # If not, stop with a clear message listing the valid names (instead of a confusing KeyError).
    return BUILDERS[method](num_labels, **kwargs)  # Look up the builder function for this method, call it with the number of labels and any extra settings, and return the model it builds.
