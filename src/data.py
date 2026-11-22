"""Load and tokenize the GLUE tasks used in the project (SST-2, MRPC, RTE).

Splits returned by `load_task`:
  train       90% of GLUE train (or 1,000 examples of it in the "1k" regime)
  dev         the other 10% of GLUE train, used for checkpoint selection
  validation  official GLUE validation split, the number we report

Both the 90/10 split and the 1k subset depend only on `seed`, so every method
sees exactly the same examples for a given (task, regime, seed).
"""  # DOCSTRING: description of this whole file; it does not run anything. This file's job: download a GLUE task, split it into train / dev / validation, optionally shrink train to 1,000 examples, and convert the text into token ids the model can read.

from __future__ import annotations  # Makes Python treat type hints as plain text, so modern hint syntax works on older Python versions. Does not change how the code runs.

from datasets import load_dataset  # Hugging Face "datasets" library function that downloads a dataset from the Hugging Face Hub (or loads it from the local cache after the first time). Without it we would have to download and parse GLUE by hand.

DATASET = "nyu-mll/glue"  # The name of the GLUE benchmark on the Hugging Face Hub. load_dataset uses this to know WHAT to download. A wrong name gives a "dataset not found" error.

TASK_KEYS = {  # For each task: the names of its text column(s). Different GLUE tasks store their text under different column names, so tok() below looks them up here. Without this table, tok() would not know which columns to read.
    "sst2": ("sentence", None),  # SST-2 (sentiment): ONE sentence per example, stored in column "sentence". The second entry is None because there is no second sentence.
    "mrpc": ("sentence1", "sentence2"),  # MRPC (paraphrase): TWO sentences per example; the label says whether they mean the same thing.
    "rte": ("sentence1", "sentence2"),  # RTE (entailment): TWO sentences; the label says whether sentence 1 implies sentence 2.
}  # End of TASK_KEYS.

NUM_LABELS = {"sst2": 2, "mrpc": 2, "rte": 2}  # Number of answer classes per task. All three are yes/no, so 2. train.py passes this to build_model so the classifier head has the right number of outputs. A wrong number gives a head of the wrong size and a crash (or wrong training).

LOW_DATA_SIZE = 1000  # How many training examples to keep in the "1k" (low-data) regime. This is the brief's "low-data regime (1k examples)" requirement. Changing it changes the size of the low-data experiment for every method.


def load_task(task: str, tokenizer, regime: str = "full", seed: int = 42,  # FUNCTION load_task: the main function of this file. Returns three ready-to-train datasets (train, dev, validation). task = "sst2"/"mrpc"/"rte"; tokenizer = RoBERTa's tokenizer (from train.py); regime = "full" or "1k"; seed = controls which examples go where.
              max_len: int = 128):  # max_len = the maximum number of tokens per example; longer inputs are cut off. 128 is enough for almost all GLUE sentences and keeps memory low.
    """Return (train, dev, validation) tokenized datasets for one GLUE task."""  # Help text.
    if task not in TASK_KEYS:  # Safety check: is the task one we support?
        raise ValueError(f"unknown task {task!r}, expected one of {list(TASK_KEYS)}")  # If not, stop immediately with a clear message listing the valid names. Without it, a typo would cause a confusing KeyError deep inside the code.
    if regime not in ("full", "1k"):  # Safety check: is the regime valid?
        raise ValueError(f"unknown regime {regime!r}, expected 'full' or '1k'")  # Without this, a typo like "1K" would silently fall through and train on the FULL data, giving wrong low-data results without any warning.

    ds = load_dataset(DATASET, task)  # Download (first time) or load from cache the chosen GLUE task. ds is a DatasetDict with splits "train", "validation" and "test" (the test split has hidden labels, so we never use it).
    k1, k2 = TASK_KEYS[task]  # Look up the text column names for this task, e.g. ("sentence1", "sentence2") for RTE or ("sentence", None) for SST-2.

    split = ds["train"].train_test_split(test_size=0.1, seed=seed)  # Split GLUE's training set: 90% for training, 10% held out as our own DEV set. We need our own dev set because GLUE test labels are hidden and validation must stay untouched for the final score. The seed makes the split identical for every method, so the comparison is fair.
    train, dev = split["train"], split["test"]  # train_test_split names the parts "train" and "test"; rename them to train and dev so their purpose is clear.
    if regime == "1k":  # Low-data experiment only:
        train = train.shuffle(seed=seed).select(range(LOW_DATA_SIZE))  # Shuffle the training part randomly (with the seed) and keep the first 1,000 examples. Because the seed is fixed, every method gets EXACTLY the same 1,000 examples; otherwise one method could get "easier" examples and the comparison would be unfair. The dev set is NOT shrunk, so best-epoch selection stays reliable.

    def tok(batch):  # INNER FUNCTION tok: converts one batch of examples from text into numbers (token ids) that RoBERTa understands.
        return tokenizer(batch[k1], batch[k2] if k2 else None,  # Give the tokenizer the first sentence(s), and the second sentence(s) for pair tasks (or None for SST-2). For pairs, RoBERTa's tokenizer joins them with special separator tokens so the model knows where one sentence ends. Without passing k2, MRPC and RTE would lose their second sentence and become impossible.
                         truncation=True, max_length=max_len)  # Cut any input longer than max_len (128) tokens. Without truncation, very long examples would make the batch huge (memory error) or exceed RoBERTa's 512-token limit (crash).

    drop = [c for c in ds["train"].column_names if c != "label"]  # List every column EXCEPT "label" (e.g. "sentence1", "sentence2", "idx"). These raw columns are removed after tokenizing, so only input_ids, attention_mask and label remain. This gives all three model types (HF, PEFT, AdapterHub) exactly the same inputs, and avoids the collator crashing on text columns.

    def prep(d):  # INNER FUNCTION prep: applies tok() to an entire split.
        return d.map(tok, batched=True, remove_columns=drop)  # .map runs tok over the whole dataset; batched=True processes many examples at once (much faster than one at a time); remove_columns drops the raw text columns. No padding here: DataCollatorWithPadding in train.py pads each batch only to its own longest example, which saves memory and time.

    return prep(train), prep(dev), prep(ds["validation"])  # Tokenize and return the three splits in this order: train (for learning), dev (for picking the best epoch and LR), validation (the official GLUE split, used ONLY for the final reported score).
