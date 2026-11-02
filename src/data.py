"""Load and tokenize the GLUE tasks used in the project (SST-2, MRPC, RTE).

Splits returned by `load_task`:
  train       90% of GLUE train (or 1,000 examples of it in the "1k" regime)
  dev         the other 10% of GLUE train, used for checkpoint selection
  validation  official GLUE validation split, the number we report

Both the 90/10 split and the 1k subset depend only on `seed`, so every method
sees exactly the same examples for a given (task, regime, seed).
"""

from __future__ import annotations

from datasets import load_dataset

DATASET = "nyu-mll/glue"

# text columns for each task (second one is None for single-sentence tasks)
TASK_KEYS = {
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "rte": ("sentence1", "sentence2"),
}

NUM_LABELS = {"sst2": 2, "mrpc": 2, "rte": 2}

LOW_DATA_SIZE = 1000


def load_task(task: str, tokenizer, regime: str = "full", seed: int = 42,
              max_len: int = 128):
    """Return (train, dev, validation) tokenized datasets for one GLUE task."""
    if task not in TASK_KEYS:
        raise ValueError(f"unknown task {task!r}, expected one of {list(TASK_KEYS)}")
    if regime not in ("full", "1k"):
        raise ValueError(f"unknown regime {regime!r}, expected 'full' or '1k'")

    ds = load_dataset(DATASET, task)
    k1, k2 = TASK_KEYS[task]

    split = ds["train"].train_test_split(test_size=0.1, seed=seed)
    train, dev = split["train"], split["test"]
    if regime == "1k":
        train = train.shuffle(seed=seed).select(range(LOW_DATA_SIZE))

    def tok(batch):
        return tokenizer(batch[k1], batch[k2] if k2 else None,
                         truncation=True, max_length=max_len)

    # keep only input_ids / attention_mask / label so the collator and every
    # model type (HF, PEFT, AdapterHub) get the same inputs
    drop = [c for c in ds["train"].column_names if c != "label"]

    def prep(d):
        return d.map(tok, batched=True, remove_columns=drop)

    return prep(train), prep(dev), prep(ds["validation"])
