"""Load and tokenize the GLUE tasks used in the project (SST-2, MRPC, RTE).

Splits returned by `load_task`:
  train       90% of GLUE train (or 1,000 examples of it in the "1k" regime)
  dev         the other 10% of GLUE train, used for checkpoint selection
  validation  official GLUE validation split, the number we report

Both the 90/10 split and the 1k subset depend only on `seed`, so every method
sees exactly the same examples for a given (task, regime, seed).
"""  # file description; no effect on running the code

from __future__ import annotations  # allows modern type hints on older Python; without it old Python may crash on hints

from datasets import load_dataset  # Hugging Face function that downloads datasets; without it we cannot get GLUE

DATASET = "nyu-mll/glue"  # name of GLUE on the Hugging Face Hub; a wrong name -> download error

TASK_KEYS = {  # which text column(s) each task uses; without this tok() would not know which columns to read
    "sst2": ("sentence", None),  # SST-2: one sentence (sentiment), so the second column is None
    "mrpc": ("sentence1", "sentence2"),  # MRPC: sentence pair (paraphrase or not)
    "rte": ("sentence1", "sentence2"),  # RTE: sentence pair (does sentence 1 imply sentence 2?)
}  # end of TASK_KEYS

NUM_LABELS = {"sst2": 2, "mrpc": 2, "rte": 2}  # all tasks are yes/no -> 2 output classes; a wrong number gives a wrong-size classifier head

LOW_DATA_SIZE = 1000  # training size in the "1k" regime; change it and the low-data experiment uses a different size


def load_task(task: str, tokenizer, regime: str = "full", seed: int = 42,  # main function: returns tokenized (train, dev, validation)
              max_len: int = 128):  # max_len = longest input in tokens; longer text gets cut
    """Return (train, dev, validation) tokenized datasets for one GLUE task."""  # short help text
    if task not in TASK_KEYS:  # check the task name is one we support
        raise ValueError(f"unknown task {task!r}, expected one of {list(TASK_KEYS)}")  # stop with a clear message; without it a typo gives a confusing KeyError later
    if regime not in ("full", "1k"):  # check the regime name
        raise ValueError(f"unknown regime {regime!r}, expected 'full' or '1k'")  # without this a typo like "1K" would silently train on full data

    ds = load_dataset(DATASET, task)  # download (or load from cache) train / validation / test splits
    k1, k2 = TASK_KEYS[task]  # names of the first and second text columns (k2 may be None)

    split = ds["train"].train_test_split(test_size=0.1, seed=seed)  # 90% train / 10% dev; GLUE test labels are hidden so we need our own dev set; same seed -> same split for every method
    train, dev = split["train"], split["test"]  # give the two parts clear names
    if regime == "1k":  # low-data experiment
        train = train.shuffle(seed=seed).select(range(LOW_DATA_SIZE))  # keep 1,000 random examples; seeded so all methods get the SAME 1,000 (otherwise the comparison is unfair)

    def tok(batch):  # turns text into token ids for one batch
        return tokenizer(batch[k1], batch[k2] if k2 else None,  # pair tasks pass two sentences, SST-2 passes one; without k2 the pair tasks would lose sentence 2
                         truncation=True, max_length=max_len)  # cut inputs longer than max_len; without truncation long inputs crash or use too much memory

    drop = [c for c in ds["train"].column_names if c != "label"]  # raw columns to remove (text, idx); keep only label so every model type gets the same inputs; extra string columns would make the collator crash

    def prep(d):  # tokenizes a whole split
        return d.map(tok, batched=True, remove_columns=drop)  # run tok in batches (fast) and drop raw columns; no padding here, DataCollatorWithPadding pads each batch later

    return prep(train), prep(dev), prep(ds["validation"])  # three tokenized splits; GLUE validation is the score we report
