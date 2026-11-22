"""Task metrics for the Hugging Face Trainer.

accuracy for every task, plus F1 (positive class = paraphrase) for MRPC,
which is the official GLUE metric for it. Values are fractions in [0, 1].
"""  # file description shown by help(); no effect on running the code

from __future__ import annotations  # lets type hints like "str | None" work on older Python; without it old Python may crash on hints

import numpy as np  # numpy for argmax over logits; without it np.argmax below fails (NameError)
from sklearn.metrics import accuracy_score, f1_score  # ready-made metric functions; without them we would have to code accuracy/F1 by hand

F1_TASKS = {"mrpc"}  # tasks that also need F1 (official GLUE metric for MRPC); if empty, MRPC would report only accuracy


def compute_metrics(task: str):  # factory: builds a metric function for one task; needed because Trainer only gives eval_pred, not the task name
    """Return a compute_metrics(eval_pred) function for Trainer."""  # short help text

    def _compute(eval_pred):  # the function Trainer calls after every evaluation; it "remembers" task from the outer function
        logits, labels = eval_pred  # unpack model scores (N x 2) and true labels (N); without this we cannot compare predictions with answers
        if isinstance(logits, (tuple, list)):  # AdapterHub heads return a tuple instead of one array
            logits = logits[0]  # take the logits (first item); without this argmax would run on a tuple and crash
        preds = np.argmax(logits, axis=-1)  # predicted class = column with the highest score; without it we have no predictions
        out = {"accuracy": float(accuracy_score(labels, preds))}  # fraction of correct predictions; float() so it saves to JSON (numpy floats can fail)
        if task in F1_TASKS:  # only MRPC gets F1
            out["f1"] = float(f1_score(labels, preds))  # F1 for class 1 (= paraphrase); without it MRPC results fail validation in results.py
        return out  # Trainer adds "eval_" in front, e.g. eval_accuracy; without return, Trainer gets None and best-model selection breaks

    return _compute  # give the inner function to Trainer; without it compute_metrics(task) returns None and no metrics are computed
