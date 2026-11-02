"""Task metrics for the Hugging Face Trainer.

accuracy for every task, plus F1 (positive class = paraphrase) for MRPC,
which is the official GLUE metric for it. Values are fractions in [0, 1].
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

F1_TASKS = {"mrpc"}


def compute_metrics(task: str):
    """Return a compute_metrics(eval_pred) function for Trainer."""

    def _compute(eval_pred):
        logits, labels = eval_pred
        # some models (e.g. AdapterHub heads) return a tuple; logits come first
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        out = {"accuracy": float(accuracy_score(labels, preds))}
        if task in F1_TASKS:
            out["f1"] = float(f1_score(labels, preds))
        return out

    return _compute
