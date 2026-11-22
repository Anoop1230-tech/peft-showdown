"""Task metrics for the Hugging Face Trainer.

accuracy for every task, plus F1 (positive class = paraphrase) for MRPC,
which is the official GLUE metric for it. Values are fractions in [0, 1].
"""  # DOCSTRING: description of this whole file (shown by help()); it does not run anything. This file's job: turn the model's raw outputs into scores (accuracy, and F1 for MRPC) that the Trainer can log and use to pick the best epoch.

from __future__ import annotations  # Makes Python treat type hints as plain text instead of evaluating them, so modern hint syntax works on older Python versions. Does not change how the code runs.

import numpy as np  # NumPy: fast maths on arrays. Used for np.argmax, which picks the predicted class from the model's scores. Without it the prediction line below fails with NameError.
from sklearn.metrics import accuracy_score, f1_score  # Ready-made, well-tested metric functions from scikit-learn. accuracy_score = fraction of correct answers; f1_score = balance of precision and recall. Using them avoids writing (and possibly getting wrong) our own formulas.

F1_TASKS = {"mrpc"}  # Set of tasks that ALSO need F1. Only MRPC, because F1 is MRPC's official GLUE metric (its classes are unbalanced: 68% are paraphrases, so accuracy alone can look good by always saying "yes"). If this were empty, MRPC results would have no F1 and validate() in results.py would reject them.


def compute_metrics(task: str):  # FUNCTION compute_metrics: a "factory" that BUILDS and returns the metric function for one task. Needed because the HF Trainer calls the metric function with only eval_pred (it never passes the task name), so the task has to be stored inside the returned function.
    """Return a compute_metrics(eval_pred) function for Trainer."""  # Help text.

    def _compute(eval_pred):  # INNER FUNCTION _compute: the function the Trainer actually calls after every evaluation (every epoch on dev, and at the end on validation). It can still see the variable "task" from the outer function (this is called a closure).
        logits, labels = eval_pred  # eval_pred contains two arrays: logits = the model's raw scores, shape (number of examples, 2), one score per class; labels = the correct answers, shape (number of examples,). Unpack them into two variables.
        if isinstance(logits, (tuple, list)):  # Some models (the AdapterHub heads) return a tuple of outputs instead of a single array ...
            logits = logits[0]  # ... the class scores are always the FIRST item, so take that. Without this, np.argmax would run on a tuple and crash, or give nonsense.
        preds = np.argmax(logits, axis=-1)  # For each example, pick the class with the HIGHEST score: axis=-1 means "along the class dimension". E.g. scores [-1.2, 2.3] -> class 1. Result: one predicted label (0 or 1) per example.
        out = {"accuracy": float(accuracy_score(labels, preds))}  # Accuracy = number of correct predictions / total, e.g. 0.761. float() converts the NumPy number to a plain Python float, because NumPy floats can fail when written to JSON.
        if task in F1_TASKS:  # Only for MRPC ...
            out["f1"] = float(f1_score(labels, preds))  # ... also compute F1 for class 1 (= "is a paraphrase"). F1 = 2 x precision x recall / (precision + recall). It punishes a model that just predicts "paraphrase" for everything less kindly than accuracy does.
        return out  # Give the scores to the Trainer. It adds the prefix "eval_" (eval_accuracy, eval_f1); train.py's metric_for_best_model="accuracy" uses eval_accuracy to choose the best epoch. Without return, the Trainer gets None and best-model selection crashes.

    return _compute  # Return the inner function itself (not its result). train.py passes it to the Trainer as compute_metrics=compute_metrics(args.task). Without this line, compute_metrics() would return None and no scores would ever be computed.
