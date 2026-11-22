"""Train and evaluate one run, then save its result JSON.

Run from the repo root (otherwise `from src...` imports fail):

    python -m src.train --method full_ft --task rte --regime 1k --seed 1

Writes results/final/{method}_{task}_{regime}_seed{seed}.json. If that file
already exists the run is skipped (use --overwrite to redo it), so a restarted
run_method.sh picks up where it stopped.

One run per process: peak GPU memory is only comparable that way.
"""  # DOCSTRING: description of this whole file, also shown by --help (description=__doc__ below). This file's job: do ONE complete experiment (one method, one task, one data regime, one seed): load data, build the model, train, evaluate, measure the cost, and save the result JSON.

from __future__ import annotations  # Makes Python treat type hints (like "float | None") as plain text, so modern hint syntax works on older Python versions. Does not change how the code runs.

import os  # Operating-system tools: environment variables (os.environ) and joining file paths (os.path.join). Imported first because the next two lines must run BEFORE torch is imported.

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # Let PyTorch see ONLY GPU number 0. On Kaggle's "T4 x2" machine the Trainer would otherwise split every batch across 2 GPUs, which halves per-GPU memory and makes our peak-memory numbers wrong and not comparable. setdefault = only set it if the user hasn't already. Must happen before "import torch".
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # Turn off the tokenizer library's internal multi-threading. Without it, a long warning about "forking after parallelism" is printed many times in the log. Our data is small, so we lose no real speed.

import argparse  # Standard library for reading command-line options like --method lora --task rte. Turns them into args.method, args.task, etc.
import shutil  # Standard library for file operations; used for shutil.rmtree, which deletes the temporary checkpoint folder at the end.
import tempfile  # Standard library for creating a temporary folder (tempfile.mkdtemp) where the Trainer can save checkpoints during the run.
import time  # Standard library for clock time; time.time() is read before and after training to measure training time.

import torch  # PyTorch, the deep-learning library everything runs on. Used here to check whether a GPU exists and to reset / read GPU memory statistics.
from transformers import (AutoTokenizer, DataCollatorWithPadding, Trainer,  # From Hugging Face Transformers: AutoTokenizer = turns text into token ids; DataCollatorWithPadding = pads each batch to equal length; Trainer = the ready-made training loop (forward, backward, optimizer, evaluation, checkpoints).
                          TrainingArguments, set_seed)  # TrainingArguments = the object holding all training settings; set_seed = sets the seed of Python, NumPy and PyTorch in one call.

from src.data import NUM_LABELS, load_task  # Our own data code (src/data.py): NUM_LABELS = classes per task, load_task = returns tokenized train/dev/validation. "from src..." only works when run from the repo root with "python -m src.train"; otherwise you get "No module named src".
from src.metrics import compute_metrics  # Our metric function (src/metrics.py): accuracy, plus F1 for MRPC.
from src.models import MODEL_NAME, METHOD_CONFIGS, build_model  # Our model code (src/models.py): the base model name, each method's default settings, and the function that builds a model for a method.
from src.results import (METHODS, REGIMES, SEEDS, STAGES, TASKS, count_params,  # From src/results.py: the allowed values (used as argparse choices) and count_params (trainable / total parameters) ...
                         dir_size_mb, new_result, peak_gpu_mem_mb, result_path,  # ... dir_size_mb (checkpoint size), new_result (empty result dict), peak_gpu_mem_mb (peak memory), result_path (where the JSON goes) ...
                         save_result)  # ... and save_result (validates and writes the JSON safely).

LEARNING_RATES = {"full_ft": 2e-5, "lora": 5e-4, "adapter": 3e-4,  # Learning rate for each method in the final runs, taken from the method papers. Full FT needs a SMALL LR (2e-5): it changes all pretrained weights, and big steps would destroy what the model already knows. LoRA and adapters train small new parts, so they can use 10-25x larger LRs.
                  "prompt_tuning": 1e-2}  # Prompt tuning trains only 15,360 prompt numbers (+ head), so it needs a VERY large LR (1e-2) to learn anything in the given epochs. The later LR search confirmed every one of these values was the best of the 3 tried.
BATCH_SIZE = 32  # Number of training examples processed together in one optimizer step. SAME for every method (fair budget). Bigger batches use more GPU memory.
EVAL_BATCH_SIZE = 64  # Batch size during evaluation. Can be larger than training because evaluation stores no gradients, so it needs much less memory; larger = faster evaluation.
MAX_LENGTH = 128  # Maximum tokens per example; longer inputs are cut off (passed to load_task). Same for every method. Long enough for almost all GLUE sentences.
WARMUP_RATIO = 0.06  # During the first 6% of training steps, the learning rate rises linearly from 0 to its full value, then decreases linearly to 0. Warmup avoids large, damaging updates at the very start when the new head is still random.
WEIGHT_DECAY = 0.01  # AdamW weight decay: each step pulls weights slightly toward 0. A mild regulariser that reduces overfitting. Same for every method.

OWNERS = {"full_ft": "deepak", "lora": "anoop", "adapter": "chiranjeet",  # Team member responsible for each method. Written into result["meta"]["owner"] automatically, so nobody has to type --owner.
          "prompt_tuning": "praveen"}  # validate() in results.py rejects any result without an owner.


def num_epochs(task: str, regime: str) -> int:  # FUNCTION num_epochs: decides how many times the model goes over the whole training set (epochs), depending on task size. All methods get the SAME number, which is part of the shared budget.
    if regime == "1k":  # Low-data runs: 1,000 examples / batch 32 = only about 32 steps per epoch ...
        return 20  # ... so train for 20 epochs to get enough steps in total. Fewer epochs would leave the model under-trained.
    return 3 if task == "sst2" else 10  # Full data: SST-2 is large (about 60,000 training examples), so 3 epochs are already plenty of steps; MRPC and RTE are small (2-3 thousand examples), so they get 10 epochs.


def save_trainable_weights(model, method: str, out_dir: str) -> None:  # FUNCTION save_trainable_weights: saves only the weights you would actually need to deploy this task. The size of that folder is our "checkpoint size" metric (475.5 MB for full FT vs 2-9 MB for PEFT methods).
    """Save what you'd ship for this task (see docs/results_schema.md)."""  # Help text.
    if method == "adapter":  # AdapterHub models have their own saving function ...
        model.save_adapter(out_dir, "glue")  # ... which saves ONLY the adapter named "glue" and its head (about 9 MB). Using save_pretrained here would save the whole 475 MB model and make adapters look as expensive as full FT.
    else:  # Full FT, LoRA and prompt tuning:
        model.save_pretrained(out_dir)  # For full FT this saves the whole model (every weight was trained). For PEFT models (LoRA, prompt tuning) save_pretrained automatically saves only the LoRA/prompt weights and the head, not the frozen RoBERTa.


def best_epoch(trainer) -> float | None:  # FUNCTION best_epoch: finds out which epoch produced the best checkpoint (highest dev accuracy). Stored in the result; it showed, for example, that prompt tuning was still improving in the last epochs.
    """Epoch of the checkpoint picked by load_best_model_at_end."""  # Help text.
    ckpt = trainer.state.best_model_checkpoint  # The Trainer remembers the folder of the best checkpoint, e.g. "/tmp/.../trainer/checkpoint-1234".
    if not ckpt:  # If there is no best checkpoint (for example, evaluation never ran) ...
        return None  # ... return None instead of crashing on the line below.
    step = int(ckpt.rstrip("/").split("-")[-1])  # Get the training step from the folder name: remove a trailing "/", split on "-", take the last part and convert to a number: "checkpoint-1234" -> 1234.
    for entry in trainer.state.log_history:  # The Trainer keeps a log of every event (training loss, evaluations) during training. Go through it ...
        if entry.get("step") == step and "eval_accuracy" in entry:  # ... and find the EVALUATION entry made at that same step ...
            return entry["epoch"]  # ... and return its epoch number, e.g. 7.0.
    return None  # Not found in the log -> None (saved as null in the JSON).


def parse_args(argv=None):  # FUNCTION parse_args: defines every command-line option this script accepts and reads them. Returns an object where args.method, args.task, ... hold the values.
    p = argparse.ArgumentParser(description=__doc__,  # Create the parser; --help will show this file's docstring as the description.
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # Keep the docstring's line breaks and indentation when printing --help (otherwise argparse squashes it into one paragraph).
    p.add_argument("--method", required=True, choices=METHODS)  # Which method to train. Required. choices=METHODS means only the 4 valid names are accepted; a typo gives a clear error immediately instead of failing later.
    p.add_argument("--task", required=True, choices=TASKS)  # Which GLUE task: sst2, mrpc or rte. Required.
    p.add_argument("--regime", required=True, choices=REGIMES)  # Full data or 1k examples. Required.
    p.add_argument("--seed", required=True, type=int, choices=SEEDS)  # Random seed 1, 2 or 3. type=int converts the text "1" to the number 1. Different seeds give different data splits and starting weights, which shows how stable a method is.
    p.add_argument("--stage", choices=STAGES, default="final",  # Which experiment type: final (default), lr_search or ablation. Decides the results sub-folder and how the run is named.
                   help="stage: final (default), lr_search, or ablation")  # Text shown by --help.
    p.add_argument("--lr", type=float, default=None,  # A learning rate to try. Only used with --stage lr_search; otherwise the method's fixed LR from LEARNING_RATES is used.
                   help="learning rate for --stage lr_search (default: the method's final LR)")  # Text shown by --help.
    p.add_argument("--ablation-name", default=None,  # For ablation runs: WHICH setting to change, e.g. lora_rank, reduction_factor, num_virtual_tokens. argparse turns the "-" into "_", so it is read as args.ablation_name.
                   help="name of ablation parameter (e.g. reduction_factor, config)")  # Text shown by --help.
    p.add_argument("--ablation-value", default=None,  # For ablation runs: the NEW value for that setting, e.g. 4 or seq_bn. Read as text; converted to a number later if possible.
                   help="value of ablation parameter (e.g. 8, seq_bn)")  # Text shown by --help.
    p.add_argument("--owner", default=None,  # Optional: override who is recorded as the owner of this run. By default the method's owner from OWNERS is used.
                   help="defaults to the method's owner (meta.owner)")  # Text shown by --help.
    p.add_argument("--notes", default="", help="free text saved in meta.notes")  # Optional free-text note stored in the result JSON (e.g. "rerun after crash").
    p.add_argument("--overwrite", action="store_true",  # A flag (no value needed): if given, args.overwrite = True and the run is done again even if its result file already exists.
                   help="redo the run even if its result JSON exists")  # Without --overwrite, finished runs are skipped, which lets a crashed batch restart cheaply.
    return p.parse_args(argv)  # Read and check the options. argv=None means "use the real command line"; tests could pass a list instead.


def run(args) -> None:  # FUNCTION run: the whole experiment for ONE combination of method, task, regime and seed. Steps: settings -> ablation change -> empty result -> skip check -> data + model -> train -> evaluate -> measure -> save JSON.
    fp16 = torch.cuda.is_available()  # Use 16-bit (half-precision) maths only when a GPU is available. fp16 roughly halves activation memory and speeds up training on a T4. On a CPU, fp16 is not supported and the Trainer would crash, so it becomes False there.
    hyperparams = {  # Collect all training settings in one dict. It is used to configure training AND saved in the result JSON, so the file shows exactly how the run was trained.
        "learning_rate": (args.lr if args.stage == "lr_search" and args.lr  # In an LR-search run, use the LR given with --lr ...
                          else LEARNING_RATES[args.method]),  # ... otherwise (final and ablation runs) use the method's fixed LR.
        "epochs": num_epochs(args.task, args.regime),  # 3, 10 or 20, depending on task and regime.
        "batch_size": BATCH_SIZE,  # 32.
        "max_length": MAX_LENGTH,  # 128 tokens.
        "warmup_ratio": WARMUP_RATIO,  # 6% warmup.
        "weight_decay": WEIGHT_DECAY,  # 0.01.
        "fp16": fp16,  # True on GPU, False on CPU.
    }  # End of hyperparams.

    ablation = None  # Description of the ablation change; stays None for normal (final / LR-search) runs.
    method_config = dict(METHOD_CONFIGS[args.method])  # A COPY of this method's default settings, e.g. {"r": 8, "lora_alpha": 16, ...}. It is saved in the result. Copying matters: changing it below must not change the shared default in models.py.
    model_kwargs = {}  # Extra arguments to pass to build_model. Empty = build the default model. The ablation code below fills it, e.g. {"r": 4}.
    if args.stage == "ablation" and args.ablation_name:  # Only for ablation runs that name a setting to change:
        val = args.ablation_value  # The new value, which arrives from the command line as TEXT, e.g. "8" or "seq_bn".
        try:  # Try to interpret it as a whole number ...
            val = int(val)  # ... "8" -> 8. Needed because the builders expect numbers (e.g. r=8, not r="8"), and a string would cause errors or wrong behaviour.
        except ValueError:  # Not a whole number ...
            try:  # ... maybe a decimal number ...
                val = float(val)  # ... "0.5" -> 0.5.
            except ValueError:  # Not a number at all ...
                pass  # ... keep it as text, e.g. "seq_bn" or "TEXT".
        ablation = {"name": args.ablation_name, "value": val}  # Record which setting was changed and to what. Saved in the result JSON and used in the file name (e.g. "..._lora_rank-4_seed1").
        if args.method == "full_ft" and args.ablation_name == "frozen_layers":  # Full FT ablation: freeze the bottom N layers.
            method_config["frozen_layers"] = val  # Record the change in the saved settings ...
            model_kwargs["frozen_layers"] = val  # ... and pass it to build_full_ft.
        elif args.method == "lora" and args.ablation_name == "lora_rank":  # LoRA ablation: change the rank r.
            method_config.update(r=val, lora_alpha=2 * val)  # Record the new rank and alpha = 2r (matching what build_lora does) ...
            model_kwargs["r"] = val  # ... and pass the rank to build_lora.
        elif args.method == "adapter":  # Adapter ablations (two kinds):
            if args.ablation_name == "reduction_factor":  # 1) bottleneck size:
                method_config["reduction_factor"] = val  # record it ...
                model_kwargs["reduction_factor"] = val  # ... and pass it to build_adapter.
            elif args.ablation_name in ("config", "adapter_config"):  # 2) adapter type (Houlsby vs Pfeiffer); two spellings accepted:
                method_config["config"] = val  # record it ...
                model_kwargs["config"] = val  # ... and pass it to build_adapter.
        elif args.method == "prompt_tuning":  # Prompt-tuning ablations (two kinds):
            if args.ablation_name in ("num_virtual_tokens", "n_tokens", "virtual_tokens"):  # 1) number of virtual tokens; several spellings accepted for convenience:
                method_config["num_virtual_tokens"] = val  # record it ...
                model_kwargs["num_virtual_tokens"] = val  # ... and pass it to build_prompt_tuning.
            elif args.ablation_name in ("init", "prompt_tuning_init"):  # 2) initialisation (RANDOM vs TEXT):
                method_config["prompt_tuning_init"] = str(val)  # record it as text ...
                model_kwargs["prompt_tuning_init"] = str(val)  # ... and pass it to build_prompt_tuning.
        if not model_kwargs:  # If no branch above matched, the ablation name is wrong for this method (e.g. lora_rank with adapter) ...
            raise SystemExit(f"unknown ablation {args.ablation_name!r} for {args.method}")  # ... stop with a clear message. Otherwise the DEFAULT model would be trained and saved under an ablation name, giving a silently wrong result.

    result = new_result(method=args.method, task=args.task, regime=args.regime,  # Create the empty result dictionary (format defined in results.py) with this run's identity ...
                        seed=args.seed, stage=args.stage,  # ... seed and stage ...
                        owner=args.owner or OWNERS[args.method],  # ... owner: the one given with --owner, or else the method's default owner ...
                        hyperparams=hyperparams,  # ... the training settings ...
                        method_config=method_config,  # ... the method settings (including any ablation change) ...
                        ablation=ablation,  # ... the ablation description (or None) ...
                        model_name=MODEL_NAME)  # ... and the base model name.
    result["meta"]["notes"] = args.notes  # Store the free-text note from --notes.

    out_path = result_path(result)  # Where this run's JSON will be saved, e.g. results/final/prompt_tuning_rte_1k_seed1.json.
    if out_path.exists() and not args.overwrite:  # If that file already exists (run done before) and --overwrite was not given ...
        print(f"skip: {out_path} already exists")  # ... say so in the log ...
        return  # ... and stop here, BEFORE any expensive work. This lets run_method.sh be restarted after a crash and continue where it stopped.

    print(f"=== {result['run_id']}  lr={hyperparams['learning_rate']} "  # Print a header line for this run in the log, e.g. "=== lora_rte_1k_seed1 lr=0.0005 epochs=20 fp16=True" ...
          f"epochs={hyperparams['epochs']} fp16={fp16}")  # ... so you can find each run when reading a long log.

    set_seed(args.seed)  # Seed Python's, NumPy's and PyTorch's random generators. This fixes the random head initialisation, dropout and data shuffling, so running the same command again gives the same result. Without it results could not be reproduced.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # Load RoBERTa's tokenizer (the exact vocabulary the model was pretrained with). Text must be split the same way as in pretraining, or the model sees garbage.
    train, dev, validation = load_task(args.task, tokenizer, regime=args.regime,  # Load and tokenize the three splits (see data.py): train for learning, dev for choosing the best epoch, validation for the reported score ...
                                       seed=args.seed, max_len=MAX_LENGTH)  # ... with the same seed, so every method gets exactly the same split and the same 1,000 examples.
    result["data"].update(n_train=len(train), n_dev=len(dev),  # Record the sizes of the splits in the result (e.g. n_train=1000 confirms a 1k run) ...
                          n_eval=len(validation))  # ... including the size of GLUE validation.

    model = build_model(args.method, NUM_LABELS[args.task], **model_kwargs)  # Build the model for this method (see models.py) with 2 output classes and any ablation change. This is where weights get frozen or not.
    result["efficiency"].update(count_params(model))  # Count trainable and total parameters and store them (e.g. 607,490 trainable for prompt tuning).
    print(f"trainable params: {result['efficiency']['trainable_params']:,} "  # Print the trainable count with thousands separators ...
          f"({result['efficiency']['trainable_pct']}%)")  # ... and as a percentage. This is a quick sanity check that the method is set up right (e.g. LoRA should print 887,042).

    work_dir = tempfile.mkdtemp(prefix=f"{result['run_id']}_")  # Create a new, empty temporary folder for this run's checkpoints, e.g. /tmp/lora_rte_1k_seed1_abc123. It is deleted at the end.
    try:  # Start a try/finally block: whatever happens inside (even a crash), the "finally" part below still deletes the temporary folder.
        training_args = TrainingArguments(  # All settings for the Hugging Face training loop:
            output_dir=os.path.join(work_dir, "trainer"),  # Where the Trainer saves checkpoints during training (inside the temporary folder).
            num_train_epochs=hyperparams["epochs"],  # How many epochs to train.
            learning_rate=hyperparams["learning_rate"],  # Peak learning rate (after warmup).
            per_device_train_batch_size=BATCH_SIZE,  # 32 examples per training step.
            per_device_eval_batch_size=EVAL_BATCH_SIZE,  # 64 examples per evaluation step.
            warmup_ratio=WARMUP_RATIO,  # 6% linear warmup, then linear decay to 0.
            weight_decay=WEIGHT_DECAY,  # AdamW weight decay 0.01 (AdamW is the Trainer's default optimizer).
            fp16=fp16,  # Mixed precision on GPU: faster and less memory.
            eval_strategy="epoch",  # Evaluate on the dev set at the end of EVERY epoch. Needed so the best epoch can be chosen.
            save_strategy="epoch",  # Save a checkpoint at the end of every epoch. Must match eval_strategy, otherwise load_best_model_at_end raises an error.
            logging_strategy="epoch",  # Write the training loss to the log once per epoch (keeps the log short).
            load_best_model_at_end=True,  # After the last epoch, reload the checkpoint with the best dev accuracy. Without this we would report the LAST epoch, which may have overfitted, especially on small tasks like RTE.
            metric_for_best_model="accuracy",  # "Best" means highest dev accuracy (eval_accuracy from metrics.py).
            greater_is_better=True,  # Higher accuracy is better. (For a loss it would be False.)
            save_total_limit=1,  # Keep at most 1 checkpoint plus the best one on disk; older ones are deleted. Without this, full FT would keep up to 20 checkpoints of ~1.4 GB each (weights + optimizer state) and fill Kaggle's ~20 GB disk.
            report_to="none",  # Don't send logs to Weights & Biases / TensorBoard. Otherwise the Trainer may ask for a login and block the run.
            seed=args.seed,  # Seed for the Trainer's own randomness (data shuffling order between epochs).
        )  # End of TrainingArguments.
        if args.method == "adapter":  # Adapters need AdapterHub's special trainer ...
            from adapters import AdapterTrainer  # ... which knows how to save and reload adapter checkpoints correctly. Imported here so other methods don't need the adapters library.
            trainer_cls = AdapterTrainer  # Use it. With the normal Trainer, reloading the best adapter checkpoint can fail.
        else:  # Full FT, LoRA, prompt tuning:
            trainer_cls = Trainer  # the normal Hugging Face Trainer works fine.

        trainer = trainer_cls(  # Create the training loop object with everything it needs:
            model=model,  # the model to train,
            args=training_args,  # the settings above,
            train_dataset=train,  # the training data,
            eval_dataset=dev,  # our 10% DEV split, used to pick the best epoch. The GLUE validation split is deliberately NOT given here, so it is never used for any choice.
            data_collator=DataCollatorWithPadding(tokenizer),  # pads every batch to the length of its longest example and builds the attention mask. Without it, examples of different lengths could not be stacked into one tensor.
            compute_metrics=compute_metrics(args.task),  # the metric function for this task. Without it there is no eval_accuracy and best-model selection fails.
        )  # End of trainer.

        if torch.cuda.is_available():  # Only if a GPU exists:
            torch.cuda.empty_cache()  # release cached, unused GPU memory from loading the model, so it doesn't affect the measurement ...
            torch.cuda.reset_peak_memory_stats()  # ... and reset the "peak memory" counter to the current level, so the peak we read later covers TRAINING ONLY.
        t0 = time.time()  # Record the start time (seconds since 1970).
        trainer.train()  # TRAIN: for every epoch, loop over batches (forward pass, loss, backward pass, optimizer step), evaluate on dev, save a checkpoint; at the end reload the best checkpoint. This is where almost all the time is spent.
        train_time = time.time() - t0  # Training time in seconds = end time minus start time.
        peak_mem = peak_gpu_mem_mb()  # Read the highest GPU memory used during training, in MiB (None on CPU).

        dev_m = trainer.evaluate(dev)  # Evaluate the BEST checkpoint on our dev split. Returns e.g. {"eval_accuracy": 0.78, ...}.
        val_m = trainer.evaluate(validation)  # Evaluate the best checkpoint on the official GLUE validation split. THIS is the number reported in the project's tables.

        result["metrics"].update(  # Store the scores in the result:
            accuracy=val_m["eval_accuracy"],  # validation accuracy (the main reported number),
            f1=val_m.get("eval_f1"),  # validation F1 for MRPC; .get() returns None for SST-2 / RTE instead of crashing,
            dev_accuracy=dev_m["eval_accuracy"],  # dev accuracy (used for the LR search),
            dev_f1=dev_m.get("eval_f1"),  # dev F1 (MRPC only),
            best_epoch=best_epoch(trainer),  # which epoch was best.
        )  # End of metrics update.

        ckpt_dir = os.path.join(work_dir, "weights")  # A sub-folder of the temporary folder where we save the deployable weights.
        save_trainable_weights(trainer.model, args.method, ckpt_dir)  # Save only what you'd ship for this task. We only save it to MEASURE its size; it is deleted afterwards.
        result["efficiency"].update(  # Store the cost numbers:
            peak_gpu_mem_mb=peak_mem,  # peak GPU memory in MiB,
            train_time_s=round(train_time, 1),  # training time in seconds, rounded to 1 decimal,
            checkpoint_size_mb=dir_size_mb(ckpt_dir),  # size of the saved weights in MiB.
        )  # End of efficiency update.
    finally:  # This ALWAYS runs, whether training succeeded or crashed:
        shutil.rmtree(work_dir, ignore_errors=True)  # delete the whole temporary folder (all checkpoints and saved weights). Kaggle gives only about 20 GB of disk, and 16 runs in a row would fill it. ignore_errors=True so a deletion problem never hides the real error.

    path = save_result(result, overwrite=args.overwrite)  # Fill in timestamp, git commit and environment, validate everything, and write the JSON safely (see results.py). Returns the file path.
    m = result["metrics"]  # Short name for the metrics, used in the summary print below.
    f1 = f"  f1={m['f1']:.4f}" if m["f1"] is not None else ""  # Build the F1 part of the summary only for MRPC; formatting None with :.4f would crash for other tasks.
    print(f"saved {path}\n  val acc={m['accuracy']:.4f}{f1}  "  # Print a summary of the finished run: where it was saved and the validation accuracy (and F1) ...
          f"dev acc={m['dev_accuracy']:.4f}  best epoch={m['best_epoch']}  "  # ... the dev accuracy and best epoch ...
          f"time={result['efficiency']['train_time_s']}s  "  # ... the training time ...
          f"peak mem={result['efficiency']['peak_gpu_mem_mb']} MiB")  # ... and the peak memory. Handy for checking results directly in the Kaggle log.


def main(argv=None) -> None:  # FUNCTION main: the entry point of the script. Kept separate from run() so other code (or tests) can call main([...]) with their own arguments.
    run(parse_args(argv))  # Read the command-line options, then do the run.


if __name__ == "__main__":  # True only when this file is executed as a program (python -m src.train ...), not when another file imports it.
    main()  # Start the script.
