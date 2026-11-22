"""Train and evaluate one run, then save its result JSON.

Run from the repo root (otherwise `from src...` imports fail):

    python -m src.train --method full_ft --task rte --regime 1k --seed 1

Writes results/final/{method}_{task}_{regime}_seed{seed}.json. If that file
already exists the run is skipped (use --overwrite to redo it), so a restarted
run_method.sh picks up where it stopped.

One run per process: peak GPU memory is only comparable that way.
"""  # file description, also shown by --help (description=__doc__ below)

from __future__ import annotations  # allows modern type hints like "float | None" on older Python

import os  # environment variables and file paths

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # use only GPU 0; on Kaggle "T4 x2" the Trainer would otherwise split batches over 2 GPUs and memory numbers break (must run before torch is imported)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # turn off tokenizer threads; without it you get a long "fork" warning in the log

import argparse  # reads command-line options like --method lora
import shutil  # used to delete the temporary checkpoint folder
import tempfile  # creates a temporary folder for checkpoints
import time  # measures training time

import torch  # PyTorch: GPU checks and memory statistics
from transformers import (AutoTokenizer, DataCollatorWithPadding, Trainer,  # tokenizer, batch padder, training loop
                          TrainingArguments, set_seed)  # training settings, and one function to seed every random generator

from src.data import NUM_LABELS, load_task  # our data loader (needs running from the repo root, else "No module named src")
from src.metrics import compute_metrics  # our accuracy / F1 function
from src.models import MODEL_NAME, METHOD_CONFIGS, build_model  # our 4 model builders
from src.results import (METHODS, REGIMES, SEEDS, STAGES, TASKS, count_params,  # allowed values + param counter
                         dir_size_mb, new_result, peak_gpu_mem_mb, result_path,  # measurement helpers and result builders
                         save_result)  # validates and writes the result JSON

LEARNING_RATES = {"full_ft": 2e-5, "lora": 5e-4, "adapter": 3e-4,  # LR per method, chosen from the papers; full FT needs a small LR or it destroys pretrained weights
                  "prompt_tuning": 1e-2}  # prompt tuning needs a very big LR because it trains so few params; with 2e-5 it would barely learn
BATCH_SIZE = 32  # examples per training step, same for every method (fair comparison); bigger = more GPU memory
EVAL_BATCH_SIZE = 64  # evaluation can use bigger batches since no gradients are stored -> faster eval
MAX_LENGTH = 128  # max tokens per example; longer inputs are cut, shorter ones padded
WARMUP_RATIO = 0.06  # first 6% of steps the LR grows from 0; without warmup big early updates can destabilise training
WEIGHT_DECAY = 0.01  # small pull of weights toward 0 (regularisation) to reduce overfitting

OWNERS = {"full_ft": "deepak", "lora": "anoop", "adapter": "chiranjeet",  # team member responsible for each method
          "prompt_tuning": "praveen"}  # saved in meta.owner; results.py rejects a result with no owner


def num_epochs(task: str, regime: str) -> int:  # how many passes over the training data
    if regime == "1k":  # small data: only ~32 steps per epoch
        return 20  # so train longer; fewer epochs would under-train
    return 3 if task == "sst2" else 10  # SST-2 is big (60k examples) so 3 is enough; MRPC/RTE are small so 10


def save_trainable_weights(model, method: str, out_dir: str) -> None:  # saves only what you'd ship for this task; the folder size becomes "checkpoint size"
    """Save what you'd ship for this task (see docs/results_schema.md)."""  # help text
    if method == "adapter":  # AdapterHub model has its own save function
        model.save_adapter(out_dir, "glue")  # saves only adapter "glue" + head (~9 MB); save_pretrained would save all 475 MB
    else:  # full FT, LoRA, prompt tuning
        model.save_pretrained(out_dir)  # full FT saves the whole model; PEFT models automatically save only LoRA / prompt weights + head


def best_epoch(trainer) -> float | None:  # finds the epoch whose checkpoint had the best dev accuracy
    """Epoch of the checkpoint picked by load_best_model_at_end."""  # help text
    ckpt = trainer.state.best_model_checkpoint  # path like ".../checkpoint-1234"
    if not ckpt:  # no best checkpoint recorded (e.g. no evaluation happened)
        return None  # return None instead of crashing on the split below
    step = int(ckpt.rstrip("/").split("-")[-1])  # take the number after the last "-" -> training step 1234
    for entry in trainer.state.log_history:  # go through every logged event
        if entry.get("step") == step and "eval_accuracy" in entry:  # the evaluation done at that step
            return entry["epoch"]  # its epoch number
    return None  # not found -> None (saved as null in the JSON)


def parse_args(argv=None):  # defines and reads the command-line options
    p = argparse.ArgumentParser(description=__doc__,  # --help shows the file description
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # keep the description's line breaks
    p.add_argument("--method", required=True, choices=METHODS)  # which of the 4 methods; choices -> a typo gives a clear error
    p.add_argument("--task", required=True, choices=TASKS)  # sst2 / mrpc / rte
    p.add_argument("--regime", required=True, choices=REGIMES)  # full data or 1k examples
    p.add_argument("--seed", required=True, type=int, choices=SEEDS)  # 1, 2 or 3; different seeds show how stable results are
    p.add_argument("--stage", choices=STAGES, default="final",  # final / lr_search / ablation; decides the results sub-folder
                   help="stage: final (default), lr_search, or ablation")  # text for --help
    p.add_argument("--lr", type=float, default=None,  # learning rate to try, only used in the LR search
                   help="learning rate for --stage lr_search (default: the method's final LR)")  # text for --help
    p.add_argument("--ablation-name", default=None,  # which setting to change in an ablation, e.g. lora_rank
                   help="name of ablation parameter (e.g. reduction_factor, config)")  # text for --help
    p.add_argument("--ablation-value", default=None,  # the new value for that setting, e.g. 4
                   help="value of ablation parameter (e.g. 8, seq_bn)")  # text for --help
    p.add_argument("--owner", default=None,  # optional: override who ran it
                   help="defaults to the method's owner (meta.owner)")  # text for --help
    p.add_argument("--notes", default="", help="free text saved in meta.notes")  # optional comment stored in the JSON
    p.add_argument("--overwrite", action="store_true",  # flag: redo a run that already has a result
                   help="redo the run even if its result JSON exists")  # without it existing runs are skipped
    return p.parse_args(argv)  # read the options (argv=None means the real command line)


def run(args) -> None:  # the whole experiment for one (method, task, regime, seed)
    fp16 = torch.cuda.is_available()  # use 16-bit math only on a GPU (faster, less memory); fp16 on CPU would crash
    hyperparams = {  # training settings, also saved in the result JSON
        "learning_rate": (args.lr if args.stage == "lr_search" and args.lr  # LR search: use the LR given on the command line
                          else LEARNING_RATES[args.method]),  # otherwise the method's fixed LR
        "epochs": num_epochs(args.task, args.regime),  # 3, 10 or 20 epochs
        "batch_size": BATCH_SIZE,  # 32
        "max_length": MAX_LENGTH,  # 128 tokens
        "warmup_ratio": WARMUP_RATIO,  # 6% warmup
        "weight_decay": WEIGHT_DECAY,  # 0.01
        "fp16": fp16,  # True on GPU
    }  # end of hyperparams

    ablation = None  # stays None for normal (non-ablation) runs
    method_config = dict(METHOD_CONFIGS[args.method])  # COPY of the method's default settings; without copying we'd change the shared default
    model_kwargs = {}  # extra arguments passed to build_model (empty = default model)
    if args.stage == "ablation" and args.ablation_name:  # only for ablation runs
        val = args.ablation_value  # value from the command line, always a string like "8"
        try:  # try to turn it into a number
            val = int(val)  # "8" -> 8; without this the builders would get the string "8" and fail
        except ValueError:  # not an integer
            try:  # maybe a decimal
                val = float(val)  # "0.5" -> 0.5
            except ValueError:  # not a number at all
                pass  # keep it as text, e.g. "seq_bn" or "TEXT"
        ablation = {"name": args.ablation_name, "value": val}  # recorded in the JSON and used in the file name
        if args.method == "full_ft" and args.ablation_name == "frozen_layers":  # full FT ablation: freeze bottom layers
            method_config["frozen_layers"] = val  # record it
            model_kwargs["frozen_layers"] = val  # pass it to build_full_ft
        elif args.method == "lora" and args.ablation_name == "lora_rank":  # LoRA ablation: change rank
            method_config.update(r=val, lora_alpha=2 * val)  # record rank and alpha (alpha = 2r, same as in build_lora)
            model_kwargs["r"] = val  # pass the rank to build_lora
        elif args.method == "adapter":  # adapter ablations
            if args.ablation_name == "reduction_factor":  # change bottleneck size
                method_config["reduction_factor"] = val  # record it
                model_kwargs["reduction_factor"] = val  # pass it to build_adapter
            elif args.ablation_name in ("config", "adapter_config"):  # change adapter type (Houlsby / Pfeiffer)
                method_config["config"] = val  # record it
                model_kwargs["config"] = val  # pass it to build_adapter
        elif args.method == "prompt_tuning":  # prompt-tuning ablations
            if args.ablation_name in ("num_virtual_tokens", "n_tokens", "virtual_tokens"):  # change number of soft tokens (several names accepted)
                method_config["num_virtual_tokens"] = val  # record it
                model_kwargs["num_virtual_tokens"] = val  # pass it to build_prompt_tuning
            elif args.ablation_name in ("init", "prompt_tuning_init"):  # change init (RANDOM / TEXT)
                method_config["prompt_tuning_init"] = str(val)  # record it as text
                model_kwargs["prompt_tuning_init"] = str(val)  # pass it to build_prompt_tuning
        if not model_kwargs:  # nothing matched -> wrong ablation name for this method
            raise SystemExit(f"unknown ablation {args.ablation_name!r} for {args.method}")  # stop; otherwise it would silently train the default model under an ablation name

    result = new_result(method=args.method, task=args.task, regime=args.regime,  # create the empty result record with the run identity
                        seed=args.seed, stage=args.stage,  # seed and stage
                        owner=args.owner or OWNERS[args.method],  # given owner, or the method's default owner
                        hyperparams=hyperparams,  # the settings above
                        method_config=method_config,  # method settings (with ablation change)
                        ablation=ablation,  # None or {"name", "value"}
                        model_name=MODEL_NAME)  # "roberta-base"
    result["meta"]["notes"] = args.notes  # store the free-text notes

    out_path = result_path(result)  # where the JSON will be saved
    if out_path.exists() and not args.overwrite:  # run already done?
        print(f"skip: {out_path} already exists")  # tell the user
        return  # stop here; lets a crashed batch restart without redoing finished runs

    print(f"=== {result['run_id']}  lr={hyperparams['learning_rate']} "  # print a header line for this run in the log
          f"epochs={hyperparams['epochs']} fp16={fp16}")  # with epochs and fp16

    set_seed(args.seed)  # seed python, numpy and torch; without it head init / shuffling change every time and results can't be reproduced
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # RoBERTa's tokenizer (text -> token ids)
    train, dev, validation = load_task(args.task, tokenizer, regime=args.regime,  # load and tokenize the 3 splits
                                       seed=args.seed, max_len=MAX_LENGTH)  # same seed -> same split for every method
    result["data"].update(n_train=len(train), n_dev=len(dev),  # record dataset sizes
                          n_eval=len(validation))  # size of GLUE validation

    model = build_model(args.method, NUM_LABELS[args.task], **model_kwargs)  # build the model for this method (+ ablation change)
    result["efficiency"].update(count_params(model))  # record trainable / total params
    print(f"trainable params: {result['efficiency']['trainable_params']:,} "  # print them so you can check the method is set up right
          f"({result['efficiency']['trainable_pct']}%)")  # as a percentage

    work_dir = tempfile.mkdtemp(prefix=f"{result['run_id']}_")  # temporary folder for checkpoints, deleted at the end
    try:  # try/finally: the folder is deleted even if training crashes
        training_args = TrainingArguments(  # all settings for the HF training loop
            output_dir=os.path.join(work_dir, "trainer"),  # where checkpoints go
            num_train_epochs=hyperparams["epochs"],  # number of epochs
            learning_rate=hyperparams["learning_rate"],  # learning rate
            per_device_train_batch_size=BATCH_SIZE,  # 32 per step
            per_device_eval_batch_size=EVAL_BATCH_SIZE,  # 64 per eval step
            warmup_ratio=WARMUP_RATIO,  # LR warmup
            weight_decay=WEIGHT_DECAY,  # weight decay
            fp16=fp16,  # mixed precision on GPU
            eval_strategy="epoch",  # evaluate on dev after every epoch; needed for picking the best epoch
            save_strategy="epoch",  # save a checkpoint after every epoch; must match eval_strategy or load_best_model_at_end errors
            logging_strategy="epoch",  # log the loss once per epoch
            load_best_model_at_end=True,  # at the end reload the best-dev checkpoint; without it we'd report the last epoch, which may have overfit
            metric_for_best_model="accuracy",  # "best" = highest dev accuracy
            greater_is_better=True,  # higher accuracy is better (for loss it would be False)
            save_total_limit=1,  # keep few checkpoints; otherwise 10-20 copies of 475 MB fill Kaggle's disk
            report_to="none",  # don't send logs to wandb / tensorboard (would ask for login)
            seed=args.seed,  # seed for the Trainer's data shuffling
        )  # end of TrainingArguments
        if args.method == "adapter":  # adapters need a special Trainer
            from adapters import AdapterTrainer  # knows how to save/load adapter checkpoints
            trainer_cls = AdapterTrainer  # use it; the normal Trainer can fail to reload the best adapter checkpoint
        else:  # other methods
            trainer_cls = Trainer  # normal HF Trainer

        trainer = trainer_cls(  # create the training loop
            model=model,  # the model to train
            args=training_args,  # the settings above
            train_dataset=train,  # training data
            eval_dataset=dev,  # our 10% dev split, used to pick the best epoch (never the reported validation set)
            data_collator=DataCollatorWithPadding(tokenizer),  # pads each batch to its longest example; without it batches of different lengths can't be stacked
            compute_metrics=compute_metrics(args.task),  # accuracy (+F1); without it there's no eval_accuracy and best-model selection fails
        )  # end of Trainer

        if torch.cuda.is_available():  # only with a GPU
            torch.cuda.empty_cache()  # free unused cached GPU memory
            torch.cuda.reset_peak_memory_stats()  # reset the peak counter so the peak covers only training (not model loading)
        t0 = time.time()  # start time
        trainer.train()  # TRAIN the model (all epochs, eval each epoch, reload best at end)
        train_time = time.time() - t0  # training time in seconds
        peak_mem = peak_gpu_mem_mb()  # highest GPU memory used during training

        dev_m = trainer.evaluate(dev)  # score of the best checkpoint on our dev split
        val_m = trainer.evaluate(validation)  # score on GLUE validation -> this is the number we report

        result["metrics"].update(  # store the scores
            accuracy=val_m["eval_accuracy"],  # validation accuracy (main result)
            f1=val_m.get("eval_f1"),  # validation F1 (MRPC only; .get gives None for other tasks instead of crashing)
            dev_accuracy=dev_m["eval_accuracy"],  # dev accuracy
            dev_f1=dev_m.get("eval_f1"),  # dev F1 (MRPC only)
            best_epoch=best_epoch(trainer),  # which epoch was best
        )  # end of metrics

        ckpt_dir = os.path.join(work_dir, "weights")  # folder to save the shippable weights
        save_trainable_weights(trainer.model, args.method, ckpt_dir)  # save them (only to measure their size)
        result["efficiency"].update(  # store efficiency numbers
            peak_gpu_mem_mb=peak_mem,  # peak memory
            train_time_s=round(train_time, 1),  # training time, 1 decimal
            checkpoint_size_mb=dir_size_mb(ckpt_dir),  # size of saved weights
        )  # end of efficiency
    finally:  # always runs, even after an error
        shutil.rmtree(work_dir, ignore_errors=True)  # delete all checkpoints; Kaggle has only ~20 GB of disk

    path = save_result(result, overwrite=args.overwrite)  # validate and write the result JSON
    m = result["metrics"]  # short name for printing
    f1 = f"  f1={m['f1']:.4f}" if m["f1"] is not None else ""  # F1 text only for MRPC; formatting None would crash
    print(f"saved {path}\n  val acc={m['accuracy']:.4f}{f1}  "  # print a summary of the run
          f"dev acc={m['dev_accuracy']:.4f}  best epoch={m['best_epoch']}  "  # dev score and best epoch
          f"time={result['efficiency']['train_time_s']}s  "  # time
          f"peak mem={result['efficiency']['peak_gpu_mem_mb']} MiB")  # memory


def main(argv=None) -> None:  # entry point
    run(parse_args(argv))  # read the options, then do the run


if __name__ == "__main__":  # true only when run as a program (python -m src.train), not when imported
    main()  # start
