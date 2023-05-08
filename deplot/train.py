import logging
import os
import sys
from functools import partial

import numpy as np
import pandas as pd
import torch
import transformers
from data_args import DataArguments
from dataset import ImageCaptioningDataset, collate_fn, create_processor, read_annotation
from engine import CustomTrainer, compute_metrics
from model_args import ModelArguments
from transformers import (
    HfArgumentParser,
    Pix2StructForConditionalGeneration,
    TrainingArguments,
    set_seed,
)
from transformers.trainer import logger
from transformers.trainer_utils import get_last_checkpoint, is_main_process

torch.set_float32_matmul_precision("high")


def main():
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()
    data_args: DataArguments
    model_args: ModelArguments
    training_args: TrainingArguments

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this"
                " behavior, change the `--output_dir` or add `--overwrite_output_dir` to train"
                " from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        # transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    # Set seed before initializing model.
    set_seed(training_args.seed)
    train_df = read_annotation("../data/train_list.txt")
    val_df = read_annotation("../data/val_list.txt")
    val_df = val_df.groupby(["chart_type"]).apply(lambda x: x.head(10)).reset_index(drop=True)
    processor = create_processor(model_args.model_name)
    train_dataset = ImageCaptioningDataset(train_df, processor, data_args.max_patches)
    val_dataset = ImageCaptioningDataset(val_df, processor, data_args.max_patches)

    # Initialize trainer
    logger.info("Initializing model...")
    model: torch.nn.Module = Pix2StructForConditionalGeneration.from_pretrained(
        model_args.model_name
    )
    model.config.text_config.is_decoder = True

    if last_checkpoint is None and model_args.resume is not None:
        logger.info(f"Loading {model_args.resume} ...")
        checkpoint = torch.load(model_args.resume, "cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint.pop("state_dict")
        model.load_state_dict(checkpoint)

    logger.info("Start training...")

    training_args.remove_unused_columns = False
    training_args.load_best_model_at_end = True
    training_args.evaluation_strategy = "epoch"
    training_args.save_strategy = "epoch"
    training_args.save_total_limit = 3
    training_args.logging_strategy = "steps"
    training_args.lr_scheduler_type = "cosine"
    # training_args.optim = "adafactor"
    # training_args.adam_epsilon = 1e-6
    training_args.greater_is_better = True
    training_args.warmup_ratio = 0.1
    training_args.metric_for_best_model = "eval_overall"

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=partial(collate_fn, processor=processor),
        compute_metrics=partial(compute_metrics, val_df=val_df, processor=processor),
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
