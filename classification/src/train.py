import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import transformers
from joblib import Parallel, delayed
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from data_args import DataArguments
from dataset import BMGADataset, collate_fn
from engine import CustomTrainer, compute_metrics
from model import Model
from model_args import ModelArguments


torch.set_float32_matmul_precision("high")
logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()

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
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        # transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    # Set seed before initializing model.
    set_seed(training_args.seed)
  
    # Load dataset
    print("Loading dataset...")
    if data_args.classification_type == "graph":
        classes = ['dot', 'line', 'scatter', 'vertical_bar', "horizontal_bar"]
    else:
        classes = ["numerical", "categorical"]
    train_dataset = BMGADataset(
        jsonl_path=data_args.train_jsonl_path,
        image_dir=data_args.train_image_folder,
        classes=classes,
        classification_type=data_args.classification_type
    )

    val_dataset = BMGADataset(
        jsonl_path=data_args.val_jsonl_path,
        image_dir=data_args.val_image_folder,
        classes=classes,
        classification_type=data_args.classification_type
    )

    # Initialize trainer
    print("Initializing model...")
    model = Model(
        model_name=model_args.model_name,
        n_classes=len(classes),
    )

    if last_checkpoint is None and model_args.resume is not None:
        logger.info(f"Loading {model_args.resume} ...")
        checkpoint = torch.load(model_args.resume, "cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint.pop("state_dict")
        checkpoint = {k[6:]: v for k, v in checkpoint.items()}
        model.model.load_state_dict(checkpoint)

        if "fc.weight" in checkpoint:
            model.fc.load_state_dict(
                {"weight": checkpoint["fc.weight"], "bias": checkpoint["fc.bias"]}
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print("Start training...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
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
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()