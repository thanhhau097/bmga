import logging
import os
import sys

import torch
import transformers
from sconf import Config
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.trainer import logger
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from .args import Arguments
from .engine import CustomTrainer
from .model import DonutConfig, DonutModel
from .multimodal import MultimodalConfig, MultimodalModel
from .util import DataCollator, DonutDataset

torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")


def main():
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()

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
    # logger.info(f"Training/evaluation parameters {training_args}")
    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = Config(args.config)
    model_cls = DonutModel
    model_config = DonutConfig(
        input_size=config.input_size,
        max_length=config.max_length,
        align_long_axis=config.align_long_axis,
        backbone_name=config.backbone_name,
    )
    if config.get("meta_arch", False):
        if config.meta_arch == "multimodal":
            model_cls = MultimodalModel
            model_config = MultimodalConfig(
                input_size=config.input_size,
                max_length=config.max_length,
                align_long_axis=config.align_long_axis,
                backbone_name=config.backbone_name,
                fusion=config.fusion,
                text_encoder_name=config.text_encoder_name,
            )

    # Initialize trainer
    if config.get("pretrained_model_name_or_path", False):
        if config.get("meta_arch") == "multimodal":
            model = model_cls.from_pretrained(
                config.pretrained_model_name_or_path,
                input_size=config.input_size,
                max_length=config.max_length,
                align_long_axis=config.align_long_axis,
                ignore_mismatched_sizes=True,
                fusion=config.fusion,
                text_encoder_name=config.text_encoder_name,
            )
        else:
            model = model_cls.from_pretrained(
                config.pretrained_model_name_or_path,
                input_size=config.input_size,
                max_length=config.max_length,
                align_long_axis=config.align_long_axis,
                ignore_mismatched_sizes=True,
            )
    else:
        model = model_cls(config=model_config)

    if (
        config.get("meta_arch") == "multimodal"
        and last_checkpoint is None
        and config.resume is not None
    ):
        logger.info(f"Loading {config.resume} ...")
        checkpoint = torch.load(config.resume, "cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint.pop("state_dict")
        checkpoint = {
            k.replace("_orig_mod.", ""): v for k, v in checkpoint.items() if "aug." not in k
        }
        model.decoder.model.resize_token_embeddings(
            checkpoint["decoder.model.model.decoder.embed_tokens.weight"].shape[0]
        )
        model.load_state_dict(checkpoint, strict=False)

    train_dataset = DonutDataset(
        config.dataset_name_or_paths[0],
        donut_model=model,
        max_length=config.max_length,
        split="train",
        task_start_token="<s_docund>",
        prompt_end_token="<s_docund>",
        sort_json_key=config.sort_json_key,
    )
    valid_dataset = DonutDataset(
        config.dataset_name_or_paths[0],
        donut_model=model,
        max_length=config.max_length,
        split="validation",
        task_start_token="<s_docund>",
        prompt_end_token="<s_docund>",
        sort_json_key=config.sort_json_key,
    )

    model = model.to(device="cuda")

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollator(train_dataset.use_text_inputs),
        compute_metrics=None,
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
        model.decoder.tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(valid_dataset, metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
