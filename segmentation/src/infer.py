import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import transformers
from joblib import Parallel, delayed
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from tqdm import tqdm

from data_args import DataArguments
from dataset import SegmentationInferenceDataset, SegmentationDataset, collate_fn
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

    # Load data
    print(f"Reading data at {data_args.train_csv}")
    val_df = pd.read_csv(data_args.train_csv)

    val_dataset = SegmentationDataset(
        df=val_df,
        data_dir=data_args.data_dir,
        size=data_args.size,
        mode='val'
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=training_args.dataloader_num_workers, 
        pin_memory=True
    )

    # Initialize trainer
    print("Initializing model...")
    model = Model(
        arch=model_args.arch,
        encoder_name=model_args.encoder_name,
        drop_path=model_args.drop_path,
        size=data_args.size
    )
    if last_checkpoint is None and model_args.resume is not None:
        logger.info(f"Loading {model_args.resume} ...")
        checkpoint = torch.load(model_args.resume, "cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint.pop("state_dict")
        # checkpoint = {k[6:]: v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)

        if "fc.weight" in checkpoint:
            model.fc.load_state_dict(
                {"weight": checkpoint["fc.weight"], "bias": checkpoint["fc.bias"]}
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print("Start inference...")
    model.eval()

    masks = []
    imgs = []
    gts = []
    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader))
        for step, batch in enumerate(tk):
            img, gt = [x.to(device) for x in batch]
            mask = model(img)
            masks.extend(mask)
            imgs.extend(img)
            gts.extend(gt)
    masks = torch.stack(masks)
    imgs = torch.stack(imgs)
    gts = torch.stack(gts)
    print(f"Saving outputs")
    torch.save(masks, "masks_out_stage1.pth")
    torch.save(imgs, "imgs_out_stage1.pth")
    torch.save(gts, "gts_out_stage1.pth")
    print("Saved!")


if __name__ == "__main__":
    main()
