import ast
from collections import defaultdict
import gc
import os

import cv2
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, default_collate, DataLoader
from tqdm import tqdm
from transformers import TrainerCallback
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import json

class SegmentationDataset(Dataset):
    def __init__(self, df, data_dir, size, mode="train"):
        super().__init__()

        self.data_dir = data_dir
        self.mode = mode
        self.df = df[df['mode'] == mode].reset_index(drop=True)
        self.image_paths, self.labels = self.preprocess_df()
        self.images = {}
        self.masks = {}

        if mode == "train":
            self.transform = A.Compose(
                [
                    # A.LongestMaxSize(max_size=int(size * 1.1), interpolation=1),
                    # A.PadIfNeeded(
                    #     min_height=int(size * 1.1),
                    #     min_width=int(size * 1.1),
                    #     border_mode=0,
                    #     value=(0, 0, 0),
                    # ),
                    A.Resize(height=int(size * 1.1), width=int(size * 1.1), interpolation=1),
                    A.RandomBrightnessContrast(p=0.1),
                    A.HorizontalFlip(p=0.1),
                    A.VerticalFlip(p=0.1),
                    A.OneOf(
                        [A.ShiftScaleRotate(), A.GridDistortion(), A.ElasticTransform()], p=0.1
                    ),
                    A.OneOf([A.GaussNoise(), A.MultiplicativeNoise()], p=0.1),
                    A.OneOf([A.Blur(blur_limit=3), A.MedianBlur(), A.MotionBlur()], p=0.1),
                    A.RandomCrop(width=int(size), height=int(size)),
                    ToTensorV2(),
                ],
            )
        else:
            self.transform = A.Compose(
                [
                    # A.LongestMaxSize(max_size=size, interpolation=1),
                    # A.PadIfNeeded(
                    #     min_height=size,
                    #     min_width=size,
                    #     border_mode=0,
                    #     value=(0, 0, 0),
                    # ),
                    A.Resize(height=size, width=size, interpolation=1),
                    ToTensorV2(),
                ],
            )

    def __len__(self):
        return len(self.df)

    def preprocess_df(self):
        return self.df["image"].to_list(), self.df["mask"].to_list()

    def __getitem__(self, idx):
        if self.image_paths[idx] in self.images:
            img = self.images[self.image_paths[idx]]
        else:
            img = cv2.imread(self.image_paths[idx])
            self.images[self.image_paths[idx]] = img

        if self.labels[idx] in self.masks:
            label = self.masks[self.labels[idx]]
        else:
            label = cv2.imread(self.labels[idx], cv2.IMREAD_GRAYSCALE)
            if "syn" in self.image_paths[idx]:
                label = 255 - label
            self.masks[self.labels[idx]] = label
        transformed = self.transform(
            image=img, mask=label
        )
        return (
            transformed["image"] / 255.0,
            transformed["mask"] / 255.0,
        )



def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return {"images": images, "labels": labels}
