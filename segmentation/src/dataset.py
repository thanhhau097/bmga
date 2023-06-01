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
        self.df = df.reset_index(drop=True)
        self.image_paths, self.labels = self.preprocess_df()
        self.images = {}
        self.masks = {}

        if mode == "train":
            self.transform = A.Compose(
                [
                    A.RandomScale(scale_limit=(-0.2, 0.2), p=1),
                    A.Resize(
                        height=size,
                        width=size,
                        interpolation=cv2.INTER_LINEAR,
                    ),
                    A.RandomBrightnessContrast(p=0.3),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.0, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.5
                    ),
                    A.OneOf(
                        [
                            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                        ],
                        p=0.3,
                    ),
                    # A.OneOf([A.GaussNoise(), A.MultiplicativeNoise()], p=0.5),
                    # A.OneOf([A.Blur(blur_limit=3), A.MedianBlur(), A.MotionBlur()], p=0.5),
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
                    A.Resize(height=size, width=size, interpolation=cv2.INTER_LINEAR),
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
