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

        if mode == 'train':
            self.transform = A.Compose(
                [  
                    A.RandomBrightnessContrast(p=0.5),
                    A.OneOf([A.ShiftScaleRotate(), A.GridDistortion(), A.ElasticTransform()], p=0.5),
                    A.OneOf([A.GaussNoise(), A.MultiplicativeNoise()], p=0.5),
                    A.OneOf([A.Blur(blur_limit=3), A.MedianBlur(), A.MotionBlur()], p=0.5),
                    A.Resize(width=int(size*1.1), height=int(size*1.1)),
                    A.RandomCrop(width=int(size), height=int(size)),
                    # A.HorizontalFlip(p=0.5),
                    # A.VerticalFlip(p=0.5),
                    # A.Rotate(p=0.8),
                    # A.Resize(width=size, height=size),
                    # A.Normalize(mean=[0.0], std=[1.0]),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(width=size, height=size), 
                    # A.Normalize(mean=[0.0], std=[1.0]), 
                    ToTensorV2()
                ]
            )
        
    def __len__(self):
        return len(self.df)

    def preprocess_df(self):
        if self.mode == 'train':
            self.df = self.df[self.df.is_train]
        else:
            self.df = self.df[~self.df.is_train]
        return self.df['image'].to_list(), self.df['labels'].to_list()

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.data_dir, self.image_paths[idx]), cv2.IMREAD_GRAYSCALE)

        label = cv2.imread(os.path.join(self.data_dir, self.labels[idx]), cv2.IMREAD_GRAYSCALE)
        if 'syn/' in self.image_paths[idx]:
            label = 255 - label
        transformed = self.transform(image=img, mask=label)
        return transformed['image'] / 255.0, transformed['mask']  / 255.0


class SegmentationInferenceDataset(Dataset):
    def __init__(self, df, data_dir, size, mode='val'):
        super().__init__()
        self.data_dir = data_dir
        self.df = df.reset_index(drop=True)
        self.image_paths = self.preprocess_df()
        self.transform = A.Compose(
            [
                A.Resize(width=size, height=size), 
                # A.Normalize(mean=[0.0], std=[1.0]), 
                ToTensorV2()
            ]
        )
        
    def __len__(self):
        return len(self.df)

    def preprocess_df(self):
        return self.df['image'].to_list()

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.data_dir, self.image_paths[idx]), cv2.IMREAD_GRAYSCALE)
        transformed = self.transform(image=img)
        return transformed['image'] / 255.0



def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return {"images": images, "labels": labels}
