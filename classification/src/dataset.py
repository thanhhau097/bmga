import os

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class BMGADataset(Dataset):
    def __init__(self, jsonl_path, image_dir, classes, classification_type, size=(640, 640), transform=None):
        self.df = pd.read_json(jsonl_path, lines=True)
        self.image_dir = image_dir
        self.transform = transform
        self.classes = classes
        self.classification_type = classification_type
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row["file_name"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.size)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        # channel first
        image = image.transpose(2, 0, 1)
        image = image.astype("float32")
        image /= 255.0

        if self.classification_type == "graph":
            label = self.classes.index(row["ground_truth"]["gt_parse"]["class"])
        elif self.classification_type == "x_type":
            label = self.classes.index(row["ground_truth"]["gt_parse"]["x_type"])
        elif self.classification_type == "y_type":
            label = self.classes.index(row["ground_truth"]["gt_parse"]["y_type"])
        else:
            raise ValueError("Invalid classification type")

        return torch.tensor(image).float(), torch.tensor(label)


def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return {"images": images, "labels": labels}


class HistogramClassificationDataset(Dataset):
    def __init__(self, df, image_dir, classes, size=(640, 320), transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.classes = classes
        self.size = size
        self.image_paths = self.df["image"].values
        self.labels = self.df["label"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image_path = os.path.join(self.image_dir, path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.size)
        # cutout bottom half
        image = image[image.shape[0] // 2:, :, :]
        # channel first
        image = image.transpose(2, 0, 1)
        image = image.astype("float32")
        image /= 255.0

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        label = self.classes.index(self.labels[idx])
        return torch.tensor(image).float(), torch.tensor(label)
