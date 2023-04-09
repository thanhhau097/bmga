#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # yolox-s
        # self.depth = 0.33
        # self.width = 0.50

        # yolox-m
        self.depth = 0.67
        self.width = 0.75

        # yolox-l
        # self.depth = 1.0
        # self.width = 1.0

        # yolox-x
        # self.depth = 1.33
        # self.width = 1.25

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        # self.data_dir = "/home/thanh/bmga/data"
        self.data_dir = "/home/thanh/bmga/data/external_data/all"
        self.train_ann = "train_coco_annotations.json"
        self.val_ann = "val_coco_annotations.json"

        self.num_classes = 3
        self.input_size = (640, 640)

        self.max_epoch = 300
        self.no_aug_epochs = 300
        self.data_num_workers = 32
        self.eval_interval = 1


    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Get dataset according to cache and cache_type parameters.
        Args:
            cache (bool): Whether to cache imgs to ram or disk.
            cache_type (str, optional): Defaults to "ram".
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
        """
        from yolox.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            img_size=self.input_size,
            name="train/images",
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=0,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name="validation/images" if not testdev else "test2017",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )
