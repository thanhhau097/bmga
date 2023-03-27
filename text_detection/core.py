import os
import torch
import cv2
import numpy as np
import math
from tqdm import tqdm

from .src.demo import Demo, Structure, Experiment, Configurable, Config



class TextDetectionModel:
    def __init__(
        self,
        weights_path,
        config_path,
        image_short_side,
        thresh,
        box_thresh,
        resize,
        polygon,
        device="cuda"
    ):
        args = {
            "resume": weights_path,
            "image_short_side": image_short_side,
            "thresh": thresh,
            "box_thresh": box_thresh,
            "resize": resize,
            "polygon": polygon
        }

        conf = Config()
        experiment_args = conf.compile(conf.load(config_path))['Experiment']
        experiment_args.update(cmd=args)
        experiment = Configurable.construct_class_from_config(experiment_args)

        self.predictor = Demo(experiment, experiment_args, cmd=args)

    def predict(self, image_paths, visualize=False):
        all_outputs = []
        for path in tqdm(image_paths):
            outputs = self.predictor.inference(path, visualize)
            all_outputs.append(outputs)

        return all_outputs

    def postprocess(self, predictions, image_paths):
        return predictions
    
    def __call__(self, image_paths, visualize=False):
        predictions = self.predict(image_paths, visualize)
        return self.postprocess(predictions, image_paths)