import sys

sys.path.insert(0, "./text_detection/src")
sys.path.append("./classification/src")
sys.path.append("./detection/src")
sys.path.append("./text_recognition/src")

import os
import pandas as pd

from classification.core import ClassificationModel
from detection.core import ObjectDetectionModel
from text_recognition.core import TextRecognitionModel
from text_detection.core import TextDetectionModel
from postprocessing.core import Postprocessing


class Pipeline:
    def __init__(
        self,
        graph_classfication_config: dict,
        x_type_classification_config: dict,
        y_type_classification_config: dict,
        keypoint_detection_config: dict,
        text_detection_config: dict,
        text_recognition_config: dict,
    ) -> None:
        self.classification_model = ClassificationModel(**graph_classfication_config)
        self.x_type_classification_model = ClassificationModel(**x_type_classification_config)
        self.y_type_classification_model = ClassificationModel(**y_type_classification_config)
        self.keypoint_detection_model = ObjectDetectionModel(**keypoint_detection_config)
        self.text_detection_model = TextDetectionModel(**text_detection_config)
        self.text_recognition_model = TextRecognitionModel(**text_recognition_config)
        self.postprocessing = Postprocessing()

    def __call__(self, image_paths: list):
        # classification_results = self.classification_model.predict(image_paths)
        # x_type_classification_results = self.x_type_classification_model.predict(image_paths)
        # y_type_classification_results = self.y_type_classification_model.predict(image_paths)
        # keypoint_detection_results = self.keypoint_detection_model.predict(image_paths)
        text_detection_results = self.text_detection_model.predict(image_paths)
        text_recognition_results = self.text_recognition_model.predict(image_paths)
        return self.postprocess(
            classification_results,
            x_type_classification_results,
            y_type_classification_results,
            keypoint_detection_results,
            text_detection_results,
            text_recognition_results,
            image_paths,
        )

    def postprocess(
        self,
        classification_results,
        x_type_classification_results,
        y_type_classification_results,
        keypoint_detection_results,
        text_detection_results,
        text_recognition_results,
        image_paths,
    ):
        return 1


if __name__ == "__main__":
    graph_classfication_config = {
        "model_name": "resnet50",
        "n_classes": 5,
        "weights_path": "./weights/graph_classification.pth",
    }

    x_type_classification_config = {
        "model_name": "resnet50",
        "n_classes": 2,
        "weights_path": "./weights/x_type_classification.pth",
    }

    y_type_classification_config = {
        "model_name": "resnet50",
        "n_classes": 2,
        "weights_path": "./weights/y_type_classification.pth",
    }

    keypoint_detection_config = {
        "name": "keypoint_detection",
        "experiment_path": "./detection/src/exps/example/custom/bmga.py",
        "weights_path": "./weights/keypoint_detection.pth",
        "classes": ["value", "x", "y", "x_label", "y_label"],
        "conf_thre": 0.15,
        "nms_thre": 0.25,
        "test_size": (640, 640),
    }

    text_detection_config = {
        "weights_path": "./weights/synthtext_totaltext_res50_dcn_fpn_scale_spatial",
        # "config_path": "/home/thanh/bmga/text_detection/src/experiments/seg_detector/totaltext_resnet50_deform_thre.yaml",
        "config_path": "/home/thanh/bmga/text_detection/src/experiments/ASF/td500_resnet50_deform_thre_asf.yaml",
        "image_short_side": 320,
        "thresh": 0.1,
        "box_thresh": 0.6,
        "resize": False,
        "polygon": True,
    }

    text_recognition_config = {
        "weights_path": "baudm/parseq",
        "model_name": "parseq",
    }

    pipeline = Pipeline(
        graph_classfication_config,
        x_type_classification_config,
        y_type_classification_config,
        keypoint_detection_config,
        text_detection_config,
        text_recognition_config,
    )

    image_folder = "./data/validation/images"
    image_paths = [os.path.join(image_folder, x) for x in os.listdir(image_folder) if ".jpg" in x]

    pipeline(image_paths)
