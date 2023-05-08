TEST_MODE = False

if TEST_MODE:
    DATA_FOLDER = "/kaggle/input/benetech-making-graphs-accessible/test/"
    WEIGHTS_FOLDER = "/kaggle/input/bmgaweights/"
else:
    DATA_FOLDER = "./data/validation/"
    WEIGHTS_FOLDER = "./weights/"


# export environment variables
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys

sys.path.insert(0, "./text_detection/src")
sys.path.append("./classification/src")
sys.path.append("./detection/src")
sys.path.append("./text_recognition/src")


import numpy as np
import pandas as pd
import os

from classification.core import ClassificationModel
from detection.core import ObjectDetectionModel
from text_recognition.core import TextRecognitionModel
from text_detection.core import TextDetectionModel
from postprocessing.core import Postprocessing


# image_folder = "./data/train/images"
# origin_image_paths = [os.path.join(image_folder, x) for x in os.listdir(image_folder) if ".jpg" in x][:500]
# image_paths = [os.path.join(image_folder, x) for x in os.listdir(image_folder) if ".jpg" in x][:500]
image_folder = os.path.join(DATA_FOLDER, "images")

origin_image_paths = [os.path.join(image_folder, x) for x in os.listdir(image_folder) if ".jpg" in x]
image_paths = [os.path.join(image_folder, x) for x in os.listdir(image_folder) if ".jpg" in x]


graph_classfication_config = {
    "model_name": "resnet50",
    "n_classes": 5,
    "weights_path": os.path.join(WEIGHTS_FOLDER, "graph_classification.pth"),
}

x_type_classification_config = {
    "model_name": "resnet50",
    "n_classes": 2,
    "weights_path": os.path.join(WEIGHTS_FOLDER, "x_type_classification.pth"),
}

y_type_classification_config = {
    "model_name": "resnet50",
    "n_classes": 2,
    "weights_path": os.path.join(WEIGHTS_FOLDER, "y_type_classification.pth"),
}

keypoint_detection_config = {
    "name": "keypoint_detection",
    "experiment_path": "./detection/src/exps/example/custom/bmga.py",
    "weights_path": os.path.join(WEIGHTS_FOLDER, "keypoint_detection.pth"),
    "classes": ["value", "x", "y"], #, "x_label", "y_label"],
    "conf_thre": 0.15,
    "nms_thre": 0.25,
    "test_size": (640, 640),
}

text_detection_config = {
    "weights_path": os.path.join(WEIGHTS_FOLDER, "synthtext_finetune_ic19_res50_dcn_fpn_dbv2"),
    "config_path": "text_detection/src/experiments/ASF/td500_resnet50_deform_thre_asf_inference.yaml",
    "image_short_side": 768,
    "thresh": 0.1,
    "box_thresh": 0.05,
    "resize": False,
    "polygon": True,
}

x_labels_text_detection_config = {
    "weights_path": os.path.join(WEIGHTS_FOLDER, "db_x_labels"),
    # "weights_path": "/home/thanh/bmga/text_detection/src/outputs/workspace/src/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/model/model_epoch_9_minibatch_34000",
    "config_path": "./text_detection/src/experiments/ASF/td500_resnet50_deform_thre_asf_inference.yaml",
    "image_short_side": 640,
    "thresh": 0.15,
    "box_thresh": 0.25,
    "resize": False,
    "polygon": True,
}

y_labels_text_detection_config = {
    "weights_path": os.path.join(WEIGHTS_FOLDER, "db_y_labels"),
    "config_path": "./text_detection/src/experiments/ASF/td500_resnet50_deform_thre_asf_inference.yaml",
    "image_short_side": 768,
    "thresh": 0.05,
    "box_thresh": 0.25,
    "resize": False,
    "polygon": True,
}

text_recognition_config = {
    "weights_path": os.path.join(WEIGHTS_FOLDER, "parseq-bb5792a6.pt"),
    # "weights_path": "baudm/parseq",
    "model_name": "parseq",
    "config_path": os.path.join(WEIGHTS_FOLDER, "parseq_hparams.json"),
}

graph_classification_model = ClassificationModel(**graph_classfication_config)
x_type_classification_model = ClassificationModel(**x_type_classification_config)
y_type_classification_model = ClassificationModel(**y_type_classification_config)
keypoint_detection_model = ObjectDetectionModel(**keypoint_detection_config)
text_detection_model = TextDetectionModel(**text_detection_config)
text_recognition_model = TextRecognitionModel(**text_recognition_config)
text_recognition_model.parseq.eval()
print()


# read ground truth from /home/thanh/bmga/data/validation/metadata.jsonl
import json

if not TEST_MODE:
    # with open("/home/thanh/bmga/data/train/metadata.jsonl", "r") as f:
    with open("/home/thanh/bmga/data/validation/metadata.jsonl", "r") as f:
        metadata = [json.loads(x) for x in f.readlines()]

    metadata_dict = {}
    for x in metadata:
        metadata_dict[x["file_name"]] = x

    filtered_image_paths = []
    filtered_original_image_paths = []

    for image_path in image_paths:
        if "images/" + image_path.split("/")[-1] not in metadata_dict.keys():
            continue
        filtered_image_paths.append(image_path)
        filtered_original_image_paths.append(image_path)

    image_paths = filtered_image_paths
    origin_image_paths = filtered_original_image_paths


from matplotlib import pyplot as plt
import cv2


# function to convert polygon points to smallest 4 points polygon
def convert_polygon_to_min_rect(polygon):
    polygon = np.array(polygon)
    polygon = polygon.reshape(-1, 2)
    polygon = polygon.astype(np.float32)
    rect = cv2.minAreaRect(polygon)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    return box

def crop_polygon_from_image(image, polygon):
    polygon = convert_polygon_to_min_rect(polygon)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [polygon], 0, 255, -1, cv2.LINE_AA)
    out = 255 - np.zeros_like(image)
    out[mask == 255] = image[mask == 255]

    # return crop from image
    try:
        crop = out[np.min(polygon[:, 1]):np.max(polygon[:, 1]), np.min(polygon[:, 0]):np.max(polygon[:, 0])]
    except:
        crop = np.ones((32, 100, 3), dtype=np.uint8) * 255
    return crop


def filter_x_polygons(polygons, img_height, img_path):
    # first, draw a line along y axis then count the number of x_label_boxes that intersect with the line
    max_count = 0
    max_count_line_y = 0

    for line_y in range(img_height):
        count = 0
        for polygon in polygons:
            if polygon:
                min_y = min([x[1] for x in polygon])
                max_y = max([x[1] for x in polygon])
            else:
                min_y = 0
                max_y = 0

            if min_y <= line_y <= max_y:
                count += 1
        if count > max_count:
            max_count = count
            max_count_line_y = line_y

    # filter out y_label_boxes that intersect with the line
    filtered_x_label_polygons = []
    for polygon in polygons:
        if polygon:
            min_y = min([x[1] for x in polygon])
            max_y = max([x[1] for x in polygon])
        else:
            min_y = 0
            max_y = 0

        if min_y <= max_count_line_y <= max_y:
            filtered_x_label_polygons.append(polygon)

    return filtered_x_label_polygons


def filter_y_polygons(polygons, img_width, image):
    # first, draw a line along x axis then count the number of y_label_boxes that intersect with the line
    max_count = 0
    max_count_line_x = 0

    for line_x in range(img_width):
        count = 0
        for polygon in polygons:
            if polygon:
                min_x = min([x[0] for x in polygon])
                max_x = max([x[0] for x in polygon])
            else:
                min_x = 0
                max_x = 0
            w = max_x - min_x
            if min_x + w // 4 <= line_x <= max_x - w // 4:
                count += 1
        if count > max_count:
            max_count = count
            max_count_line_x = line_x

    # filter out y_label_boxes that intersect with the line
    filtered_y_label_polygons = []
    for polygon in polygons:
        if polygon:
            min_x = min([x[0] for x in polygon])
            max_x = max([x[0] for x in polygon])
        else:
            min_x = 0
            max_x = 0
        if min_x <= max_count_line_x <= max_x:
            filtered_y_label_polygons.append(polygon)

    return filtered_y_label_polygons
    # # second, do text recognition on y_label_boxes
    # crops = []
    # for polygon in filtered_y_label_polygons:
    #     crop = crop_polygon_from_image(image, polygon)
    #     crops.append(crop)

    # text_recognition_results = text_recognition_model.predict(crops)

    # # filter out those boxes that the values can't be converted to float: TODO: only case that y labels are numbers, have to update
    # filtered_y_label_boxes_2 = []
    # for i, box in enumerate(filtered_y_label_polygons):
    #     try:
    #         text = "".join([c for c in text_recognition_results[0][i][0] if c in "0123456789."])
    #         if not text:
    #             float(text)
    #         filtered_y_label_boxes_2.append(box)
    #     except:
    #         pass

    # return filtered_y_label_boxes_2

def calculate_iou(polygon1, polygon2, image):
    # calculate iou between two polygons
    polygon1 = np.array(polygon1)
    polygon2 = np.array(polygon2)
    polygon1 = polygon1.reshape(-1, 2)
    polygon2 = polygon2.reshape(-1, 2)
    polygon1 = polygon1.astype(np.float32)
    polygon2 = polygon2.astype(np.float32)

    rect1 = cv2.minAreaRect(polygon1)
    box1 = cv2.boxPoints(rect1)
    box1 = np.int0(box1)

    rect2 = cv2.minAreaRect(polygon2)
    box2 = cv2.boxPoints(rect2)
    box2 = np.int0(box2)

    mask1 = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask1, [box1], 0, 255, -1, cv2.LINE_AA)
    mask2 = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask2, [box2], 0, 255, -1, cv2.LINE_AA)

    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score

def calculate_label_polygons_accuracy(pred_polygons, gt_polygons, image, is_x_label=True, iou_thre=0.5):
    if len(pred_polygons) != len(gt_polygons):
        return 0
    
    if is_x_label:
        gt_polygons = sorted(gt_polygons, key=lambda x: min([y[0] for y in x]) if x else 0)
        gt_polygons = sorted(pred_polygons, key=lambda x: min([y[0] for y in x]) if x else 0)
    else:
        gt_polygons = sorted(gt_polygons, key=lambda x: min([y[1] for y in x]) if x else 0)
        gt_polygons = sorted(pred_polygons, key=lambda x: min([y[1] for y in x]) if x else 0)

    iou_score = 0
    for i in range(len(gt_polygons)):
        iou = calculate_iou(gt_polygons[i], gt_polygons[i], image)
        if iou > iou_thre:
            iou_score += 1

    if iou_score == len(gt_polygons):
        return 1

    return 0

def visualize(image_path, value_boxes, x_boxes, y_boxes, x_labels_polygons, y_labels_polygons):
    image = cv2.imread(image_path)

    if value_boxes is not None:
        for box in value_boxes:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

    if x_boxes is not None:
        for box in x_boxes:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    if y_boxes is not None:
        for box in y_boxes:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)   
    
    if x_labels_polygons is not None:
        # visualize x_label_boxes
        for polygon in x_labels_polygons:
            polygon = np.array(polygon)
            polygon = polygon.reshape(-1, 2)
            polygon = polygon.astype(np.int32)
            cv2.drawContours(image, [polygon], 0, (255, 255, 0), 2)

    if y_labels_polygons is not None:
        # visualize y_label_boxes
        for polygon in y_labels_polygons:
            polygon = np.array(polygon)
            polygon = polygon.reshape(-1, 2)
            polygon = polygon.astype(np.int32)
            cv2.drawContours(image, [polygon], 0, (0, 255, 255), 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(image)


graph_classes = ['dot', 'line', 'scatter', 'vertical_bar', "horizontal_bar"]

graph_type_predictions = graph_classification_model.predict(image_paths=image_paths)

# convert predictions to graph type
graph_type_predictions = np.argmax(graph_type_predictions, axis=1)
graph_type_predictions = [graph_classes[i] for i in graph_type_predictions]


if not TEST_MODE:
    gt_classes = []

    for image_path in image_paths:
        gt_classes.append(metadata_dict["images/" + image_path.split("/")[-1]]["ground_truth"]["gt_parse"]["class"])

    # calculate accuracy
    acc = 0
    for idx in range(len(image_paths)):
        if graph_type_predictions[idx] == gt_classes[idx]:
            acc += 1

    print("acc: ", acc / len(image_paths))
    print(np.unique(gt_classes, return_counts=True))


y_labels_text_detection_model = TextDetectionModel(**y_labels_text_detection_config)
for threshold in [0.1, 0.15, 0.2, 0.25, 0.3]:
    for box_thre in [0.1, 0.15, 0.2, 0.25, 0.3]:
        x_labels_text_detection_config["box_thresh"] = box_thre
        x_labels_text_detection_config["thresh"] = threshold
        x_labels_text_detection_model = TextDetectionModel(**x_labels_text_detection_config)

        # x_labels_predictions = x_labels_text_detection_model.predict(image_paths=image_paths)
        # y_labels_predictions = y_labels_text_detection_model.predict(image_paths=image_paths)

        from tqdm import tqdm
        # calucate accuracy
        x_acc = 0
        y_acc = 0

        for idx in tqdm(range(len(image_paths))):
            image = cv2.imread(image_paths[idx])

            try:
                x_labels_predictions = x_labels_text_detection_model.predict(image_paths=[image_paths[idx]])
                x_labels_polygons = x_labels_predictions[0][0][0]
            except:
                x_labels_polygons = []

            try:
                y_labels_predictions = y_labels_text_detection_model.predict(image_paths=[image_paths[idx]])
                y_labels_polygons = y_labels_predictions[0][0][0]
            except:
                y_labels_polygons = []

            x_labels_polygons = filter_x_polygons(
                x_labels_polygons,
                image.shape[0],
                image_paths[idx],
            )

            y_labels_polygons = filter_y_polygons(
                y_labels_polygons,
                image.shape[1],
                image
            )
            
            x_acc += calculate_label_polygons_accuracy(
                x_labels_polygons,
                metadata_dict["images/" + image_paths[idx].split("/")[-1]]["ground_truth"]["gt_parse"]["x_labels_polygons"],
                image=image,
                is_x_label=True,
            )
            
            y_acc += calculate_label_polygons_accuracy(
                y_labels_polygons,
                metadata_dict["images/" + image_paths[idx].split("/")[-1]]["ground_truth"]["gt_parse"]["y_labels_polygons"],
                image=image,
                is_x_label=False,    
            )

        print("-----------------------------------------------------------------")
        print("threshold: ", threshold)
        print("box_thre: ", box_thre)
        print("x_acc: ", x_acc / len(image_paths))
        print("y_acc: ", y_acc / len(image_paths))
