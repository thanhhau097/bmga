### 

### TODO:
- [x] Collect external data
- [x] Generate data for training
- [x] Post process for line data generation, only keep points that are corresponding to the x labels
- [x] Improve db detection model and hyperparameters tuning: db_x_labels_new

- [x] Improve keypoint detection model: may be we should use separate model for each type of graph: line/scatter/bar
- [x] Hanle multiple words in one label box using DB model
- [x] Rotate horizontal (maybe all?) labels before OCR (if necessary)

- [ ] With external data, we can:
    - [ ] Error in ChartInfo data: with the y labels that contain %, the value boxes are wrong
    - [x] ssUse original data + all chartinfo data to train x/y labels DB model and x/y points detection model
    - [x] Use original data + filtered chartinfo data to train value boxes detection model

- [ ] Line segmentation model: 
    - [ ] pseudo labels
    - [ ] full data
- [ ] Improve scatter model: 
    - [ ] external data -> full data
- [ ] x/y keypoint detection model
    - [ ] keypoint models: mmpose
    - [ ] full data

- [ ] Vertical bar: histogram charts are all wrong on y, missing x labels

- [ ] Convert horizontal bar data to vertical bar data to train vertical bar only
- [ ] Model Ensemble
    - [ ] Classification: graph, x_type
    - [ ] Object detection: keypoint
    - [ ] Text detection: x_label, y_label


### Approach

#### Approach 1: End-to-End solution using Image to Text model
- Predict directly type of the graph and value for each point in graph

#### Approach 2: Multi-stages pipeline
1. Use keypoint detection model to detect the points on the graph as well as the points in each axis:
    - For graph: predict all points
    - For axis: predict only key points, number of points is equal to number of labels in each axis
2. Use (Text detection + Recognition) / Donut model to predict label in each axis
3. Use classification model / Donut to predict the type of label: categorical or numerical and type of the graph
4. Map labels in each axis to corresponding points
    - If the label is categorical: mapping the points in the graph to categorical classes
    - If the label is numerical: mapping the point to x/y-axis then calculate the value based on the ratio

#### Approach 3: End-to-end solution with two branches/or 2 different models
1. Branch 1: predict keypoints from image: https://github.com/HRNet/DEKR
    - For graph: predict all points
    - For axis: predict only key points, number of points is equal to number of labels in each axis
2. Branch 2: Text generation branch to predict label in each axis, type of the labels: categorical/numerical, type of graph
3. Map labels in each axis to corresponding points

#### Approach 4:
1. Image classification model for graph classification, x_type/y_type classification -> DONE
2. 
    - Object detection model to detect keypoint and labels box
    - maybe we should use segmentation model for label box instead to avoid overlap between label boxes/or 2 DB models to detect
3. Text/word detection + recognition (ready solution) -> map with the boxes at step 2 to get x_labels and y_labels
4. Map labels in each axis to corresponding points

### External data
1. https://chartinfo.github.io/toolsanddata.html: good dataset, need to filter
2. https://iitmnlp.github.io/PlotQA/ + https://arxiv.org/pdf/1909.00997.pdf: good dataset, need to filter
3. https://arxiv.org/pdf/2203.10244v1.pdf: https://github.com/vis-nlp/ChartQA/tree/main/ChartQA%20Dataset + https://drive.google.com/file/d/17-aqtiq_KJ16PIGOp30W0y6OJNax6SVT/view: good, need to filter
4. FIGUREQA: https://arxiv.org/pdf/1710.07300.pdf + https://github.com/Maluuba/FigureQA: good, need to filter: https://www.microsoft.com/en-hk/download/details.aspx?id=100635: good, need to filter
5. DVQA: https://arxiv.org/pdf/1801.08163.pdf: has text labels and final info.

#### Pretrained data:
1. https://github.com/tingyaohsu/SciCap
2. 
    - https://github.com/JasonObeid/Chart2Text: only have chart type and final info
    - https://github.com/vis-nlp/Chart-to-text: only have final info

### External tools
1. https://github.com/kdavila/ChartInfo_annotation_tools: https://www.dropbox.com/s/fqnoq6bwnkgfqa9/ICDAR2023_CHARTINFO_UB_UNITEC_PMC_TRAIN_V1.0.zip

### Data generation
1. https://squid.org/rpg-random-generator

### Command
Neptune
```
export NEPTUNE_PROJECT="thanhhau097/bmga"
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMTRjM2ExOC1lYTA5LTQwODctODMxNi1jZjEzMjdlMjkxYTgifQ=="
```


```
CUDA_VISIBLE_DEVICES=1 python -m donut.src.train --config ./donut/src/config/swinv2-bmga.yaml --output_dir outputs --do_train --do_eval --remove_unused_columns False --per_device_train_batch_size 12 --per_device_eval_batch_size 128 --learning_rate 2e-5 --warmup_ratio 0.01 --lr_scheduler_type cosine --save_strategy epoch --evaluation_strategy epoch --logging_strategy steps --logging_steps 20 --save_total_limit 2 --load_best_model_at_end True --fp16 --optim adamw_torch --weight_decay 1e-2 --num_train_epochs 20 --metric_for_best_model eval_class_acc --greater_is_better=True --dataloader_num_workers=32 --max_grad_norm=1.0 --gradient_accumulation_steps=4 --overwrite_output_dir=True --report_to neptune
```

Validation:
```
CUDA_VISIBLE_DEVICES=1 python -m donut.src.train --config ./donut/src/config/swinv2-bmga.yaml --output_dir outputs --do_eval --remove_unused_columns False --per_device_train_batch_size 8 --per_device_eval_batch_size 96 --learning_rate 2e-5 --warmup_ratio 0.01 --lr_scheduler_type cosine --save_strategy epoch --evaluation_strategy epoch --logging_strategy steps --logging_steps 20 --save_total_limit 2 --load_best_model_at_end True --fp16 --optim adamw_torch --weight_decay 1e-2 --num_train_epochs 20 --metric_for_best_model eval_loss --dataloader_num_workers=32 --max_grad_norm=1.0  --report_to none
```

### Ensemble methods
1. https://github.com/ZFTurbo/Weighted-Boxes-Fusion
2. https://www.kaggle.com/code/pypiahmad/great-barrier-reef-yolox-yolov5-ensemble