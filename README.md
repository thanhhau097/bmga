### 

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
1. Image classification model for graph classification, x_type/y_type classification
2. Object detection model to detect keypoint and labels box
3. Text/word detection + recognition (ready solution) -> map with the boxes at step 2 to get x_labels and y_labels
4. Map labels in each axis to corresponding points

### External data
1. https://chartinfo.github.io/toolsanddata.html
2. https://iitmnlp.github.io/PlotQA/ + https://arxiv.org/pdf/1909.00997.pdf
3. https://arxiv.org/pdf/2203.10244v1.pdf
4. https://github.com/JasonObeid/Chart2Text

### External tools
1. https://github.com/kdavila/ChartInfo_annotation_tools: https://www.dropbox.com/s/fqnoq6bwnkgfqa9/ICDAR2023_CHARTINFO_UB_UNITEC_PMC_TRAIN_V1.0.zip

### Data generation
1. https://squid.org/rpg-random-generator
2. 


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