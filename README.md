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


### External data
1. https://chartinfo.github.io/toolsanddata.html
2. https://iitmnlp.github.io/PlotQA/ + https://arxiv.org/pdf/1909.00997.pdf
3. https://arxiv.org/pdf/2203.10244v1.pdf

### External tools
1. https://github.com/kdavila/ChartInfo_annotation_tools