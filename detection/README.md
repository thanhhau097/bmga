

### Commands
```
cd data
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

```
PYTHONPATH=./ CUDA_VISIBLE_DEVICES=0 python tools/train.py -f /home/thanh/bmga/detection/src/exps/example/custom/bmga.py -d 1 -b 64 --fp16 -o -c /home/thanh/bmga/data/yolox_m.pth --cache
```

### Rules:
1. Detect x points and y points using linear regression (x mean, y mean) and position (left and bottom of image)
2. Map x/y labels to corresponding point to filter out outlier and add missing point if there is