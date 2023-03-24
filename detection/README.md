

### Commands
```
cd data
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

```
PYTHONPATH=./ CUDA_VISIBLE_DEVICES=1 python tools/train.py -f /home/thanh/shared_disk/thanh/bmga/detection/src/exps/example/custom/bmga.py -d 1 -b 64 --fp16 -o -c /home/thanh/shared_disk/thanh/bmga/data/yolox_s.pth --cache
```