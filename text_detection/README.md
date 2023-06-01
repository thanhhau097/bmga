### Commands
```
CUDA_VISIBLE_DEVICES=0 python train.py /home/thanh/bmga/text_detection/src/experiments/ASF/td500_resnet50_deform_thre_asf.yaml --resume ../../weights/synthtext_finetune_ic19_res50_dcn_fpn_dbv2 --batch_size 32 --validate --num_workers 10
CUDA_VISIBLE_DEVICES=0 python train.py /home/thanh/bmga/text_detection/src/experiments/ASF/td500_resnet152_thre_asf.yaml --batch_size 16 --validate --num_workers 10
CUDA_VISIBLE_DEVICES=0 python train.py /home/thanh/bmga/text_detection/src/experiments/ASF/td500_resnet152_deform_thre_asf.yaml --batch_size 16 --validate --num_workers 10

```