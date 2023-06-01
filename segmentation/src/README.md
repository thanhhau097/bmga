# SEGMENTATION HUGGING FACE TRAINER BASELINE

## Scripts

```
python infer.py --logging_strategy steps --logging_steps 1 --fp16 --dataloader_num_workers 4 --remove_unused_columns False --output_dir ./outputs_test/ --report_to none --train_csv ./data/train.csv --data_dir ./data --encoder_name tf_efficientnetv2_b0 --resume outputs/pytorch_model.bin
```


```
python -m torch.distributed.launch --use_env --nproc_per_node=8 train.py --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 --logging_strategy steps --logging_steps 10 --fp16 --warmup_ratio 0.01 --lr_scheduler_type cosine --adam_eps 1e-6 --optim adamw_torch --do_train --do_eval --metric_for_best_model eval_iou_score --dataloader_num_workers 12 --learning_rate 2e-3 --num_train_epochs 200 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --remove_unused_columns False --load_best_model_at_end --output_dir ./outputs_v2s/ --report_to none --train_csv data/train_line.csv --data_dir data/ --encoder_name tf_efficientnetv2_s_in21ft1k --ignore_data_skip True --overwrite_output_dir --size 256
```

```
python -m torch.distributed.launch --use_env --nproc_per_node=8 train.py --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 --logging_strategy steps --logging_steps 10 --fp16 --warmup_ratio 0.01 --lr_scheduler_type cosine --adam_eps 1e-6 --optim adamw_torch --do_train --do_eval --metric_for_best_model eval_iou_score --dataloader_num_workers 12 --learning_rate 2e-4 --num_train_epochs 100 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --remove_unused_columns False --load_best_model_at_end --output_dir ./outputs_v2s_512/ --report_to none --train_csv data/train_line.csv --data_dir data/ --encoder_name tf_efficientnetv2_s_in21ft1k --ignore_data_skip True --overwrite_output_dir --resume outputs_v2s/pytorch_model.bin
```