# CLASSIFIER HUGGING FACE TRAINER BASELINE

## Scripts
```
python train.py --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 --logging_strategy steps --logging_steps 10 --fp16 --warmup_ratio 0.01 --lr_scheduler_type cosine --adam_eps 1e-6 --optim adamw_torch --do_train --do_eval --metric_for_best_model eval_iou_score --dataloader_num_workers 4 --learning_rate 1e-4 --num_train_epochs 100 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --remove_unused_columns False --load_best_model_at_end --output_dir ./outputs/ --report_to none --train_csv ./data/train.csv --data_dir ./data --encoder_name convnext_pico --ignore_data_skip True --overwrite_output_dir 
```

```
python infer.py --logging_strategy steps --logging_steps 1 --fp16 --dataloader_num_workers 4 --remove_unused_columns False --output_dir ./outputs_test/ --report_to none --train_csv ./data/train.csv --data_dir ./data --encoder_name tf_efficientnetv2_b0 --resume outputs/pytorch_model.bin
```


