### Classification module
Used for: graph classification, x_type/y_type classification

#### Commands
```
PYTHONPATH=./classification/src CUDA_VISIBLE_DEVICES=0 python -m classification.src.train --output_dir outputs_graph_classification --do_train --do_eval --remove_unused_columns False --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --learning_rate 1e-4 --warmup_ratio 0.01 --lr_scheduler_type cosine --save_strategy epoch --evaluation_strategy epoch --logging_strategy steps --logging_steps 20 --save_total_limit 2 --load_best_model_at_end True --fp16 --optim adamw_torch --weight_decay 1e-2 --num_train_epochs 20 --metric_for_best_model eval_accuracy --greater_is_better=True --dataloader_num_workers=32 --max_grad_norm=1.0 --overwrite_output_dir=True --classification_type graph --report_to none
```

```
PYTHONPATH=./classification/src CUDA_VISIBLE_DEVICES=0 python -m classification.src.train --output_dir outputs_x_type_classification --do_train --do_eval --remove_unused_columns False --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --learning_rate 1e-4 --warmup_ratio 0.01 --lr_scheduler_type cosine --save_strategy epoch --evaluation_strategy epoch --logging_strategy steps --logging_steps 20 --save_total_limit 2 --load_best_model_at_end True --fp16 --optim adamw_torch --weight_decay 1e-2 --num_train_epochs 20 --metric_for_best_model eval_accuracy --greater_is_better=True --dataloader_num_workers=32 --max_grad_norm=1.0 --overwrite_output_dir=True --classification_type x_type --report_to none
```

```
PYTHONPATH=./classification/src CUDA_VISIBLE_DEVICES=0 python -m classification.src.train --output_dir outputs_y_type_classification --do_train --do_eval --remove_unused_columns False --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --learning_rate 1e-4 --warmup_ratio 0.01 --lr_scheduler_type cosine --save_strategy epoch --evaluation_strategy epoch --logging_strategy steps --logging_steps 20 --save_total_limit 2 --load_best_model_at_end True --fp16 --optim adamw_torch --weight_decay 1e-2 --num_train_epochs 20 --metric_for_best_model eval_accuracy --greater_is_better=True --dataloader_num_workers=32 --max_grad_norm=1.0 --overwrite_output_dir=True --classification_type y_type --report_to none
```

Hisstogram training
```
PYTHONPATH=./classification/src CUDA_VISIBLE_DEVICES=0 python -m classification.src.train --output_dir outputs_histogram_classification --do_train --do_eval --remove_unused_columns False --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --learning_rate 1e-4 --warmup_ratio 0.01 --lr_scheduler_type cosine --save_strategy epoch --evaluation_strategy epoch --logging_strategy steps --logging_steps 20 --save_total_limit 2 --load_best_model_at_end True --fp16 --optim adamw_torch --weight_decay 1e-2 --num_train_epochs 20 --metric_for_best_model eval_accuracy --greater_is_better=True --dataloader_num_workers=32 --max_grad_norm=1.0 --overwrite_output_dir=True --classification_type histogram --report_to none --train_image_folder ./data --val_image_folder ./data
```