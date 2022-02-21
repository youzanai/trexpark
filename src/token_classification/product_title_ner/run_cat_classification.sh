#!/bin/sh

python title_token_classification.py --output_dir=output --do_train --do_eval --overwrite_output_dir --num_train_epochs=5000 --save_total_limit=5 --logging_steps=50 --per_device_train_batch_size=128 --per_device_eval_batch_size=128 --evaluation_strategy=steps --learning_rate=1e-4