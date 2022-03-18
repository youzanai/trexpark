#!/bin/sh

`python title_token_classification.py --output_dir=output --do_train --train_data_file=sample_data/train.csv --do_eval --eval_data_file=sample_data/test.csv --overwrite_output_dir --num_train_epochs=500 --save_total_limit=5 --logging_steps=100 --save_steps=100 --per_device_train_batch_size=512 --per_device_eval_batch_size=512 --evaluation_strategy=steps --learning_rate=1e-3