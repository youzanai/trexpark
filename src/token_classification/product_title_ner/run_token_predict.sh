#!/bin/sh

python token_predicor.py --predict_data_file=sample_data/test.csv --label_file=sample_data/labels.json --format=csv --model_name=output --tokenizer_name=youzanai/bert-product-title-chinese --batch_size=512 --streaming=False