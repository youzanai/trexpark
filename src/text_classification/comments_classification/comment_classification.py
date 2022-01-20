# Copyright 2022 The Youzan-AI Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#! /usr/bin/env python
# -*- coding: utf-8 -*-

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import (HfArgumentParser, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, EvalPrediction)
import numpy as np

def compute_metrics(p: EvalPrediction):
    """
        计算并返回分类准确率
    """
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = float((preds == labels).mean())
    return {
        "accuracy": acc
    }

def run(training_args):
    data_files = {
        "train":"./sample_data/train.csv",
        "eval":"./sample_data/eval.csv"
    }
    print('start load data')
    dataset = load_dataset("csv", data_files=data_files, streaming=False, delimiter='\t')

    pretrained_model_path = "youzanai/bert-product-comment-chinese" 
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    train_ds = dataset["train"]
    eval_ds = dataset["eval"]

    train_ds = train_ds.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length'), batched=True)
    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    eval_ds = eval_ds.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length'), batched=True)
    eval_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=2)

    for param in model.bert.parameters():
        param.requires_grad = False    

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics
    )

    print('start train')
    train_result = trainer.train()

    trainer.save_model()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

if __name__ == '__main__':
    parser = HfArgumentParser(TrainingArguments)
    training_args, = parser.parse_args_into_dataclasses()
    run(training_args)
