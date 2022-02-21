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


"""
    基于有赞的商品标题预训练模型finetune商品类目预测模型。测试数据用于验证fewshot条件下，freeze模型的预训练部分，仅仅增加一个dense层，采用少量样本进行建模的效果。
    我们随机挑选了10个类目，以及这10个类目中的各100个商品作为训练样本，而测试数据中包含平均每个类目800+的样本。模型在eval数据集上的准确率可以达到99.5%。
"""

import numpy as np
from transformers import (HfArgumentParser, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, EvalPrediction, DataCollatorWithPadding)
from datasets import load_dataset


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


if __name__ == '__main__':
    # 解析训练参数
    parser = HfArgumentParser(TrainingArguments)
    training_args, = parser.parse_args_into_dataclasses()
    training_args.seed = 12345

    # 从之前训练的checkpoint恢复
    # training_args.resume_from_checkpoint = "output/checkpoint-14000"

    # 设置gradient_checkpointing为True，可以减少显存占用，但是由于梯度数据会写入硬盘暂存，训练速度会有所降低
    training_args.gradient_checkpointing = True

    # 加载训练和测试样本
    data_files = {"train": "sample_data/sample_title_class_train.txt",
                  "eval": "sample_data/sample_title_class_eval.txt"}
    dataset_dict = load_dataset("json", data_files=data_files, streaming=False)
    train_ds = dataset_dict["train"]
    eval_ds = dataset_dict["eval"]

    # 对测试样本进行采样
    # eval_ds = eval_ds.shuffle()
    # eval_ds = eval_ds.select(range(10000))

    # 加载通过有赞商品标题预训练的bert模型
    tokenizer = AutoTokenizer.from_pretrained(
        "youzanai/bert-product-title-chinese")
    train_ds = train_ds.map(lambda e: tokenizer(
        e['text'], truncation=True, padding='max_length'), batched=True)
    train_ds.set_format(type='torch', columns=[
                        'input_ids', 'attention_mask', 'label'])
    eval_ds = eval_ds.map(lambda e: tokenizer(
        e['text'], truncation=True, padding='max_length'), batched=True)
    eval_ds.set_format(type='torch', columns=[
                       'input_ids', 'attention_mask', 'label'])
    data_collator = DataCollatorWithPadding(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        "youzanai/bert-product-title-chinese", num_labels=10)

    # 冻结预训练参数
    for param in model.bert.parameters():
        param.requires_grad = False

    # 使用Huggingface官方的trainer进行模型的finetune
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    train_result = trainer.train()

    # 保存模型
    trainer.save_model()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
