from lib2to3.pgen2 import token
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, AutoModelForTokenClassification, Trainer, \
    DataCollatorForTokenClassification
import json
import numpy as np

with open("C:/projects/trexpark/src/token_classification/product_title_ner/sample_data/labels.json", "rt") as fp:
    lable_dict = json.load(fp)
label_names = list(dict(lable_dict).keys())

metric = load_metric("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    flattened_results = {
        "overall_precision": results["overall_precision"],
        "overall_recall": results["overall_recall"],
        "overall_f1": results["overall_f1"],
        "overall_accuracy": results["overall_accuracy"],
    }
    for k in results.keys():
        if (k not in flattened_results.keys()):
            flattened_results[k + "_f1"] = results[k]["f1"]

    return flattened_results


def align_labels(tokenizer, label_dict):
    def _align_labels(sample):
        tokenized_inputs = tokenizer(sample["text"], truncation=True, is_split_into_words=False)
        tokens = [tokenizer.convert_ids_to_tokens(input_id) for input_id in tokenized_inputs["input_ids"]]
        labels = []
        # label_json = json.loads(sample["label"])
        for i, (token, label_text) in enumerate(zip(tokens, sample["label"])):
            offsets = tokenized_inputs.encodings[i].offsets
            label_ids = [0] * len(offsets)
            if label_text is not None:
                label_items = json.loads(label_text)
                for item in label_items:
                    start = 0
                    while offsets[start][0] < item["start"] or offsets[start][1] == 0:
                        start += 1
                    end = start
                    while offsets[end][1] <= item["end"] and offsets[end][1] != 0:
                        end += 1
                    if offsets[end - 1][1] < item["end"]:
                        end += 1
                    label_b = label_dict["B-" + item["labels"][0]]
                    label_i = label_dict["I-" + item["labels"][0]]
                    for j in range(start, end):
                        if j == start:
                            label_ids[j] = label_b
                        else:
                            label_ids[j] = label_i
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    return _align_labels


if __name__ == "__main__":
    # 解析训练参数
    parser = HfArgumentParser(TrainingArguments)
    training_args, = parser.parse_args_into_dataclasses()
    training_args.seed = 12345

    # 从之前训练的checkpoint恢复
    # training_args.resume_from_checkpoint = "output/checkpoint-14000"

    # 设置gradient_checkpointing为True，可以减少显存占用，但是由于梯度数据会写入硬盘暂存，训练速度会有所降低
    training_args.gradient_checkpointing = True

    # 加载训练和测试样本
    datafiles = {"train": "C:/projects/trexpark/src/token_classification/product_title_ner/sample_data/train.csv"}
    train_ds = load_dataset("csv", data_files=datafiles, split="train")
    tokenizer = AutoTokenizer.from_pretrained("youzanai/bert-product-title-chinese", truncation=True)
    map_func = align_labels(tokenizer, lable_dict)
    train_ds = train_ds.map(map_func, batched=True)
    train_ds.set_format(columns=["input_ids", "attention_mask", "labels"])
    eval_ds = train_ds
    eval_ds.set_format(columns=["input_ids", "attention_mask", "labels"])
    data_collator = DataCollatorForTokenClassification(tokenizer, padding="max_length", max_length=128,
                                                       label_pad_token_id=-100)
    model = AutoModelForTokenClassification.from_pretrained(
        "youzanai/bert-product-title-chinese", num_labels=9)

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

    pass
