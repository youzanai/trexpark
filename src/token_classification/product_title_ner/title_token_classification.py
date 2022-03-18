from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, AutoModelForTokenClassification, Trainer, \
    DataCollatorForTokenClassification
import json
import numpy as np
from bert_crf import BertCRF
from dataclasses import field, dataclass
from typing import Optional

# 加载标签列表
with open("sample_data/labels.json", "rt") as fp:
    lable_dict = json.load(fp)
label_names = list(dict(lable_dict).keys())

# 使用seqeval作为metric
metric = load_metric("seqeval")


def compute_metrics(p):
    """
    计算metrics，包括准确率、召回率、F1，以及各个分类的F1
    """
    predictions, labels = p
    # predictions = np.argmax(predictions, axis=2)

    predictions = predictions.astype(int)
    labels = labels
    mask = np.where(labels >= 0, 0, -100)
    predictions = predictions + mask

    # 忽略-100的label
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
    """
    对文本进行tokenize，并且将标注与token进行对齐
    """

    def _align_labels(sample):
        tokenized_inputs = tokenizer(sample["text"], truncation=True, is_split_into_words=False)
        tokens = [tokenizer.convert_ids_to_tokens(input_id) for input_id in tokenized_inputs["input_ids"]]
        labels = []
        # label_json = json.loads(sample["label"])
        for i, (token, label_text) in enumerate(zip(tokens, sample["label"])):
            # 获取每个token的偏移量，类似[(0,0),(1,2),(2,3),(0,0)]的格式，其中第一个token为[CLS]，最后一个为[SEP]
            offsets = tokenized_inputs.encodings[i].offsets
            label_ids = [0] * len(offsets)
            # 将[CLS]和[SEP]的标签设置为-100
            label_ids[0] = 0
            label_ids[-1] = 0
            if label_text is not None:
                # 加载标签数据
                label_items = json.loads(label_text)
                for item in label_items:
                    # 对于每个标注，计算标注对应token的label。
                    # 标注格式为{"start": 12, "end": 14, "text": "陈醋", "labels": ["PRD"]}
                    # 标注格式和label-studio导出CSV格式的样本保持一致
                    start = 0
                    try:
                        while offsets[start][0] < item["start"] or offsets[start][1] == 0:
                            start += 1
                        end = start
                        while offsets[end][1] <= item["end"] and offsets[end][1] != 0:
                            end += 1
                        if offsets[end - 1][1] < item["end"]:
                            end += 1
                        label_b = label_dict['B-' + item["labels"][0]]
                        label_i = label_dict['I-' + item["labels"][0]]
                        cur_label = label_b
                        for j in range(start, end):
                            # Tokenizer 可能出现某个token从属于上一个的情况，这类token会以##开头
                            if j > start and not token[j].startswith("##"):
                                cur_label = label_i
                            label_ids[j] = cur_label
                    except Exception as err:
                        pass
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    return _align_labels


@dataclass
class TrainingDataArguments:
    """
    Arguments pertaining to what sample_data we are going to input our model for training and eval.
    """

    train_data_file: str = field(
        default=None, metadata={"help": "The input training sample_data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation sample_data file to evaluate the perplexity on (a text file)."},
    )


if __name__ == "__main__":
    # 解析训练参数
    parser = HfArgumentParser((TrainingDataArguments, TrainingArguments))
    training_data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.seed = 12345

    # 从之前训练的checkpoint恢复
    # training_args.resume_from_checkpoint = "output/checkpoint-4000"

    # 设置gradient_checkpointing为True，可以减少显存占用，但是由于梯度数据会写入硬盘暂存，训练速度会有所降低
    training_args.gradient_checkpointing = False

    # 加载训练和测试样本，由于示例样本比较少，train和eval用了同一个文件，实际使用时指定正确的文件即可
    # datafiles = {"train": "sample_data/brand_samples.csv"}
    # train_ds = load_dataset("csv", data_files=datafiles, split="train[:20480]")
    # eval_ds = load_dataset("csv", data_files=datafiles, split="train[-10240:]")
    datafiles = {'train': training_data_args.train_data_file}
    if training_args.do_eval and training_data_args.eval_data_file is not None:
        datafiles['eval'] = training_data_args.eval_data_file
        eval_ds = load_dataset("csv", data_files=datafiles, split="eval")
    train_ds = load_dataset("csv", data_files=datafiles, split="train")

    tokenizer = AutoTokenizer.from_pretrained("youzanai/bert-product-title-chinese", truncation=True)
    # 对样本文本进行tokenize，并且将标签和token进行对齐
    map_func = align_labels(tokenizer, lable_dict)
    train_ds = train_ds.map(map_func, batched=True)
    train_ds.set_format(columns=["input_ids", "attention_mask", "labels"])
    if training_args.do_eval:
        eval_ds = eval_ds.map(map_func, batched=True)
        eval_ds.set_format(columns=["input_ids", "attention_mask", "labels"])
    else:
        eval_ds = None
    # Padding训练样本，标签使用-100进行padding，可以避免标签产生loss
    data_collator = DataCollatorForTokenClassification(tokenizer, padding="max_length", max_length=128,
                                                       label_pad_token_id=-100)
    # 加载预训练模型
    model = BertCRF.from_pretrained(
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
