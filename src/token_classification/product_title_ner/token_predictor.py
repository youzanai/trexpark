from typing import Optional

from datasets import load_dataset, Features, Value
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, HfArgumentParser, \
    DataCollatorForTokenClassification, BertTokenizerFast, DataCollatorWithPadding
import numpy as np
from tqdm import tqdm
from transformers.trainer_pt_utils import IterableDatasetShard
import json

from bert_crf import BertCRF
from dataclasses import dataclass, field
import torch


@dataclass
class TokenLabelPredictArguments:
    predict_data_file: str = field(metadata={"help": "The input file path."})
    label_file: str = field(metadata={"help": "Json file for label definition."})
    model_name: str = field(
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    format: str = field(default="csv", metadata={"help": "Format of the input file"})
    split: Optional[str] = field(default=None, metadata={"help": "Split suffix of the train split. Such as [:2000]"})
    streaming: bool = field(default=False, metadata={
        "help": "Whether loading the file with streaming flavor. "
                "Enable streaming when the input file is too big to fit in memory."})
    feature_name: str = field(default="text",
                              metadata={"help": "For csv dataset, define which field stores the text to predict."})
    num_workers: int = field(default=0, metadata={"help": "Number of workers of the dataloader"})
    batch_size: int = field(default=256, metadata={"help": "Batch size of the dataloader"})
    use_cuda: bool = field(default=True, metadata={"help": "Whether use cuda device to calculate predictions."})


class NerTag:
    def __init__(self, offset, label):
        self.offset: (int, int) = offset
        self.label: str = label
        self.text = ""

    def __repr__(self):
        return f"{self.text}:{self.label}:({self.offset[0]}-{self.offset[1]})"

    def __add__(self, other):
        offset = (min(self.offset[0], other.offset[0]), max(self.offset[1], other.offset[1]))
        label = self.label
        return NerTag(offset, label)


class TokenClassifier:
    def __init__(self, args):
        self.args = args
        self.model = self.load_model()
        self.labels = self.load_labels()

    def tokenize(self, tokenizer):
        def _tokenize(sample):
            text_arr = [text.lower() for text in sample[self.args.feature_name]]
            tokenized_inputs = tokenizer(text_arr, truncation=True, padding='max_length', is_split_into_words=False)
            tokens = [tokenizer.convert_ids_to_tokens(input_id) for input_id in tokenized_inputs["input_ids"]]
            offsets = [encoding.offsets for encoding in tokenized_inputs.encodings]
            # tokenized_inputs['input'] = tokenized_inputs['input_ids']
            tokenized_inputs['tokens'] = tokens
            tokenized_inputs['offsets'] = offsets
            tokenized_inputs[self.args.feature_name] = sample[self.args.feature_name]
            return tokenized_inputs

        return _tokenize

    def tensor_collate(self, batch):
        batch_dict = {}
        for record in batch:
            for key in record.keys():
                feature = batch_dict.setdefault(key, [])
                feature.append(record[key])
        batch_dict["input_ids"] = torch.IntTensor(batch_dict["input_ids"])
        if self.args.use_cuda:
            batch_dict["input_ids"] = batch_dict["input_ids"].cuda()
        return batch_dict

    def load_data(self):
        if self.args.split is not None:
            split = f"train{self.args.split}"
        else:
            split = "train"

        dataset = load_dataset(
            self.args.format, data_files={"train": self.args.predict_data_file}, streaming=self.args.streaming,
            split=split, features=Features({self.args.feature_name: Value('string')}))

        tokenizer = BertTokenizerFast.from_pretrained(self.args.tokenizer_name, truncation=True)
        self.dataset = dataset.map(self.tokenize(tokenizer), batched=True)
        if self.args.streaming:
            self.dataset = IterableDatasetShard(self.dataset)
        data_loader = DataLoader(self.dataset, collate_fn=self.tensor_collate, num_workers=self.args.num_workers,
                                 shuffle=False,
                                 batch_size=self.args.batch_size)
        self.data_loader = data_loader
        return data_loader

    def load_labels(self):
        with open(self.args.label_file, encoding="utf-8") as fp:
            label_dict = json.load(fp)
            return dict([(v, k[2:]) for k, v in label_dict.items()])

    def load_model(self):
        model = BertCRF.from_pretrained(self.args.model_name)
        if args.use_cuda:
            model = model.cuda()
        return model

    def predict(self):
        self.model.eval()
        results = []
        for step, batch in tqdm(enumerate(self.data_loader)):
            with torch.no_grad():
                predictions = self.model(batch["input_ids"]).numpy()
            for text, tokens, offsets, preds in zip(batch[self.args.feature_name], batch["tokens"], batch["offsets"],
                                                    predictions):
                tags = self.decode(text, tokens, offsets, preds)
                results.append([tags, text])
        return results

    def _get_label(self, label_id):
        return self.labels[label_id]

    def _match_whole_word(self, pos, tokens, offsets, text):
        word_start, word_end = offsets[pos]
        i = pos
        while True:
            start, end = offsets[i]
            if not tokens[i].startswith("##"):
                word_start = start
                break
            else:
                i -= 1
        i = pos + 1
        while i < len(tokens) and tokens[i].startswith("##"):
            start, end = offsets[i]
            word_end = end
            i += 1

        return i, word_start, word_end, text[word_start: word_end]

    def decode(self, text, tokens, offsets, preds):
        i = 0
        ner_tags = []
        prev_label = None
        ner_tag = None

        while i < len(tokens) and tokens[i] != '[PAD]':
            if preds[i] > 0:
                cur_label = self._get_label(preds[i])
                i, word_start, word_end, word = self._match_whole_word(i, tokens, offsets, text)
                cur_tag = NerTag((word_start, word_end), cur_label)
                if cur_label == prev_label:
                    ner_tag += cur_tag
                else:
                    if ner_tag is not None:
                        ner_tag.text = text[ner_tag.offset[0]: ner_tag.offset[1]]
                        ner_tags.append(ner_tag)
                    prev_label = cur_label
                    ner_tag = cur_tag
            else:
                if ner_tag is not None:
                    ner_tag.text = text[ner_tag.offset[0]: ner_tag.offset[1]]
                    ner_tags.append(ner_tag)
                prev_label = None
                ner_tag = None
                i += 1
        if ner_tag is not None:
            ner_tag.text = text[ner_tag.offset[0]: ner_tag.offset[1]]
            ner_tags.append(ner_tag)
        return ner_tags


def parse_args():
    parser = HfArgumentParser(TokenLabelPredictArguments)
    _args, = parser.parse_args_into_dataclasses()
    if _args.tokenizer_name is None:
        _args.tokenizer_name = _args.model_name
    return _args


if __name__ == '__main__':
    args = parse_args()
    tc = TokenClassifier(args)
    tc.load_model()
    data_loader = tc.load_data()
    results = tc.predict()
    with open("result.csv", "wt", encoding="utf-8") as fp:
        for result in results:
            tags = '|'.join([str(t) for t in result[0]])
            text = result[1]
            fp.write(f"{tags},{text}\n")
