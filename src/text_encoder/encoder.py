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
    通过有赞的商品标题预训练bert模型进行商品标题相似度计算的例子。sample_data/goods_title_mlm_eval.txt文件中包含了5140个用于预训练效果评估的商品标题，通过计算相似度，可以找到商品的相似商品。
"""

from transformers import (AutoModel, AutoTokenizer)
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


def encode(batch, tokenizer, model):
    """
        对文本进行编码，并返回编码后的tensor向量

        Parameters:
            batch - 需要进行编码的文本，比如一个字符串的list
            tokenizer - 模型对应的tokenizer，把文本转化为ID序列
            model - 用来进行编码的预训练模型

        Returns:
            通过预训练模型编码后的tensor向量
    """

    # 设置模型为推理模式
    model.eval()

    # 不计算梯度
    with torch.no_grad():
        # 通过tokenizer将文本转化为input_ids
        pt_batch = tokenizer(batch, padding=True, return_tensors="pt")
        # 模型推理
        pt_outputs = model(
            **pt_batch, output_hidden_states=False, output_attentions=False)
        # 得到pooling后的文本表征向量
        outputs = pt_outputs.pooler_output.numpy()
        # 对表征向量进行归一
        outputs = normalize(outputs, axis=0)
        return outputs


if __name__ == "__main__":
    #加载预训练的tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(
        "youzanai/bert-product-title-chinese")
    pt_model = AutoModel.from_pretrained("youzanai/bert-product-title-chinese")
    
    # 载入语料
    batch = []
    with open("sample_data/goods_title_mlm_eval.txt") as fp:
        for line in fp:
            batch.append(line)
    # 对语料进行编码
    output = encode(batch, tokenizer, pt_model)
    
    # 对商品编码构建KD树，用来进行相似度计算
    tree = KDTree(output)
    batch = np.array(batch)
    result = []
    for i, x in tqdm(enumerate(output)):
        dist, ind = tree.query(output[i:i+1], k=5)
        result.append(batch[ind][0].tolist())
    
    # 保存相似度计算结果
    result = pd.DataFrame(result)
    if not os.path.exists("result"):
        os.mkdir("result")
    result.to_csv("result/sim_products_result.csv")
