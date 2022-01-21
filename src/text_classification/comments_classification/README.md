# 评价正负面评价分类训练

在hugging-face预训练模型库中的基于电商评价的预训练模型“youzanai/bert-product-comment-chinese”基础上，进行的正负面评价分类训练示例

----

## 训练数据
训练数据样例保存在sample_data文件夹下，由训练数据train.csv，验证数据eval.csv两个由Tab键分割的CSV文件组成。
其中每个文件头有两列，第一列为text文本数据，第二列label由0，1两种label组成，0表示负面评价，1表示正面评价

## 训练过程
在环境都安装完整的前提下，执行run_cat_classification.sh脚本就可以了进行训练。包括batch_size，epoch，训练数据目录等配置都可以在这个脚本中进行修改
