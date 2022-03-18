# T'rex Park（霸王龙公园）

Trexpark项目由有赞数据智能团队开源，是国内首个基于电商大数据训练的开源NLP和图像项目。我们预期将逐步开放基于商品标题，评论，客服对话等NLP语聊，以及商品主图，品牌logo等进行预训练的NLP和图像模型。

----

### 为什么是霸王龙？

![霸王龙](images/trex_white.png)

霸王龙是有赞的吉祥物。呃，准确的说这不是个吉祥物，而是有赞人自我鞭策的精神图腾。早期我们的网站经常崩溃，导致浏览器会显示一个霸王龙的图案，提示页面崩溃了。于是我们就把霸王龙作为我们的吉祥物，让大家时刻警惕故障和缺陷。

----

### 为什么要开源模型？

和平台电商不同，有赞是一家商家服务公司，我们的使命是帮助每一位重视产品和服务的商家成功。因此我们放弃了通过开放接口提供服务的方式，直接把底层能力开放出来，提供给需要的商家和中小型电商企业，帮助他们在有赞的数据沉淀基础上，快速构建自己的机器学习应用。

----
### 为什么要做领域预训练模型？

目前各个开源大模型往往基于通用语料训练，而通用语料的语言模型用于特定领域的机器学习任务，往往效果不佳，或者需要对预训练模型部分进行finetune。我们的实践发现，基于电商数据finetune以后的预训练模型，能更好的学习到领域知识，并且在多项任务中，无须额外训练，或者仅仅对模型的预测部分进行训练就可以达到很好的效果。

我们基于电商领域语料训练的预训练模型非常适合小样本的机器学习任务，用于解决中小电商企业和商家的fewshot难题。以商品标题分类为例，每个类目只需要100个样本，就能得到很好的分类效果，具体例子可以看[这里](https://github.com/youzanai/trexpark/blob/main/src/text_classification/title_classification/title_classification.py)。

我们的模型已经在HuggingFace的model hub上发布，想要使用我们的模型，只需要几行代码
```Python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("youzanai/bert-product-title-chinese")
model = AutoModel.from_pretrained("youzanai/bert-product-title-chinese")
```
模型加载后，我们就可以执行简单的encoder任务了
```Python
batch = tokenizer(["青蒿精油手工皂", "超级飞侠乐迪太空车"])
outputs = model(**batch)
print(outputs.logits)
```

项目的[src](https://github.com/youzanai/trexpark/tree/main/src)目录中有完整的代码和测试用的数据，可以直接运行浏览效果。

----

### 已开发模型列表
商品标题语言模型：[youzanai/bert-product-title-chinese](https://huggingface.co/youzanai/bert-product-title-chinese)

商品图片标题CLIP模型：[youzanai/clip-product-title-chinese](https://huggingface.co/youzanai/clip-product-title-chinese)

商品评价语言模型：[youzanai/bert-product-comment-chinese](https://huggingface.co/youzanai/bert-product-comment-chinese)

收货地址语言模型：[youzanai/bert-shipping-address-chinese](https://huggingface.co/youzanai/bert-shipping-address-chinese)

客服聊天用户问题语言模型：[youzanai/bert-customer-message-chinese](https://huggingface.co/youzanai/bert-customer-message-chinese)

### 文档和帮助

详细的使用文档我们还在编写中，大家可以先参考[src](https://github.com/youzanai/trexpark/tree/main/src)目录中的示例代码。为了让代码更容易理解，我们已经尽可能的对代码进行了精简。T'rex Park底层使用了HuggingFace的Transformer框架，关于Transformer的文档可以看[这里](https://huggingface.co/docs/transformers/quicktour)

----
