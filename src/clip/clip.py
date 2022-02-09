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



import torch
import torch.nn as nn
from transformers import BertConfig, CLIPVisionConfig, CLIPVisionModel
from transformers import BertTokenizer, CLIPFeatureExtractor, BertModel
from transformers.tokenization_utils_base import BatchEncoding
from transformers.models.clip.modeling_clip import clip_loss, CLIPOutput
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging 
import copy


logger = logging.get_logger(__name__)


class ClipChineseConfig(PretrainedConfig):
    model_type = "clip_chinese_model"
    is_composition = True

    def __init__(
        self,
        text_config_dict=None,
        vision_config_dict=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        **kwargs
    ):
        super().__init__(text_config_dict=text_config_dict, vision_config_dict=vision_config_dict, **kwargs)

        if text_config_dict is None:
            text_config_dict = {}
            logger.info("text_config_dict is None. Initializing the CLIPTextConfig with default values.")

        if vision_config_dict is None:
            vision_config_dict = {}
            logger.info("vision_config_dict is None. initializing the CLIPVisionConfig with default values.")

        self.text_config = BertConfig(**text_config_dict)
        self.vision_config = CLIPVisionConfig(**vision_config_dict)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(cls, text_config: BertConfig, vision_config: CLIPVisionConfig, **kwargs):
        r"""
        Instantiate a [`CLIPConfig`] (or a derived class) from clip text model configuration and
        clip vision model configuration.

        Returns:
            [`CLIPConfig`]: An instance of a configuration object
        """

        return cls(text_config_dict=text_config.to_dict(), vision_config_dict=vision_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default
        [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["text_config"] = self.text_config.to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


class ClipProcesserChinese:
    def __init__(self, feature_extractor, tokenizer) -> None:
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.current_processor = self.feature_extractor

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:

            encoding = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
            # if encoding.get('token_type_ids'):
            #     del encoding['token_type_ids']

        if images is not None:
            image_features = self.feature_extractor(images, return_tensors=return_tensors, **kwargs)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features.pixel_values
            return encoding
        elif text is not None:
            return encoding
        else:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizer's
        [`~PreTrainedTokenizer.batch_decode`]. Please refer to the docstring of this method for more
        information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizer's [`~PreTrainedTokenizer.decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)


    def save_pretrained(self, save_directory):
        self.feature_extractor.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        feature_extractor = CLIPFeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)


class ClipChineseModel(PreTrainedModel):
    config_class = ClipChineseConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    base_model_prefix = 'clip'

    def __init__(self, config: ClipChineseConfig, 
    pretrained_text_name=None, pretrained_vision_name=None):
        super().__init__(config)
        self.text_model_name = pretrained_text_name
        self.vision_model_name = pretrained_vision_name
        self.config = config
        text_config = self.config.text_config
        vision_config = self.config.vision_config

        self.projection_dim = self.config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size
        
        if pretrained_text_name is not None:
            self.text_model = BertModel.from_pretrained(self.text_model_name)
        else:
            self.text_model = BertModel(text_config)
     
        if pretrained_vision_name is not None:
            self.vision_model = CLIPVisionModel.from_pretrained(self.vision_model_name)
        else:
            self.vision_model = CLIPVisionModel(vision_config)
        
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * self.config.logit_scale_init_value)
    
    def get_text_features(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings
            obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].

        Examples:

        ``` TODO: add usage doc
        ```"""
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features

    def get_image_features(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings
            obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].

        Examples:
        git clone https://github.com/youzanai/trexpark.git
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from src.clip.clip import ClipChineseModel, CLIPProcessorChinese

        >>> model = ClipChineseModel.from_pretrained("youzanai/clip-product-title-chinese")
        >>> processor = CLIPProcessorChinese.from_pretrained("youzanai/clip-product-title-chinese")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features

    
    def forward(self, input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        return_loss=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True, **kwargs):
        
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class CLIPInfer:
    def __init__(self, pretrained_model_name):
        self.model = ClipChineseModel.from_pretrained(pretrained_model_name)
        self.clip_processor = ClipProcesserChinese.from_pretrained(pretrained_model_name)

    def __call__(self, imgs, texts):
        f = self.clip_processor(texts, imgs, return_tensors='pt', truncation=True, padding=True)
        del f['token_type_ids']
        with torch.no_grad():
            out = self.model(**f)
        logits_per_image, logits_per_text = out['logits_per_image'], out['logits_per_text']
        return logits_per_image.softmax(dim=-1).cpu().detach().numpy()


if __name__ == '__main__':
    # clip_processor = ClipProcesserChinese.from_pretrained('youzanai/clip-product-title-chinese')
    # model = ClipChineseModel.from_pretrained('youzanai/clip-product-title-chinese')
    import requests
    from PIL import Image
    url = 'http://img.yzcdn.cn/upload_files/2015/04/21/0140dac4657f874f2acff9294b28088c.jpg'
    clip_infer = CLIPInfer('youzanai/clip-product-title-chinese')
    img = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    imgs = [img]
    texts = ['运动鞋', '红色连衣裙', '黑色连衣裙', '大衣', '文具']
    out = clip_infer(imgs, texts)
    print(out)