"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


from .base_model import BaseModel
from .Qformer import BertConfig, BertLMHeadModel
from .clip_vit import create_clip_vit_L
from transformers import BertTokenizer


class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls):
        # tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese")
        tokenizer = BertTokenizer(vocab_file="clip_model/vocab.txt")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    @classmethod
    def init_Qformer(cls, bert_model, num_query_token, vision_width):
        # encoder_config = BertConfig.from_pretrained(bert_model)
        encoder_config = BertConfig.from_pretrained("clip_model/")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        # Qformer = BertLMHeadModel.from_pretrained(bert_model, config=encoder_config)
        Qformer = BertLMHeadModel(encoder_config)

        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @classmethod
    def init_vision_encoder(
        cls, vit_name,  img_size, drop_path_rate, use_grad_checkpoint, precision):
        assert vit_name in [
            "eva_clip_g",
            "clip_L",
        ], "vit model must be eva_clip_g or clip_L"
        # if vit_name == "eva_clip_g":
            # visual_encoder = create_eva_vit_g(
            #     img_size, drop_path_rate, use_grad_checkpoint, precision
            # )
        # if vit_name == "clip_L":
        visual_encoder = create_clip_vit_L(img_size,  use_grad_checkpoint, precision)
        ln_vision = LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision

    def load_from_pretrained(self, filename):

        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

