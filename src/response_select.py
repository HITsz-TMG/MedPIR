from abc import ABC
import torch
from src.utils.gpt_modeling import TransformerEncoder, TransformerDecoderLM
from torch import nn
from src.utils.hugging_face_bert import BertModel
from src.utils.hugging_face_gpt import GPT2Model
from src.utils.hugging_face_gpt import Attention
from transformers import BertConfig, GPT2Config
import torch.nn.functional as F
import json
from typing import Any, Tuple
from torch.nn.utils.rnn import pad_sequence


class ResponseSelector(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()

        #  cls (history) sep (response) sep
        #   0      0      0       1      1
        # linear
        # sigmoid -> probability
        self.config = dict(config)
        self.encoder_config = BertConfig.from_json_file(config["pretrained_encoder_config_path"])
        self.encoder = BertModel(
            config=self.encoder_config,
            add_pooling_layer=False,
            decoder_like_gpt=False
        )
        res = self.encoder.load_state_dict(self._load_modified_state_dict(), strict=False)
        assert res.missing_keys == ['embeddings.position_ids'] and res.unexpected_keys == []

        self.scoring = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

    def _load_modified_state_dict(self):
        state_dict = torch.load(self.config["pretrained_state_dict_path"], map_location='cpu')
        new_state_dict = []
        for k, v in state_dict.items():
            if k.startswith("bert.pooler") or k.startswith("cls"):
                continue
            elif k.startswith("bert."):
                new_state_dict.append((k[5:], v))
            else:
                new_state_dict.append((k, v))
        return dict(new_state_dict)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        # batch_size * sentence_num *  seq_len

        input_ids = input_ids.transpose(1, 0)
        token_type_ids = token_type_ids.transpose(1, 0)
        attention_mask = attention_mask.transpose(1, 0)

        # sentence_num * batch_size *  seq_len

        all_score = []

        for sentence_id, (s_input_ids, s_token_type_ids, s_attention_mask) in enumerate(
                zip(input_ids, token_type_ids, attention_mask)):
            encoder_outputs = self.encoder(
                input_ids=s_input_ids,
                token_type_ids=s_token_type_ids,
                attention_mask=s_attention_mask,
            )
            sequence_output = encoder_outputs[0]
            cls_hidden = sequence_output[:, 0, :]
            score = self.scoring(cls_hidden)
            all_score.append(score)

        all_score = torch.stack(all_score).transpose(1, 0).squeeze(dim=-1)  # batch_size, sentence_num

        if labels is not None:
            weight = torch.ones_like(labels) + 2 * labels
            loss = F.binary_cross_entropy(all_score, labels.to(all_score.dtype), weight=weight.float())
        else:
            loss = None

        return all_score, loss
