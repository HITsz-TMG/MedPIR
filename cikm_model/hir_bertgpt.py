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


class HireFusionModel(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        entity_attention_type = config.get("entity_attention_model", "EntityAttention")
        self.entity_attention_type = entity_attention_type
        encoder_json = json.load(open(config["pretrained_encoder_config_path"], 'r'))
        # if config['expand_token_type_embed']:
        #     encoder_json['type_vocab_size'] = 3
        # self.encoder_config = BertConfig.from_json_file(config["pretrained_encoder_config_path"])
        self.encoder_config = BertConfig.from_dict(encoder_json)

        if self.config.get('sentence_add_entity', False) is True:
            self.encoder_config.type_vocab_size = 3
        self.encoder = BertModel(config=self.encoder_config,
                                 add_pooling_layer=False,
                                 decoder_like_gpt=False,
                                 use_entity_appendix=config.get('entity_predict', False)
                                 )

        self.decoder_config = BertConfig.from_json_file(config["pretrained_decoder_config_path"])
        self.decoder = BertModel(config=self.decoder_config,
                                 add_pooling_layer=False,
                                 decoder_like_gpt=True,
                                 use_hir_attention=True,
                                 )
        # entity_attention=True if entity_attention_type == 'EntityAttention' else False)
        self.lm_linear = nn.Linear(self.encoder_config.hidden_size, self.encoder_config.vocab_size,
                                   # bias=True)
                                   bias=False)

        self.load_bert_gpt_state_dict()

    def load_bert_gpt_state_dict(self):
        ignore_keys = [
            "encoder.embeddings.position_ids",
            "decoder.embeddings.position_ids",
            "lm_linear.bias",

            'encoder.embeddings.position_ids',
            'encoder.entity_appendix_embeddings.word_embeddings.weight',
            'encoder.entity_appendix_embeddings.position_embeddings.weight',
            'encoder.entity_appendix_embeddings.token_type_embeddings.weight',
            'encoder.entity_appendix_embeddings.LayerNorm.weight',
            'encoder.entity_appendix_embeddings.LayerNorm.bias',
        ]

        for i in range(12):
            ignore_keys = ignore_keys + [
                "decoder.encoder.layer.{}.crossattention.self.query.weight".format(i),
                "decoder.encoder.layer.{}.crossattention.self.query.bias".format(i),
                "decoder.encoder.layer.{}.crossattention.self.key.weight".format(i),
                "decoder.encoder.layer.{}.crossattention.self.key.bias".format(i),
                "decoder.encoder.layer.{}.crossattention.self.value.weight".format(i),
                "decoder.encoder.layer.{}.crossattention.self.value.bias".format(i),
                "decoder.encoder.layer.{}.crossattention.output.dense.weight".format(i),
                "decoder.encoder.layer.{}.crossattention.output.dense.bias".format(i),
                "decoder.encoder.layer.{}.crossattention.output.LayerNorm.weight".format(i),
                "decoder.encoder.layer.{}.crossattention.output.LayerNorm.bias".format(i),
            ]
        state_dict = torch.load(self.config['bertgpt_state_dict'], map_location='cpu')
        new_state_dict = []
        for k, v in state_dict.items():
            if k.startswith("decoder.transformer"):
                new_state_dict.append((k.replace(".transformer", ""), v))
            elif k == "decoder.projection.weight":
                new_state_dict.append(("lm_linear.weight", v))
            else:
                new_state_dict.append((k, v))
        # decoder.projection.weight
        # lm_linear.weight
        new_state_dict = dict(new_state_dict)

        if self.config.get('expand_token_type_embed', False) is True:
            print("token type num embedding expand to 3")
            token_type_embed = new_state_dict['encoder.embeddings.token_type_embeddings.weight']
            token_type_embed = torch.cat(
                [token_type_embed, torch.normal(size=(1, 768), mean=0, std=0.02)], dim=0
            )
            new_state_dict['encoder.embeddings.token_type_embeddings.weight'] = token_type_embed
            print(new_state_dict['encoder.embeddings.token_type_embeddings.weight'].shape)

        load_result = self.load_state_dict(new_state_dict, strict=False)
        # filter_missing_keys = [i for i in load_result.missing_keys if "entity_fuse_layer" not in i]
        # assert len(filter_missing_keys) <= len(ignore_keys)
        # for i in filter_missing_keys:
        #     assert i in ignore_keys
        # assert len(load_result.unexpected_keys) == 0

    def forward(self, history_ids=None, history_mask=None, history_speaker=None,
                response_mask=None, response_ids=None,
                entity=None, entity_mask=None, summary=None, summary_mask=None
                ):

        encoded_history, past_key_values, history_mask, entity_loss, entity_mem, summary_mem = self.encode_step(
            history_ids=history_ids,
            history_mask=history_mask,
            token_type_ids=history_speaker,
            entity=entity,
            entity_mask=entity_mask,
            summary=summary,
            summary_mask=summary_mask
        )

        logits = self.decode_step(
            response_ids=response_ids,
            response_mask=response_mask,
            history_mask=history_mask,
            past_key_values=past_key_values,
            past_key_values_len=0,

            entity_mem=entity_mem,
            summary_mem=summary_mem,
            entity_mask=entity_mask,
            summary_mask=summary_mask
        )

        return logits, entity_loss

    def encode_entity_and_summary(self, entity, summary, entity_mask, summary_mask):
        entity = self.encoder(entity, entity_mask)
        summary = self.encoder(summary, summary_mask)
        entity = entity[0]
        summary = summary[0]
        return entity, summary

    def encode_step(
            self,
            history_ids=None,
            history_mask=None,
            token_type_ids=None,
            entity=None, entity_mask=None, summary=None, summary_mask=None
    ):

        encoder_output = self.encoder(
            history_ids, history_mask, token_type_ids=token_type_ids, use_cache=True,
        )
        encoded_history = encoder_output.last_hidden_state
        past_key_values = encoder_output.past_key_values

        history_mask = encoder_output.pooler_output

        entity_mem, summary_mem = self.encode_entity_and_summary(entity, summary, entity_mask, summary_mask)

        return encoded_history, past_key_values, history_mask, None, entity_mem, summary_mem

    def decode_step(
            self, response_ids=None, response_mask=None, history_mask=None,
            past_key_values=None, past_key_values_len=0,
            entity_mem=None, summary_mem=None, entity_mask=None, summary_mask=None
    ):

        mask = torch.cat([history_mask, response_mask], dim=1)

        decoder_output = self.decoder(
            response_ids,
            mask,
            past_key_values=past_key_values,
            past_key_values_len=past_key_values_len,
            entity_mem=entity_mem, entity_mask=entity_mask,
            summary_mem=summary_mem, summary_mask=summary_mask
        )
        hidden_state = decoder_output.last_hidden_state
        logits = self.lm_linear(hidden_state)

        return logits
