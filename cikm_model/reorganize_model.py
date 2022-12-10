from abc import ABC
import torch
from torch import nn
from src.utils.hugging_face_bert import BertModel
from transformers import BertConfig, GPT2Config
from copy import deepcopy


class Reorganize(nn.Module, ABC):
    def __init__(self, config):
        super(Reorganize, self).__init__()
        self.config = dict(config)
        entity_attention_type = config.get("entity_attention_model", "EntityAttention")
        self.entity_attention_type = entity_attention_type
        self.encoder_config = BertConfig.from_json_file(config["pretrained_encoder_config_path"])
        if self.config.get('sentence_add_entity', False) is True:
            self.encoder_config.type_vocab_size = 3
        self.encoder = BertModel(
            config=self.encoder_config,
            add_pooling_layer=False,
            decoder_like_gpt=False,
        )

        if config.get("use_references_or_history_crossattention"):
            self.use_encoder_for_crossattention = True
        else:
            self.use_encoder_for_crossattention = False

        self.decoder_config = BertConfig.from_json_file(config["pretrained_decoder_config_path"])
        self.decoder = BertModel(
            config=self.decoder_config,
            add_pooling_layer=False,
            decoder_like_gpt=True,

            use_references_crossattention=self.use_encoder_for_crossattention,
        )

        self.lm_linear = nn.Linear(self.encoder_config.hidden_size, self.encoder_config.vocab_size, bias=False)

        self.load_bert_gpt_state_dict()

        if self.use_encoder_for_crossattention:
            self.encoder_for_crossattention = BertModel(
                config=self.encoder_config,
                add_pooling_layer=False,
                decoder_like_gpt=False,
            )
            self.encoder_for_crossattention.load_state_dict(deepcopy(self.encoder.state_dict()))

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
        crossattention_keys = []

        for i in range(12):
            crossattention_keys += [
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
        ignore_keys = ignore_keys + crossattention_keys

        ref_crossattention_keys = []
        if self.use_encoder_for_crossattention:
            ref_crossattention_keys = [i.replace("crossattention", "references_crossattention") for i in
                                       crossattention_keys]
        ignore_keys = ignore_keys + ref_crossattention_keys

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

        if self.config.get('sentence_add_entity', False) is True:
            print("token type num embedding expand to 3")
            token_type_embed = new_state_dict['encoder.embeddings.token_type_embeddings.weight']
            token_type_embed = torch.cat(
                [token_type_embed, torch.normal(size=(1, 768), mean=0, std=0.02)], dim=0
            )
            new_state_dict['encoder.embeddings.token_type_embeddings.weight'] = token_type_embed
            print(new_state_dict['encoder.embeddings.token_type_embeddings.weight'].shape)

        load_result = self.load_state_dict(new_state_dict, strict=False)
        filter_missing_keys = [i for i in load_result.missing_keys if "entity_fuse_layer" not in i]
        assert len(filter_missing_keys) <= len(ignore_keys)
        for i in filter_missing_keys:
            assert i in ignore_keys
        assert len(load_result.unexpected_keys) == 0

    def forward(
            self,
            # 历史不一定是历史 历史有可能在input for crossattention
            history_ids=None,
            history_mask=None,
            history_speaker=None,
            response_mask=None,
            response_ids=None,
            # reorganize
            input_for_crossattention=None,
            crossattention_mask=None,
    ):

        encoded_history, past_key_values, history_mask, outer_encoded_hidden_states, crossattention_mask = self.encode_step(
            history_ids=history_ids,
            history_mask=history_mask,
            token_type_ids=history_speaker,
            # reorganize
            input_for_crossattention=input_for_crossattention,
            crossattention_mask=crossattention_mask
        )

        logits = self.decode_step(
            response_ids=response_ids,
            response_mask=response_mask,
            history_mask=history_mask,
            past_key_values=past_key_values,
            past_key_values_len=0,

            outer_encoded_hidden_states=outer_encoded_hidden_states,
            outer_attention_mask=crossattention_mask
        )

        return logits,

    def encode_step(
            self,
            history_ids=None,
            history_mask=None,
            token_type_ids=None,
            input_for_crossattention=None,
            crossattention_mask=None
    ):

        if self.use_encoder_for_crossattention:
            outer_encoder_outputs = self.encoder_for_crossattention(
                input_for_crossattention,
                crossattention_mask
            )
            outer_encoded_hidden_states = outer_encoder_outputs[0]
        else:
            outer_encoded_hidden_states = None

        encoder_output = self.encoder(
            history_ids,
            history_mask,
            token_type_ids=token_type_ids,
            use_cache=True,
            entity_appendix=None,
            entity_appendix_mask=None,

        )
        encoded_history = encoder_output.last_hidden_state
        past_key_values = encoder_output.past_key_values

        history_mask = encoder_output.pooler_output

        return encoded_history, past_key_values, history_mask, outer_encoded_hidden_states, crossattention_mask

    def decode_step(
            self,
            response_ids=None,
            response_mask=None,
            history_mask=None,
            past_key_values=None,
            past_key_values_len=0,
            # reorganize
            outer_encoded_hidden_states=None,
            outer_attention_mask=None
    ):

        mask = torch.cat([history_mask, response_mask], dim=1)

        decoder_output = self.decoder(
            response_ids,
            mask,
            past_key_values=past_key_values,
            past_key_values_len=past_key_values_len,

            # reorganize
            outer_encoded_hidden_states=outer_encoded_hidden_states,
            outer_attention_mask=outer_attention_mask
        )
        hidden_state = decoder_output.last_hidden_state
        logits = self.lm_linear(hidden_state)

        return logits
