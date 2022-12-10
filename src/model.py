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


class NextEntityClassifier(nn.Module, ABC):
    def __init__(self, input_size, class_num, dropout=0.1):
        super(NextEntityClassifier, self).__init__()
        self.input_size = input_size
        self.class_num = class_num,
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, input_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(input_size, class_num),
            nn.Sigmoid()
        )

    def forward(self, features):
        return self.classifier(features)


class NextEntityPredict(nn.Module, ABC):
    def __init__(self, config, encoder=None):
        super().__init__()
        self.config = config

        if config.get('pcl_encoder_predict_entity', False) or encoder is None:
            print("use PCL BERT -> next entity predict")
            self.encoder_config = BertConfig.from_json_file(config["pretrained_encoder_config_path"])
            self.encoder = BertModel(config=self.encoder_config,
                                     add_pooling_layer=False,
                                     decoder_like_gpt=False)
            res = self.encoder.load_state_dict(self._load_modified_state_dict(), strict=False)
            assert res.missing_keys == ['embeddings.position_ids'] and res.unexpected_keys == []
            self.usl_pcl_encoder = True
        else:
            self.usl_pcl_encoder = False
            self.encoder = encoder

        self.eid2entity = config['entity']

        self.entity_type_num = config['entity_type_num']

        self.symptom_classifier = NextEntityClassifier(
            768, self.entity_type_num["Symptom"]
        )
        self.medicine_classifier = NextEntityClassifier(
            768, self.entity_type_num["Medicine"]
        )
        self.test_classifier = NextEntityClassifier(
            768, self.entity_type_num["Test"]
        )
        self.attribute_classifier = NextEntityClassifier(
            768, self.entity_type_num["Attribute"]
        )
        self.disease_classifier = NextEntityClassifier(
            768, self.entity_type_num["Disease"]
        )

        self.classifier = [
            self.symptom_classifier,
            self.medicine_classifier,
            self.test_classifier,
            self.attribute_classifier,
            self.disease_classifier
        ]

        self.token2idx = dict()
        self.idx2token = dict()
        with open(config['vocab_path'], 'r', encoding='utf-8') as reader:
            for idx, token in enumerate(list(reader.readlines())):
                token = token.strip()
                self.token2idx[token] = idx
                self.idx2token[idx] = token

        self.all_linked_inputs = []
        self.unk_idx = self.token2idx['[UNK]']
        self.sep_idx = self.token2idx['[SEP]']
        self.cls_idx = self.token2idx['[CLS]']
        self.pad_idx = self.token2idx['[PAD]']
        self.entity_sep_id = self.token2idx['[entities]']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    def translate_prob_to_appendix(self, probs):
        # 按照概率大小排序
        values, indices = probs.sort(descending=True)
        entities = []
        for v, i in zip(values, indices):
            cur_entities = []
            for vv, ii in zip(v, i.tolist()):
                if vv > 0.5:
                    cur_entities.append(self.eid2entity[ii])
                    if len(cur_entities) >= 10:
                        break
                else:
                    break
            entities.append(cur_entities)
        print("")
        print(entities)
        appendix = []

        for es in entities:
            cur_appendix = [self.entity_sep_id]
            if len(es) > 0:
                for e in es:
                    cur_appendix.extend(
                        [self.token2idx[t.lower()] for t in e]
                    )
                    cur_appendix.append(self.entity_sep_id)
            appendix.append(torch.tensor(cur_appendix).to(self.device))

        appendix = pad_sequence(appendix, batch_first=True, padding_value=self.pad_idx)
        appendix_mask = (appendix != self.pad_idx).long()
        return appendix, appendix_mask

    def create_entity_query(self, encoded_history, history_mask):
        history_length = history_mask.sum(dim=1)
        mask_encoded_history = encoded_history * history_mask.unsqueeze(dim=-1)
        avg_pooled_history = mask_encoded_history.sum(dim=1) / history_length.unsqueeze(dim=-1)
        return avg_pooled_history

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, label=None):
        encoder_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        encoded_history = encoder_output[0]
        # entity_query = sequence_hidden_states[:, 0]
        entity_query = self.create_entity_query(encoded_history, attention_mask)
        five_topic_probs = [classify(entity_query) for classify in self.classifier]
        topic_probs = torch.cat(five_topic_probs, -1)

        if label is not None:
            topic_weight = torch.ones_like(label) * 0.5 + 3.5 * label
            loss = F.binary_cross_entropy(topic_probs, label.to(topic_probs.dtype), topic_weight.float())
        else:
            loss = None

        return topic_probs, five_topic_probs, loss,


class BERTGPTold(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.encoder = TransformerEncoder()  # def forward(self, input_ids, mask, token_type_ids=None, past=None)
        self.decoder = TransformerDecoderLM()  # def forward(self, input_ids, mask, past=None, past_length=None)
        self.load_state_dict(torch.load(config['bertgpt_state_dict'], map_location='cpu'))

    def forward(self, history_ids=None, response_ids=None, history_speaker=None,
                history_mask=None, response_mask=None):
        """
            "history_ids": history_ids,
            "response_ids": response_ids,
            "history_speaker": history_speaker,
            "history_mask": history_mask,
            "response_mask": response_mask
        """
        encoded_history, past = self.encoder(history_ids, history_mask, token_type_ids=history_speaker)
        mask = torch.cat([history_mask, response_mask], dim=1)
        logits, _ = self.decoder(response_ids, mask, past=past, past_length=0)

        return logits,


class BERTGPTCrossAtt(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder_config = BertConfig.from_json_file(config["pretrained_encoder_config_path"])
        self.encoder = BertModel(config=self.encoder_config,
                                 add_pooling_layer=False,
                                 decoder_like_gpt=False)
        res = self.encoder.load_state_dict(self._load_modified_state_dict(), strict=False)
        assert res.missing_keys == ['embeddings.position_ids'] and res.unexpected_keys == []

        self.gpt2_config = GPT2Config.from_json_file(config['gpt2_config_path'])
        self.decoder = GPT2Model(config=self.gpt2_config)

        self.lm_linear = nn.Linear(self.encoder_config.hidden_size, self.encoder_config.vocab_size)

    def _load_modified_state_dict(self):
        state_dict = torch.load(self.config["pretrained_state_dict_path"])
        new_state_dict = []
        for k, v in state_dict.items():
            if k.startswith("bert.pooler") or k.startswith("cls"):
                continue
            elif k.startswith("bert."):
                new_state_dict.append((k[5:], v))
            else:
                new_state_dict.append((k, v))
        return dict(new_state_dict)

    def forward(self, history_ids=None, response_ids=None, history_speaker=None,
                history_mask=None, response_mask=None):
        encoder_output = self.encoder(
            input_ids=history_ids,
            attention_mask=history_mask,
            token_type_ids=history_speaker
        )

        encoded_history = encoder_output[0]

        decoder_output = self.decoder(
            input_ids=response_ids,
            attention_mask=response_mask,
            encoder_hidden_states=encoded_history,
            encoder_attention_mask=history_mask
        )

        output = decoder_output[0]
        logits = self.lm_linear(output)

        return logits,


class BERT2BERT(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder_config = BertConfig.from_json_file(config["pretrained_encoder_config_path"])
        self.encoder = BertModel(config=self.encoder_config,
                                 add_pooling_layer=False,
                                 decoder_like_gpt=False)
        res = self.encoder.load_state_dict(self._load_modified_state_dict(), strict=False)
        assert res.missing_keys == ['embeddings.position_ids'] and res.unexpected_keys == []
        self.decoder_config = BertConfig.from_json_file(config["pretrained_decoder_config_path"])
        self.decoder = BertModel(config=self.decoder_config,
                                 add_pooling_layer=False,
                                 decoder_like_gpt=True)
        res = self.decoder.load_state_dict(self._load_modified_state_dict(), strict=False)
        res.missing_keys.remove("embeddings.position_ids")
        assert all("crossattention" in k for k in res.missing_keys) and res.unexpected_keys == []
        self.lm_linear = nn.Linear(self.encoder_config.hidden_size, self.encoder_config.vocab_size)

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

    def forward(self, history_ids=None, response_ids=None, history_speaker=None,
                history_mask=None, response_mask=None):
        encoder_output = self.encoder(
            input_ids=history_ids,
            attention_mask=history_mask,
            token_type_ids=history_speaker
        )

        encoded_history = encoder_output[0]

        decoder_output = self.decoder(
            input_ids=response_ids,
            attention_mask=response_mask,
            encoder_hidden_states=encoded_history,
            encoder_attention_mask=history_mask
        )

        output = decoder_output[0]
        logits = self.lm_linear(output)

        return logits,


class BERTGPTEntity(nn.Module, ABC):
    def __init__(self, config):
        super(BERTGPTEntity, self).__init__()
        self.config = dict(config)
        entity_attention_type = config.get("entity_attention_model", "EntityAttention")
        self.entity_attention_type = entity_attention_type
        encoder_json = json.load(open(config["pretrained_encoder_config_path"], 'r'))
        if config['expand_token_type_embed']:
            encoder_json['type_vocab_size'] = 3
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
                                 decoder_like_gpt=True, )
        # entity_attention=True if entity_attention_type == 'EntityAttention' else False)
        self.lm_linear = nn.Linear(self.encoder_config.hidden_size, self.encoder_config.vocab_size,
                                   # bias=True)
                                   bias=False)

        # self.load_bert_gpt_state_dict()

        if config['entity_predict']:
            self.next_entity_predictor = NextEntityPredict(config, encoder=self.encoder)  # 在里面判断是否共用encoder

    def init_bert_gpt_by_pcl_bert(self):
        pcl_bert = torch.load("./pretrained/PCL-MedBERT/pytorch_model.bin")
        from collections import OrderedDict

        state_dict = OrderedDict()
        for k, v in pcl_bert.items():
            if k.startswith("bert."):
                k = k[5:]
            if k in ["pooler.dense.weight", "pooler.dense.bias", "cls.predictions.bias",
                     "cls.predictions.transform.dense.weight", "cls.predictions.transform.dense.bias",
                     "cls.predictions.transform.LayerNorm.weight", "cls.predictions.transform.LayerNorm.bias",
                     "cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]:
                continue
            state_dict[k] = v

        enc_load_err = self.encoder.load_state_dict(state_dict, strict=False)
        dec_load_err = self.decoder.load_state_dict(state_dict, strict=False)


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
        filter_missing_keys = [i for i in load_result.missing_keys if "entity_fuse_layer" not in i]
        assert len(filter_missing_keys) <= len(ignore_keys)
        for i in filter_missing_keys:
            assert i in ignore_keys
        assert len(load_result.unexpected_keys) == 0

    def slide_window_encode(self, group_ids=None, group_mask=None, start_end_expand_as_batch_size=None,
                            each_sample_chunks_num=None, ):
        encode_result = []
        past_key_values = []
        device = group_ids[0].device
        group_output = []
        group_pk = [[] for _ in range(12)]
        group_pv = [[] for _ in range(12)]

        for idx in range(len(group_ids)):
            outputs = self.encoder(
                input_ids=group_ids[idx],
                attention_mask=group_mask[idx],
                use_cache=True
            )
            origin_output = outputs.last_hidden_state
            # tuple of (batch_size, num_heads, sequence_length, embed_size_per_head)
            origin_past_key_values = outputs.past_key_values

            for idx_j in range(len(start_end_expand_as_batch_size[idx])):
                start, end = start_end_expand_as_batch_size[idx][idx_j]
                select_output = origin_output[idx_j][start:end]
                group_output.append(select_output)

                for layer_idx in range(12):
                    select_pk = origin_past_key_values[layer_idx][0][idx_j][:, start:end, :]
                    select_pv = origin_past_key_values[layer_idx][1][idx_j][:, start:end, :]
                    group_pk[layer_idx].append(select_pk)
                    group_pv[layer_idx].append(select_pv)

        start = 0
        combine_k, combine_v = [[] for _ in range(12)], [[] for _ in range(12)]
        for n in each_sample_chunks_num:
            try:
                encode_result.append(torch.cat(group_output[start:start + n], dim=0))
            except RuntimeError:
                print("")
            # batch_size num_heads, sequence_length, embed_size_per_head, 在dim-1
            for layer_idx in range(12):
                combine_k[layer_idx].append(torch.cat(group_pk[layer_idx][start:start + n], dim=1))
                combine_v[layer_idx].append(torch.cat(group_pv[layer_idx][start:start + n], dim=1))

                group_pk[layer_idx][start:start + n] = [None] * n
                group_pv[layer_idx][start:start + n] = [None] * n

            start = start + n

        lengths = [len(i) for i in encode_result]
        max_len = max(lengths)

        # encode_result = None

        num_heads, _, embed_size_per_head = combine_v[0][0].shape
        pad_zeros = torch.zeros(embed_size_per_head).to(device)
        # combine_k_pad, combine_v_pad = [[] for _ in range(12)], [[] for _ in range(12)]
        for idx in range(len(lengths)):
            cur_len = combine_k[0][idx].shape[1]  # combine_k[layer-id][id-in-batch].shape[1]  len
            cur_pad = pad_zeros.expand(num_heads, max_len - cur_len, embed_size_per_head)
            for layer in range(12):
                combine_k[layer][idx] = torch.cat([combine_k[layer][idx], cur_pad], dim=1)
                combine_v[layer][idx] = torch.cat([combine_v[layer][idx], cur_pad], dim=1)

        for layer in range(12):
            past_key_values.append(
                (torch.stack(combine_k[layer]), torch.stack(combine_v[layer]))
            )
            combine_k[layer] = [None]
            combine_v[layer] = [None]

        history_mask = []
        for i in lengths:
            t = torch.arange(0, max_len)
            history_mask.append(t < i)
        history_mask = torch.stack(history_mask).long().to(device)

        encode_result = pad_sequence(encode_result, batch_first=True).to(device)

        return encode_result, history_mask, past_key_values

    def create_entity_query(self, encoded_history, history_mask):
        history_length = history_mask.sum(dim=1)
        mask_encoded_history = encoded_history * history_mask.unsqueeze(dim=-1)
        avg_pooled_history = mask_encoded_history.sum(dim=1) / history_length.unsqueeze(dim=-1)
        if self.config['entity_query_model'] == "avg_pool_linear":
            avg_pooled_history = self.query_transform(avg_pooled_history)
        return avg_pooled_history

    def forward(self, history_ids=None, history_mask=None, history_speaker=None,
                response_mask=None, response_ids=None, entity_label=None,
                ##
                group_history_ids=None,
                group_history_mask=None,
                start_end_expand_as_batch_size=None,
                each_sample_chunks_num=None,
                ):

        encoded_history, past_key_values, history_mask, entity_loss = self.encode_step(
            history_ids=history_ids,
            history_mask=history_mask,
            token_type_ids=history_speaker,
            entity_label=entity_label,

            # slide window
            group_history_ids=group_history_ids,
            group_history_mask=group_history_mask,
            start_end_expand_as_batch_size=start_end_expand_as_batch_size,
            each_sample_chunks_num=each_sample_chunks_num,

        )

        logits = self.decode_step(
            response_ids=response_ids,
            response_mask=response_mask,
            history_mask=history_mask,
            past_key_values=past_key_values,
            past_key_values_len=0
        )

        return logits, entity_loss

    def encode_step(self,
                    history_ids=None,
                    history_mask=None,
                    token_type_ids=None,
                    entity_label=None,
                    ##
                    group_history_ids=None,
                    group_history_mask=None,
                    start_end_expand_as_batch_size=None,
                    each_sample_chunks_num=None,
                    ):
        entity_loss = None

        if group_history_ids is not None:
            encoded_history, history_mask, past_key_values = self.slide_window_encode(
                group_ids=group_history_ids,
                group_mask=group_history_mask,
                start_end_expand_as_batch_size=start_end_expand_as_batch_size,
                each_sample_chunks_num=each_sample_chunks_num,
            )
        else:

            if self.config['entity_predict']:
                topic_probs, five_topic_probs, entity_loss = self.next_entity_predictor(
                    input_ids=history_ids,
                    attention_mask=history_mask,
                    token_type_ids=token_type_ids,
                    label=entity_label,
                )
                entity_appendix, entity_appendix_mask = \
                    self.next_entity_predictor.translate_prob_to_appendix(topic_probs)

            else:
                entity_appendix, entity_appendix_mask = None, None

            encoder_output = self.encoder(
                history_ids, history_mask, token_type_ids=token_type_ids, use_cache=True,
                # entity_appendix=entity_appendix, entity_appendix_mask=entity_appendix_mask
            )
            encoded_history = encoder_output.last_hidden_state
            past_key_values = encoder_output.past_key_values

            history_mask = encoder_output.pooler_output

        if self.config["entity_attention"] and self.training:
            entities_query = self.create_entity_query(encoded_history, history_mask)
            entities_embedding = self.bert_entity_encoder()
            _, attention_score = self.bert_entity_encoder.query_entities(
                entities_query, entities_embedding
            )
            topic_weight = torch.ones_like(entity_label, dtype=torch.float) + 3 * entity_label
            entity_loss = F.binary_cross_entropy(attention_score, entity_label, topic_weight)

        return encoded_history, past_key_values, history_mask, entity_loss

    def decode_step(self, response_ids=None, response_mask=None, history_mask=None,
                    past_key_values=None, past_key_values_len=0):

        mask = torch.cat([history_mask, response_mask], dim=1)

        decoder_output = self.decoder(
            response_ids,
            mask,
            past_key_values=past_key_values,
            past_key_values_len=past_key_values_len,
            # entity_info=entity_info,  # entity_info: batch_size, 1, hidden_size
        )
        hidden_state = decoder_output.last_hidden_state
        logits = self.lm_linear(hidden_state)

        return logits

#
# if __name__ == '__main__':
#     config = {
#         "bert_config_path": "../pretrained_model/config.json",
#         "bertgpt_state_dict": "../pretrained/bertGPT_pretrained_model.pth",
#         "pretrained_state_dict_path": "../pretrained/PCL-MedBERT/pytorch_model.bin",
#         "pretrained_encoder_config_path": "../pretrained/PCL-MedBERT/config.json",
#         "pretrained_decoder_config_path": "../pretrained/PCL-MedBERT/config_for_decoder.json",
#         "gpt2_config_path": "../pretrained/gpt2/config.json",
#     }
#
#     BERTGPT_HF(config)
