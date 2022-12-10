from abc import ABC
import torch
from src.utils.gpt_modeling import TransformerEncoder, TransformerDecoderLM
from torch import nn
from src.utils.hugging_face_bert import BertModel
from src.utils.hugging_face_gpt import GPT2Model
from src.utils.hugging_face_gpt import Attention
from transformers import BertConfig, GPT2Config
import torch.nn.functional as F
from src.model import BERTGPTEntity
import json
from typing import Any, Tuple


# PointerNet
class LSTMSummaryGenerator(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)

        in_size, out_size = 768, 768

        self.lstm = nn.GRU(
            input_size=in_size,
            hidden_size=in_size,
            bidirectional=False,  # 这是decoder
            batch_first=True
        )
        self.query = nn.Linear(in_size, out_size)
        self.key = nn.Linear(in_size, out_size)
        self.value = nn.Linear(in_size, out_size)
        self.attention_fusion_with_embedding = nn.Linear(in_size * 2, out_size)

    @staticmethod
    def expand_encoded_mask(encoded_mask):
        # entity_mask = (1.0 - entity_mask) * -10000.0
        expand_history_mask = (1.0 - encoded_mask) * -10000.0
        return expand_history_mask

    def init_decoder_states(self, encoded_history=None, history_mask=None):
        history_length = history_mask.sum(dim=1)
        mask_encoded_history = encoded_history * history_mask.unsqueeze(dim=-1)
        avg_pooled_history = mask_encoded_history.sum(dim=1) / history_length.unsqueeze(dim=-1)
        return avg_pooled_history

    def forward(
            self,
            encoded_history=None,
            history_mask=None,
            summary_input_ids=None,  # 有监督时用
            history_input_ids=None,
            history_input_embeddings=None,
            bos_embedding=None,
            summary_generation_constrains=None,
            # 这个embedding需要加上position embedding和token type embedding吗
            # 先不加. token type 用2 position 用lstm的输出顺序
    ):
        # convert = summary_generation_constrains['dataset'].convert_ids_to_tokens
        # input_ids: batch * seq_len
        cur_decoder_states = self.init_decoder_states(
            encoded_history=encoded_history,
            history_mask=history_mask
        )
        # cur_decoder_states = None
        expand_history_mask = self.expand_encoded_mask(history_mask)
        collect_next_input_embedding = []
        collect_extract_tokens = []
        next_input_embedding = bos_embedding  # cls embedding  这里可能不能用cls开始
        max_step = 50
        steps = 0
        while True:
            decode_step_outputs = self.decode_step(
                input_embedding=next_input_embedding,
                cur_states=cur_decoder_states,
                encoded_history=encoded_history,
                expand_history_mask=expand_history_mask
            )
            lstm_outputs, cur_decoder_states, attention_score, \
            attention_score_logits = decode_step_outputs

            history_input_select_one_hot = torch.softmax((attention_score_logits / 0.1), dim=1)
            select_position = torch.argmax(history_input_select_one_hot, dim=1)
            next_input_embedding = torch.matmul(history_input_select_one_hot.unsqueeze(dim=1), history_input_embeddings)
            next_input_embedding = next_input_embedding.squeeze(dim=1)
            collect_next_input_embedding.append(next_input_embedding)
            collect_extract_tokens.append(
                torch.gather(history_input_ids, 1, select_position.unsqueeze(dim=-1)).squeeze(dim=-1)
            )
            steps += 1
            if steps >= max_step:
                break
        hard_tokens_sequence = torch.stack(collect_extract_tokens, dim=1)
        soft_embeddings_sequence = torch.stack(collect_next_input_embedding, dim=1)
        summary_output = {
            "soft_embeddings_sequence": soft_embeddings_sequence,
            "hard_tokens_sequence": hard_tokens_sequence,
        }
        return summary_output

    def decode_step(self, input_embedding=None, cur_states=None, encoded_history=None, expand_history_mask=None):
        # input_ids: batch * 1
        key_layer = self.key(encoded_history)
        value_layer = self.value(encoded_history)
        query_layer = self.query(cur_states)
        attention_score_logits = torch.bmm(query_layer.unsqueeze(dim=1), key_layer.permute(0, 2, 1))
        attention_score_logits = attention_score_logits.squeeze(dim=1)
        attention_score_logits = attention_score_logits + expand_history_mask
        attention_score = torch.softmax(attention_score_logits, dim=1)
        context = torch.bmm(attention_score.unsqueeze(dim=1), value_layer)
        context = context.squeeze(dim=1)
        inputs = self.attention_fusion_with_embedding(torch.cat([context, input_embedding], dim=-1))
        lstm_outputs, last_states = self.lstm(inputs.unsqueeze(dim=1), cur_states.unsqueeze(dim=0))
        last_states = last_states.squeeze(dim=0)
        return lstm_outputs, last_states, attention_score, attention_score_logits


class GPTSummaryGenerator(nn.Module, ABC):
    pass


class BertSummaryGpt(nn.Module, ABC):
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
                                 use_hir_attention=False,
                                 )
        # entity_attention=True if entity_attention_type == 'EntityAttention' else False)
        self.lm_linear = nn.Linear(self.encoder_config.hidden_size, self.encoder_config.vocab_size,
                                   # bias=True)
                                   bias=False)

        encoder_state_dict, decoder_state_dict, lm_state_dict = self.load_bert_gpt_state_dict()
        self.encoder.load_state_dict(encoder_state_dict, strict=False)
        self.decoder.load_state_dict(decoder_state_dict, strict=False)
        self.lm_linear.load_state_dict(lm_state_dict, strict=False)

        if config.get("summary_model_type") == "bertgpt":
            self.summary_decoder = BertModel(
                config=self.decoder_config,
                add_pooling_layer=False,
                decoder_like_gpt=True,
                use_hir_attention=False,
            )
            self.summary_lm_linear = nn.Linear(
                self.encoder_config.hidden_size, self.encoder_config.vocab_size, bias=False
            )
            summary_state_dict = torch.load(config['summary_state_dict'])
            summary_decoder_state, summary_lm_state = dict(), dict()
            for i, j in summary_state_dict.items():
                if i.startswith('decoder.'):
                    summary_decoder_state[i[len('decoder.'):]] = j
                    # summary_decoder_state[i] = j
                elif i.startswith('lm_linear.'):
                    summary_lm_state[i[len('lm_linear.'):]] = j
                    # summary_lm_state[i] = j
            self.summary_decoder.load_state_dict(summary_decoder_state)
            self.summary_lm_linear.load_state_dict(summary_lm_state)
        elif config.get('summary_model_type') == "pointer_net":
            self.summary_pointer_net = LSTMSummaryGenerator(config)

        for p_name, param in self.named_parameters():
            if p_name.startswith("encoder") or p_name.startswith("decoder")\
                    or p_name.startswith('lm_linear'):
                param.requires_grad = False

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

        print("token type num embedding expand to 3")
        token_type_embed = new_state_dict['encoder.embeddings.token_type_embeddings.weight']
        token_type_embed = torch.cat(
            [token_type_embed, torch.normal(size=(1, 768), mean=0, std=0.02)], dim=0
        )
        new_state_dict['encoder.embeddings.token_type_embeddings.weight'] = token_type_embed
        print(new_state_dict['encoder.embeddings.token_type_embeddings.weight'].shape)

        encoder_state_dict, decoder_state_dict, lm_state_dict = dict(), dict(), dict()

        for i, j in new_state_dict.items():
            if i.startswith('encoder'):
                encoder_state_dict[i] = j
            elif i.startswith('decoder'):
                decoder_state_dict[i] = j
            elif i.startswith('lm_linear'):
                lm_state_dict[i] = j
            else:
                raise ValueError
        return encoder_state_dict, decoder_state_dict, lm_state_dict

    def forward(
            self,
            history_ids=None,
            history_mask=None,
            history_speaker=None,
            response_mask=None,
            response_ids=None,
            summary_generation_constrains=None,
            summary=None,
            summary_mask=None,
            summary_task=None
    ):

        encoded_history, past_key_values, history_mask, = self.encode_step(
            history_ids=history_ids,
            history_mask=history_mask,
            token_type_ids=history_speaker,
            summary_generation_constrains=summary_generation_constrains,
            summary=summary,
            summary_mask=summary_mask,
            summary_task=summary_task
        )

        logits = self.decode_step(
            response_ids=response_ids,
            response_mask=response_mask,
            history_mask=history_mask,
            past_key_values=past_key_values,
            past_key_values_len=0,
        )

        return logits, None

    def encode_step(
            self,
            history_ids=None,
            history_mask=None,
            token_type_ids=None,

            summary=None,
            summary_mask=None,
            summary_generation_constrains=None,
            summary_task=None
    ):

        encoder_output = self.encoder(
            history_ids, history_mask, token_type_ids=token_type_ids, use_cache=True,
        )
        encoded_history = encoder_output.last_hidden_state
        past_key_values = encoder_output.past_key_values
        history_mask = encoder_output.pooler_output
        extra_information = encoder_output.extra_information
        history_input_embeddings = extra_information['history_input_embeddings']

        summary_outputs = self.generate_summary(
            past_key_values=past_key_values,
            past_key_values_len=0,
            history_ids=history_ids,
            history_input_embeddings=history_input_embeddings,
            encoded_history=encoded_history,
            history_mask=history_mask,
            summary_ids=summary,
            summary_mask=summary_mask,
            generation_constrains=summary_generation_constrains,
            embedding_matrix_of_encoder=self.encoder.get_input_embeddings().weight,
            task=summary_task,
        )

        if summary_task == "supervised":
            # TODO
            exit(8)
            logits = summary_outputs[0]
            summary_encoded_output = self.encoder(
                input_ids=summary,
                attention_mask=summary_mask,
                token_type_ids=(torch.ones_like(summary) * 2).to(summary.device),
                use_cache=True,
            )
        elif summary_task == "unsupervised":
            soft_embeddings_sequence = summary_outputs['soft_embeddings_sequence']
            hard_tokens_sequence = summary_outputs['hard_tokens_sequence']
            summary_ids = hard_tokens_sequence

            convert = summary_generation_constrains['dataset'].convert_ids_to_tokens
            print("-" * 30)
            print("".join(convert([_ for _ in history_ids[0].tolist() if _ != 0])))
            print("".join(convert(summary_ids[0].tolist())))
            # print("".join(convert([_ for _ in history_ids[1].tolist() if _ != 0])))
            # print("".join(convert(summary_ids[1].tolist())))
            print("-" * 30)

            summary_mask = (summary_ids != summary_generation_constrains['pad_idx']).long().to(summary_ids.device)
            summary_encoded_output = self.encoder(
                input_ids=None,
                inputs_embeds=soft_embeddings_sequence,
                attention_mask=summary_mask,
                token_type_ids=(torch.ones_like(summary_ids) * 2).to(summary_ids.device),
                use_cache=True,
            )
        else:
            # TODO
            exit(8)
            summary_ids = summary_outputs[0]
            summary_mask = (summary_ids != summary_generation_constrains['pad_idx']).long().to(summary_ids.device)
            summary_encoded_output = self.encoder(
                input_ids=None,
                inputs_embeds=summary_ids,
                attention_mask=summary_mask,
                token_type_ids=(torch.ones_like(summary_ids) * 2).to(summary_ids.device),
                use_cache=True,
            )
        encoded_summary = summary_encoded_output.last_hidden_state
        summary_past_key_values = summary_encoded_output.past_key_values
        summary_mask = summary_encoded_output.pooler_output

        # return encoded_history, past_key_values, history_mask
        return encoded_history, summary_past_key_values, summary_mask

    def generate_summary(
            self,
            past_key_values=None,
            past_key_values_len=0,
            history_ids=None,
            history_input_embeddings=None,
            encoded_history=None,
            history_mask=None,
            summary_ids=None,
            summary_mask=None,
            generation_constrains=None,  # dict
            embedding_matrix_of_encoder=None,
            task=None,  # supervised, unsupervised, inference
    ):
        if self.config['summary_model_type'] == 'bertgpt':
            summary_outputs = self.gpt_generate_summary(
                past_key_values=past_key_values,
                past_key_values_len=0,
                history_mask=history_mask,
                summary_ids=summary_ids,
                summary_mask=summary_mask,
                generation_constrains=generation_constrains,
                embedding_matrix_of_encoder=embedding_matrix_of_encoder,
                task=task
            )
        elif self.config['summary_model_type'] == 'pointer_net':
            cls_idx = torch.tensor([generation_constrains['cls_idx']] * encoded_history.shape[0]). \
                to(encoded_history.device)
            bos_embedding = self.encoder.embeddings.word_embeddings(cls_idx)
            summary_outputs = self.summary_pointer_net(
                encoded_history=encoded_history,
                history_mask=history_mask,
                summary_input_ids=summary_ids,
                history_input_ids=history_ids,
                history_input_embeddings=history_input_embeddings,
                bos_embedding=bos_embedding,
            )
        else:
            raise NotImplementedError

        return summary_outputs

    def lstm_generate_summary(
            self,
            encoded_history=None,
            history_mask=None,
            summary_input_ids=None,
            history_input_ids=None,
            history_input_embeddings=None
    ):
        summary_outputs = self.summary_pointer_net(
            encoded_history=encoded_history,
            history_mask=history_mask,
            summary_input_ids=summary_input_ids,
            history_input_ids=history_input_ids,
            history_input_embeddings=history_input_embeddings,
            bos_embedding=None,
        )
        return summary_outputs

    def gpt_generate_summary(
            self,
            past_key_values=None,
            past_key_values_len=0,
            history_mask=None,
            summary_ids=None,
            summary_mask=None,
            generation_constrains=None,  # dict
            embedding_matrix_of_encoder=None,
            task=None,  # supervised, unsupervised, inference
    ):
        if task == "supervised":
            mask = torch.cat([history_mask, summary_mask], dim=1)
            summary_decoder_output = self.summary_decoder(
                summary_ids,
                mask,
                past_key_values=past_key_values,
                past_key_values_len=past_key_values_len,
            )
            hidden_state = summary_decoder_output.last_hidden_state
            logits = self.lm_linear(hidden_state)
            return logits,
        elif task == "unsupervised":
            device = history_mask.device
            pad_token_id, eos_token_id = generation_constrains['pad_idx'], generation_constrains['sep_idx']
            max_len = generation_constrains['max_len']
            batch_size = history_mask.shape[0]
            seqs = torch.tensor([[generation_constrains['cls_idx']]] * batch_size).to(device)
            unfinished_sequences = torch.ones(seqs.shape[0], dtype=torch.long).to(device)
            steps = 0
            collect_soft_output = []
            collect_next_tokens = []
            while steps < max_len:
                decode_step_output = self.summary_decode_step(
                    response_ids=seqs,
                    response_mask=torch.ones_like(seqs).to(device),
                    history_mask=history_mask,
                    past_key_values=past_key_values,
                    past_key_values_len=0
                )

                logits = decode_step_output
                next_token_logits = logits[:, -1, :]
                logits = next_token_logits / 0.5
                soft_output = F.softmax(logits, dim=-1)
                next_tokens = torch.argmax(soft_output, dim=-1)
                collect_soft_output.append(soft_output)
                collect_next_tokens.append(next_tokens)
                next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)
                seqs = torch.cat([seqs, next_tokens.unsqueeze(-1)], dim=-1)
                unfinished_sequences = unfinished_sequences.mul((~(next_tokens == eos_token_id)).long())

                steps += 1
                if unfinished_sequences.max() == 0:
                    break

            collect_soft_output = torch.stack(collect_soft_output, dim=1).to(device)
            collect_next_tokens = torch.stack(collect_next_tokens, dim=1).to(device)
            soft_embeddings_sequence = torch.matmul(collect_soft_output, embedding_matrix_of_encoder)
            summary_outputs = {
                "soft_embeddings_sequence": soft_embeddings_sequence,
                "hard_tokens_sequence": collect_next_tokens
            }
            return summary_outputs
            # return collect_soft_output, collect_next_tokens
        elif task == "inference":
            device = history_mask.device
            max_len = generation_constrains['max_len']
            batch_size = history_mask.shape[0]
            seqs = torch.tensor([[generation_constrains['cls_idx']]] * batch_size).to(device)
            unfinished_sequences = torch.ones(seqs.shape[0], dtype=torch.long).to(device)
            steps = 0
            pad_token_id, eos_token_id = generation_constrains['pad_idx'], generation_constrains['sep_idx']
            while steps < max_len:
                decode_step_output = self.summary_decode_step(
                    response_ids=seqs,
                    response_mask=torch.ones_like(seqs).to(device),
                    history_mask=history_mask,
                    past_key_values=past_key_values,
                    past_key_values_len=0
                )

                logits = decode_step_output
                next_token_logits = logits[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)
                seqs = torch.cat([seqs, next_tokens.unsqueeze(-1)], dim=-1)
                unfinished_sequences = unfinished_sequences.mul((~(next_tokens == eos_token_id)).long())

                steps += 1
                if unfinished_sequences.max() == 0:
                    break
            return seqs,

    def decode_step(
            self, response_ids=None, response_mask=None, history_mask=None,
            past_key_values=None, past_key_values_len=0,
    ):

        mask = torch.cat([history_mask, response_mask], dim=1)

        decoder_output = self.decoder(
            response_ids,
            mask,
            past_key_values=past_key_values,
            past_key_values_len=past_key_values_len,
        )
        hidden_state = decoder_output.last_hidden_state
        logits = self.lm_linear(hidden_state)

        return logits

    def summary_decode_step(
            self, response_ids=None, response_mask=None, history_mask=None,
            past_key_values=None, past_key_values_len=0,
    ):

        mask = torch.cat([history_mask, response_mask], dim=1)

        decoder_output = self.summary_decoder(
            response_ids,
            mask,
            past_key_values=past_key_values,
            past_key_values_len=past_key_values_len,
        )
        hidden_state = decoder_output.last_hidden_state
        logits = self.lm_linear(hidden_state)

        return logits
