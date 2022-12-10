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
import textrank4zh
from cikm_model.graph_net import SentRGAT
from torch.nn.utils.rnn import pad_sequence
import random


class PrioriRankNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.query_linear = nn.Linear(768, 512)
        self.key_linear = nn.Linear(768, 512)

    def forward(self, all_sentence_hidden, sentence_num):
        # sentence_num: batch
        compare_matrix = torch.arange(0, max(sentence_num)).repeat(len(sentence_num), 1)
        sentence_num_expand = sentence_num.unsqueeze(dim=-1).repeat(1, max(sentence_num))
        mask = (sentence_num_expand > compare_matrix).long()
        mask_num = (1 - mask) * -2000.0
        all_sentence_hidden_after_mask = all_sentence_hidden * mask.unsqueeze(dim=-1)
        avg_is_query = all_sentence_hidden_after_mask.sum(dim=1) / sentence_num.unsqueeze(dim=-1)

        query = self.query_linear(avg_is_query)  # batch, 1, hidden
        key = self.key_linear(all_sentence_hidden)  # batch, n, hidden
        scores = torch.bmm(key, query.permute(0, 2, 1))  # batch, n, 1
        scores += mask_num
        scores = torch.softmax(scores.unsqueeze(dim=-1), dim=-1)
        return scores


class PosteriorRankNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.query_linear = nn.Linear(768, 512)
        self.key_linear = nn.Linear(768, 512)

    def forward(self, all_sentence_hidden, target_response_hidden, sentence_num):
        compare_matrix = torch.arange(0, max(sentence_num)).repeat(len(sentence_num), 1)
        sentence_num_expand = sentence_num.unsqueeze(dim=-1).repeat(1, max(sentence_num))
        mask = (sentence_num_expand > compare_matrix).long()
        mask_num = (1 - mask) * -2000.0
        query = self.query_linear(target_response_hidden)  # batch, 1, hidden
        key = self.key_linear(all_sentence_hidden)  # batch, n, hidden
        scores = torch.bmm(key, query.permute(0, 2, 1))
        scores += mask_num
        scores = torch.softmax(scores.unsqueeze(dim=-1), dim=-1)
        return scores


class VSumDialog(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.encoder_config = BertConfig.from_json_file(config["pretrained_encoder_config_path"])
        self.decoder_config = BertConfig.from_json_file(config["pretrained_decoder_config_path"])

        self.sentence_encoder = BertModel(config=self.encoder_config, add_pooling_layer=False)

        self.sentence_rgat = SentRGAT(embed_dim=768, edge_embed_dim=768, flag_embed_dim=768, edge_num=300, flag_num=300)

        self.prior_ranking_net = PrioriRankNet(config)
        self.posterior_ranking_net = PosteriorRankNet(config)

        self.skr_entity_encoder = BertModel(config=self.encoder_config, add_pooling_layer=False)
        self.skr_summary_encoder = BertModel(config=self.encoder_config, add_pooling_layer=False)

        self.decoder = BertModel(config=self.decoder_config, add_pooling_layer=False, decoder_like_gpt=True)
        self.lm_linear = nn.Linear(self.encoder_config.hidden_size, self.encoder_config.vocab_size, bias=False)

        self.extract_num = 3

        self.is_train = True
        self.dc = self.config['dataset_class']
        self.token2idx = self.dc.token2idx
        self.idx2token = self.dc.idx2token
        self.entity_cls = self.token2idx['[ENTITY]']
        self.summary_cls = self.token2idx['[SUMMARY]']
        self.summary_end_idx = self.token2idx['[SummaryEnd]']
        self.summary_sep_idx = self.token2idx['[SummarySep]']
        self.entity_end_idx = self.token2idx['[EntityEnd]']
        self.unk_idx = self.token2idx['[UNK]']
        self.pad_idx = self.token2idx['[PAD]']
        self.sep_idx = self.token2idx['[SEP]']
        self.cls_idx = self.token2idx['[CLS]']

    def rgat_sentence_scores(self, sentence_adjacent_matrix, head_type, edge_type,
                             sentence_embedding, target_response_embedding, sentence_num):
        sentence_rgat_hidden = self.sentence_rgat(
            sentence_adjacent_matrix, head_type, edge_type, sentence_embedding
        )
        prior_scores = self.prior_ranking_net(sentence_rgat_hidden, sentence_num)
        # prior_top_value, prior_top_idx = prior_scores.sort()
        if self.is_train:
            posterior_scores = self.posterior_ranking_net(sentence_rgat_hidden, target_response_embedding, sentence_num)
            # post_top_value, post_top_idx = posterior_scores.sort()
        else:
            post_top_value, post_top_idx, posterior_scores = None, None, None

        return prior_scores, posterior_scores

    def prepare_summary_input(self, summaries):
        res_summary = []
        for summary_list in summaries:
            cur_summary_data = []
            summary_sentence_ids = []
            for sent in summary_list:
                cur_sent = self.dc.build_sentence(sent)[:80]
                summary_sentence_ids.append(cur_sent)
            for i in summary_sentence_ids:
                cur_summary_data = cur_summary_data + i + [self.summary_sep_idx]
            res_summary.append(cur_summary_data[:-1])
        return res_summary

    def prepare_decoder_input(self, original_target_response, summary, entity_ids):
        targets = []
        summary_end_pos = []
        response_start_pos = []
        prefix = []
        for r, s, e in zip(original_target_response, summary, entity_ids):
            target_ids = [self.cls_idx]
            summary = [_ for _ in summary if _ != self.unk_idx]
            if len(summary) > 0:
                summary_ids = summary + [self.summary_end_idx]
            else:
                summary_ids = []
            target_ids = target_ids + summary_ids
            if len(entity_ids) > 0:
                entity_ids = entity_ids + [self.entity_end_idx]
                target_ids = target_ids + entity_ids
            prefix.append(torch.tensor(target_ids))
            response_start_pos.append(len(target_ids))
            target_ids = target_ids + r + [self.sep_idx]
            try:
                cur_summary_end_pos = target_ids.index(self.summary_end_idx)
                summary_end_pos.append(cur_summary_end_pos)
            except ValueError:
                summary_end_pos.append(-1)
            targets.append(torch.tensor(target_ids))
        targets = pad_sequence(targets, batch_first=True, padding_value=0)
        return targets, summary_end_pos, response_start_pos, prefix

    def sampled_summary_forward(
            self,
            original_sentence=None,
            sentence_num=None,
            original_target_response=None,
            entity_for_decoder_ids=None,
            input_for_crossattention=None,
            crossattention_mask=None,
            entity_ids=None,
            entity_mask=None,
    ):
        sampled_summary_memory = []
        for bid in range(len(original_sentence)):
            sampled_cur_sum_mem = []
            sampled_idx = random.sample(list(range(0, sentence_num[bid])),
                                        min(self.extract_num, sentence_num[bid]))
            sampled_idx = sorted(sampled_idx)
            for tid in sampled_idx:
                sampled_cur_sum_mem.append(original_sentence[bid][tid])
            sampled_summary_memory.append(sampled_cur_sum_mem)
        sampled_summary_memory = self.prepare_summary_input(sampled_summary_memory)
        sampled_summary_ids = []
        for i in sampled_summary_memory:
            sampled_summary_ids.append([self.cls_idx] + i + [self.summary_end_idx])
        sampled_summary_ids = pad_sequence(sampled_summary_ids, batch_first=True, padding_value=self.pad_idx)
        sampled_summary_mask = (sampled_summary_ids != self.pad_idx).long().to(sampled_summary_ids.device)
        sampled_decoder_input, sampled_summary_end_pos, sampled_response_start_pos, sampled_prefix = \
            self.prepare_decoder_input(original_target_response, sampled_summary_memory, entity_for_decoder_ids)
        sampled_decoder_input = pad_sequence(sampled_decoder_input, batch_first=True, padding_value=self.pad_idx)
        sampled_decoder_mask = (sampled_decoder_input != self.pad_idx).long().to(sampled_decoder_input.device)
        s_outer_encoded_hidden_states, s_crossattention_mask, s_summary_hidden, s_entity_hidden = self.encode_step(
            input_for_crossattention=input_for_crossattention,
            crossattention_mask=crossattention_mask,
            summary_ids=sampled_summary_ids,
            summary_mask=sampled_summary_mask,
            entity_ids=entity_ids,
            entity_mask=entity_mask,
        )
        s_logits = self.decode_step(
            response_ids=sampled_decoder_input,
            response_mask=sampled_decoder_mask,
            outer_encoded_hidden_states=s_outer_encoded_hidden_states,
            outer_attention_mask=s_crossattention_mask,
            summary_hidden=s_summary_hidden,
            summary_mask=sampled_summary_mask,
            entity_hidden=s_entity_hidden,
            entity_mask=entity_mask,
        )

        return s_logits,

    def get_summary_and_decoder_input(
            self,
            sentences_ids=None,
            sentences_ids_mask=None,
            original_sentence=None,
            sentence_num=None,
            sentence_adjacent_matrix=None,
            head_type=None,
            edge_type=None,
            target_response=None,
            target_response_mask=None,
            original_target_response=None,
            entity_for_decoder_ids=None,
    ):
        sentence_encoder_outputs = self.sentence_encoder(sentences_ids, sentences_ids_mask)
        sentence_embedding = sentence_encoder_outputs[0]
        sentence_embedding = sentence_embedding * sentences_ids_mask.unsqueeze(dim=-1)
        each_sentence_len = sentences_ids_mask.sum(dim=1)
        each_sentence_len = torch.where(each_sentence_len != 0, each_sentence_len, 1)
        sentence_embedding = sentence_embedding.sum(dim=1) / each_sentence_len.unsqueeze(dim=-1)
        batch_size = len(sentence_num)
        max_sent_num = max(sentence_num)
        sentence_embedding = sentence_embedding.view(batch_size, max_sent_num, -1)
        target_response_outputs = self.sentence_encoder(target_response, target_response_mask)
        target_response_embedding = target_response_outputs[0]
        target_response_embedding = target_response_embedding * target_response_mask.unsqueeze(dim=-1)
        each_response_len = target_response_mask.sum(dim=1)
        each_response_len = torch.where(each_response_len != 0, each_response_len, 1)
        target_response_embedding = target_response_embedding.sum(dim=1) / each_response_len.unsqueeze(dim=-1)

        prior_scores, posterior_scores = self.rgat_sentence_scores(
            sentence_adjacent_matrix, head_type, edge_type,
            sentence_embedding, target_response_embedding, sentence_num
        )
        scores = posterior_scores if posterior_scores is not None else prior_scores

        summary_memory = []
        for bid in range(len(original_sentence)):
            cur_sum_mem = []
            value, top_idx = scores[bid].topk(self.extract_num)
            top_idx, _ = top_idx.sort()
            for tid in top_idx:
                cur_sum_mem.append(original_sentence[bid][tid])
            summary_memory.append(cur_sum_mem)
        summary_memory = self.prepare_summary_input(summary_memory)
        summary_ids = []
        for i in summary_memory:
            summary_ids.append([self.cls_idx] + i + [self.summary_end_idx])
        summary_ids = pad_sequence(summary_ids, batch_first=True, padding_value=self.pad_idx)
        summary_mask = (summary_ids != self.pad_idx).long().to(summary_ids.device)
        decoder_input, summary_end_pos, response_start_pos, prefix = self.prepare_decoder_input(
            original_target_response, summary_memory, entity_for_decoder_ids
        )
        decoder_input = pad_sequence(decoder_input, batch_first=True, padding_value=self.pad_idx)
        decoder_mask = (decoder_input != self.pad_idx).long().to(decoder_input.device)

        return summary_ids, summary_mask, decoder_input, decoder_mask

    def forward(
            self,
            sentences_ids=None,
            original_sentence=None,  # list 没有对sentence数量做pad
            sentence_num=None,
            sentence_adjacent_matrix=None,
            head_type=None,
            edge_type=None,
            target_response=None,
            original_target_response=None,
            entity_for_decoder_ids=None,
            input_for_crossattention=None,
            crossattention_mask=None,
            entity_ids=None,  # entity_for_mem
            entity_mask=None,  # entity_for_mem_mask
            sentences_ids_mask=None,
            target_response_mask=None,

    ):
        summary_ids, summary_mask, decoder_input, decoder_mask = self.get_summary_and_decoder_input(
            sentences_ids=sentences_ids,
            sentences_ids_mask=sentences_ids_mask,
            target_response_mask=target_response_mask,
            original_sentence=original_sentence,
            sentence_num=sentence_num,
            sentence_adjacent_matrix=sentence_adjacent_matrix,
            head_type=head_type,
            edge_type=edge_type,
            target_response=target_response,
            original_target_response=original_target_response,
            entity_for_decoder_ids=entity_for_decoder_ids
        )
        outer_encoded_hidden_states, crossattention_mask, summary_hidden, entity_hidden = self.encode_step(
            input_for_crossattention=input_for_crossattention,
            crossattention_mask=crossattention_mask,
            summary_ids=summary_ids,
            summary_mask=summary_mask,
            entity_ids=entity_ids,
            entity_mask=entity_mask,
        )
        logits = self.decode_step(
            response_ids=decoder_input,
            response_mask=decoder_mask,
            outer_encoded_hidden_states=outer_encoded_hidden_states,
            outer_attention_mask=crossattention_mask,
            summary_hidden=summary_hidden,
            summary_mask=summary_mask,
            entity_hidden=entity_hidden,
            entity_mask=entity_mask,
        )
        return logits,

    def requires_grad_ed(self, require):
        self.sentence_encoder.requires_grad_(require)
        self.prior_ranking_net.requires_grad_(require)
        self.skr_entity_encoder.requires_grad_(require)
        self.skr_summary_encoder.requires_grad_(require)
        self.decoder.requires_grad_(require)
        self.lm_linear.requires_grad_(require)

    def encode_step(
            self,
            input_for_crossattention=None,
            crossattention_mask=None,
            summary_ids=None,
            summary_mask=None,
            entity_ids=None,
            entity_mask=None,
    ):
        outer_encoder_outputs = self.encoder_for_crossattention(
            input_for_crossattention,
            crossattention_mask
        )
        outer_encoded_hidden_states = outer_encoder_outputs[0]
        summary_hidden, entity_hidden = None, None
        if self.config['summary_entity_encoder']:
            summary_outputs = self.skr_summary_encoder(
                summary_ids,
                summary_mask
            )
            summary_hidden = summary_outputs[0]
            entity_outputs = self.skr_entity_encoder(
                entity_ids,
                entity_mask,
            )
            entity_hidden = entity_outputs[0]
        return outer_encoded_hidden_states, crossattention_mask, summary_hidden, entity_hidden

    def decode_step(
            self,
            response_ids=None,
            response_mask=None,
            outer_encoded_hidden_states=None,
            outer_attention_mask=None,

            summary_hidden=None,
            summary_mask=None,
            entity_hidden=None,
            entity_mask=None,
    ):
        decoder_output = self.decoder(
            response_ids,
            response_mask,
            outer_encoded_hidden_states=outer_encoded_hidden_states,
            outer_attention_mask=outer_attention_mask,
            summary_hidden=summary_hidden,
            summary_mask=summary_mask,
            entity_hidden=entity_hidden,
            entity_mask=entity_mask,
        )
        hidden_state = decoder_output.last_hidden_state
        logits = self.lm_linear(hidden_state)

        return logits

    def set_train(self):
        self.is_train = True

    def set_eval(self):
        self.is_train = False
