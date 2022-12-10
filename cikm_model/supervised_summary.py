import copy
from abc import ABC
import torch
from torch import nn
from src.utils.hugging_face_bert import BertModel
from transformers import BertConfig, GPT2Config
from copy import deepcopy
from cikm_model.graph_net import SentRGAT
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence


def sentence_pool(sentences, sentences_mask, with_pad_sent=False):
    """
    :param sentences: (batch_size * sent_num) * sent_len * hidden
    :param sentences_mask: (batch_size * sent_num) * sent_len
    """
    sentences_len = sentences_mask.sum(dim=1)
    if with_pad_sent:
        sentences_len = torch.where(sentences_len != 0, sentences_len, 1)
    mask_sentences = sentences * sentences_mask.unsqueeze(dim=-1)
    pooled_sentences = mask_sentences.sum(dim=1) / sentences_len.unsqueeze(dim=-1)
    return pooled_sentences


class GATScore(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.edge_num = 5
        self.flag_num = 2  # speaker
        self.embed_dim = 512
        self.hidden_project = nn.Linear(768, self.embed_dim)
        self.rgat = SentRGAT(
            embed_dim=self.embed_dim,
            edge_embed_dim=self.embed_dim,
            flag_embed_dim=self.embed_dim,
            edge_num=self.edge_num,
            flag_num=self.flag_num
        )

        self.query_layer = nn.Linear(768, self.embed_dim)
        self.key_layer = nn.Linear(self.embed_dim, self.embed_dim)
        self.layer_norm_q = nn.LayerNorm(self.embed_dim)
        self.layer_norm_k = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            sentences_hidden,
            sentences_num,
            sentences_mask,
            sent_adjacent_matrix,
            head_type,
            edge_type,
            node_query,
            target_recall=None
    ):
        # sentences_mask: (batch * sent) * sent_len
        # all_sentence_hidden_after_mask = sentences_hidden * sentences_mask.unsqueeze(dim=-1)
        # each_sentence_len = sentences_mask.sum(dim=1)
        # each_sentence_len = torch.where(each_sentence_len != 0, each_sentence_len, 1)
        # sentence_node_embed = all_sentence_hidden_after_mask.sum(dim=1) / each_sentence_len.unsqueeze(dim=-1)
        # node_query = node_query.data
        sentence_node_embed = sentence_pool(sentences_hidden, sentences_mask, with_pad_sent=True)
        sentence_node_embed = self.hidden_project(sentence_node_embed)
        batch_size = len(sentences_num)
        max_sent_num = max(sentences_num)
        sentence_node_embed = sentence_node_embed.view(batch_size, max_sent_num, self.embed_dim)
        sentence_node_hidden = self.rgat(sent_adjacent_matrix, head_type, edge_type, sentence_node_embed)
        query = self.query_layer(node_query)  # batch*hidden
        query = self.layer_norm_q(query)
        key = self.key_layer(sentence_node_hidden)  # batch*sent_num*hidden
        key = self.layer_norm_k(key)

        # batch * sent_num * hidden  bmm  batch * hidden * 1
        recall_scores = torch.bmm(key, query.unsqueeze(dim=1).transpose(1, 2))
        recall_scores = recall_scores.squeeze(dim=-1)  # batch,
        recall_scores = torch.sigmoid(recall_scores)

        pad_of_sent_mask = (sentences_mask.sum(dim=-1) != 0).long().view(batch_size, max_sent_num)
        recall_scores = recall_scores * pad_of_sent_mask

        loss, accuracy = None, None
        if target_recall is not None:
            loss = F.binary_cross_entropy(recall_scores, target_recall.to(recall_scores.dtype))
            not_ignore = pad_of_sent_mask.bool()
            num_targets = pad_of_sent_mask.sum().item()
            preds = (recall_scores > 0.5).long()
            correct = (preds == target_recall) & not_ignore
            correct = correct.float().sum()
            accuracy = correct / num_targets

            # print(recall_scores)
            recall_scores = target_recall

        return recall_scores, loss, accuracy, sentence_node_hidden


class GenSummaryEntityResponse(nn.Module, ABC):
    def __init__(self, config):
        super(GenSummaryEntityResponse, self).__init__()
        self.config = dict(config)
        self.encoder_config = BertConfig.from_json_file(config["pretrained_encoder_config_path"])

        self.use_encoder_for_crossattention = True

        self.recall_strategy = config.get('recall_gate_network', None)
        if self.recall_strategy is not None:
            if "GAT" in self.recall_strategy:
                self.recall_gate_network = GATScore(config)
            else:
                raise NotImplementedError
        else:
            self.recall_gate_network = None

        self.entity_gate_open = config['entity_gate_open']
        self.summary_gate_open = config['summary_gate_open']  # summary is changed to recall

        self.rsep_as_associate = config['rsep_as_associate']

        self.decoder_config = BertConfig.from_json_file(config["pretrained_decoder_config_path"])
        self.decoder = BertModel(
            config=self.decoder_config,
            add_pooling_layer=False,
            decoder_like_gpt=True,
            use_references_crossattention=self.use_encoder_for_crossattention,
            entity_gate_open=self.entity_gate_open,
            summary_gate_open=self.summary_gate_open,
            rsep_as_associate=self.rsep_as_associate,
            # use_summary_entity_crossattention=config['summary_entity_encoder']
        )

        self.lm_linear = nn.Linear(self.encoder_config.hidden_size, self.encoder_config.vocab_size, bias=False)

        self.rc_linear = None
        if self.config.get("model_recall_strategy", None ) is not None:
            self.rc_linear = nn.Linear(self.encoder_config.hidden_size, self.encoder_config.vocab_size, bias=False)


        self.encoder_for_crossattention = BertModel(
            config=self.encoder_config,
            add_pooling_layer=False,
            decoder_like_gpt=False,
        )

        encoder_state_dict, decoder_state_dict, lm_state_dict = self.load_bert_gpt_state_dict()
        self.encoder_for_crossattention.load_state_dict(encoder_state_dict, strict=False)
        self.decoder.load_state_dict(decoder_state_dict, strict=False)
        self.lm_linear.load_state_dict(lm_state_dict, strict=False)

        if self.rc_linear is not None:
            self.rc_linear.load_state_dict(lm_state_dict, strict=False)

        if self.summary_gate_open:
            self.skr_summary_encoder = BertModel(
                config=self.encoder_config,
                add_pooling_layer=False,
                decoder_like_gpt=False,
            )
            self.skr_summary_encoder.load_state_dict(copy.deepcopy(encoder_state_dict), strict=False)
            self.gat_embed_project = nn.Linear(512, 768)
        if self.entity_gate_open:
            self.skr_entity_encoder = BertModel(
                config=self.encoder_config,
                add_pooling_layer=False,
                decoder_like_gpt=False,
            )
            self.skr_entity_encoder.load_state_dict(copy.deepcopy(encoder_state_dict), strict=False)

        self.with_target_recall = True

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
            response_mask=None,
            response_ids=None,
            input_for_crossattention=None,
            crossattention_mask=None,
            token_type_ids=None,

            summary_ids=None,
            summary_mask=None,
            entity_ids=None,
            entity_mask=None,

            sentences_ids=None,
            sentences_mask=None,
            sentences_num=None,
            adjacent_matrix=None,
            head_type=None,
            edge_type=None,
            target_recall=None,

            rsep_position=None,
            pooled_entity_hidden=None,
    ):
        # if self.with_target_recall is False:
        #     target_recall = None


        outer_encoded_hidden_states, crossattention_mask, summary_hidden, \
        entity_hidden, summary_mask, recall_loss, recall_acc = self.encode_step(
            # reorganize
            input_for_crossattention=input_for_crossattention,
            crossattention_mask=crossattention_mask,
            token_type_ids=token_type_ids,

            summary_ids=summary_ids,
            summary_mask=summary_mask,
            entity_ids=entity_ids,
            entity_mask=entity_mask,

            sentences_ids=sentences_ids,
            sentences_mask=sentences_mask,
            sentences_num=sentences_num,

            adjacent_matrix=adjacent_matrix,
            head_type=head_type,
            edge_type=edge_type,
            target_recall=target_recall,
        )

        logits = self.decode_step(
            response_ids=response_ids,
            response_mask=response_mask,
            outer_encoded_hidden_states=outer_encoded_hidden_states,
            outer_attention_mask=crossattention_mask,

            summary_hidden=summary_hidden,
            summary_mask=summary_mask,
            entity_hidden=entity_hidden,
            entity_mask=entity_mask,
            rsep_position=rsep_position,
        )

        return logits, recall_loss, recall_acc

    def encode_step(
            self,
            input_for_crossattention=None,
            crossattention_mask=None,
            token_type_ids=None,
            summary_ids=None,
            summary_mask=None,
            entity_ids=None,
            entity_mask=None,

            sentences_ids=None,
            sentences_mask=None,
            sentences_num=None,
            adjacent_matrix=None,
            head_type=None,
            edge_type=None,
            target_recall=None,
    ):
        if self.with_target_recall is False:
            target_recall = None

        outer_encoder_outputs = self.encoder_for_crossattention(
            input_for_crossattention,
            crossattention_mask,
            token_type_ids=token_type_ids
        )

        outer_encoded_hidden_states = outer_encoder_outputs[0]

        summary_hidden, entity_hidden = None, None
        recall_loss, recall_acc = None, None
        if self.summary_gate_open:
            if self.recall_strategy == 'GAT':
                sentences_outputs = self.skr_summary_encoder(sentences_ids, sentences_mask)
                sentences_hidden = sentences_outputs[0]
                node_query = sentence_pool(outer_encoded_hidden_states, crossattention_mask)

                recall_net_outputs = self.recall_gate_network(
                    sentences_hidden, sentences_num, sentences_mask, adjacent_matrix,
                    head_type, edge_type, node_query, target_recall=target_recall
                )
                if target_recall is not None:
                    sentences_recall_score = recall_net_outputs[0]
                else:
                    # recall score:
                    pred_recall_scores = recall_net_outputs[0]
                    topk_indices = pred_recall_scores.topk(min(3, pred_recall_scores.shape[1])).indices
                    sentences_recall_score = torch.zeros_like(pred_recall_scores).to(pred_recall_scores.device)
                    for bid in range(sentences_recall_score.shape[0]):
                        for topk_idx in topk_indices[bid].tolist():
                            if topk_idx < sentences_num[bid]:
                                sentences_recall_score[bid][topk_idx] = 1
                batch_size = len(sentences_num)
                sent_num = max(sentences_num)
                sent_len = sentences_hidden.shape[1]
                sentences_hidden = sentences_hidden.view(batch_size, sent_num, sent_len, 768)
                sentence_node_hidden_from_gat = recall_net_outputs[3]
                sentences_hidden = self.sentences_hidden_fuse_gat_hidden(
                    sentences_hidden, sentence_node_hidden_from_gat
                )
                summary_hidden, summary_mask = self.weighted_and_concat(
                    sentences_hidden, sentences_recall_score, sentences_num, sentences_mask,
                    target_recall=target_recall
                )
                recall_loss, recall_acc = recall_net_outputs[1], recall_net_outputs[2]
            else:
                summary_outputs = self.skr_summary_encoder(summary_ids, summary_mask)
                summary_hidden = summary_outputs[0]

        if self.entity_gate_open:
            entity_outputs = self.skr_entity_encoder(entity_ids, entity_mask)
            entity_hidden = entity_outputs[0]

        return outer_encoded_hidden_states, crossattention_mask, summary_hidden, \
               entity_hidden, summary_mask, recall_loss, recall_acc

    def sentences_hidden_fuse_gat_hidden(self, sentences_hidden, gat_hidden):
        # gat_hidden: batch_size, max_sent_num, gat_embed_dim
        # sentences_hidden: batch_size, max_sent_num, sent_len, sentence_hidden_dim
        batch_size, max_sent_num, sent_len, sentence_hidden_dim = sentences_hidden.shape
        gat_hidden = self.gat_embed_project(gat_hidden).unsqueeze(dim=2)
        zeros = torch.zeros(batch_size, max_sent_num, sent_len - 1, sentence_hidden_dim).to(gat_hidden.device)
        gat_hidden = torch.cat([gat_hidden, zeros], dim=2)
        sentences_hidden = gat_hidden + sentences_hidden
        return sentences_hidden

    def weighted_and_concat(self, sentences_hidden, gate_score, sentences_num, sentences_mask, target_recall=None):
        # sentences_hidden: (batch * sent_num) * len * hidden

        # if target_recall is None:
        sentences_hidden = sentences_hidden * gate_score.unsqueeze(dim=-1).unsqueeze(dim=-1)
        batch_size, sent_num, sent_len, hidden_size = sentences_hidden.shape
        weighted_history = sentences_hidden.view(batch_size, sent_num * sent_len, hidden_size)
        weighted_history_mask = sentences_mask.view(batch_size, sent_num, sent_len). \
            view(batch_size, sent_num * sent_len)
        # else:
        #     weighted_history = []
        #     weighted_history_mask = []
        #     batch_size, sent_num, sent_len, hidden_size = sentences_hidden.shape
        #     sentences_mask = sentences_mask.view(batch_size, sent_num, sent_len)
        #     for cur_target_recall, sentences, masks in zip(target_recall, sentences_hidden, sentences_mask):
        #         cur_history = []
        #         cur_mask = []
        #         for idx in range(len(cur_target_recall)):
        #             if cur_target_recall[idx] == 1:
        #                 cur_history.append(sentences[idx])
        #                 cur_mask.append(masks[idx])
        #         cur_history = torch.cat(cur_history, dim=0)
        #         cur_mask = torch.cat(cur_mask, dim=0)
        #         weighted_history.append(cur_history)
        #         weighted_history_mask.append(cur_mask)
        #
        #     weighted_history = pad_sequence(weighted_history, batch_first=True, padding_value=0)
        #     weighted_history_mask = pad_sequence(weighted_history_mask, batch_first=True, padding_value=0)
        #     # weighted_history = torch.stack(weighted_history, dim=0)
        #     # weighted_history_mask = torch.stack(weighted_history_mask, dim=0)

        return weighted_history, weighted_history_mask

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

            rsep_position=None,
            return_past_key_and_values=False,
            past_key_and_values=None,
    ):
        pooled_entity_hidden = None
        if self.rsep_as_associate:
            if entity_mask is None:
                entity_mask = torch.ones(entity_hidden.shape[:2]).to(entity_hidden.device)
            pooled_entity_hidden = sentence_pool(entity_hidden, entity_mask, with_pad_sent=False)
        decoder_output = self.decoder(
            response_ids,
            response_mask,
            outer_encoded_hidden_states=outer_encoded_hidden_states,
            outer_attention_mask=outer_attention_mask,
            summary_hidden=summary_hidden,
            summary_mask=summary_mask,
            entity_hidden=entity_hidden,
            entity_mask=entity_mask,

            rsep_position=rsep_position,
            pooled_entity_hidden=pooled_entity_hidden,
            past_key_values=past_key_and_values,
        )
        hidden_state = decoder_output.last_hidden_state
        logits = self.lm_linear(hidden_state)

        if self.rc_linear is not None:
            logits = [logits, self.rc_linear(hidden_state)]

        if not return_past_key_and_values:
            return logits
        else:
            return logits, decoder_output.past_key_values
