import os
from abc import ABC
import regex
import torch
from torch.utils import data
import pickle
from tqdm import tqdm
import copy
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from typing import List, Dict
from copy import deepcopy
import math
import random
from cikm_dataset.dataset import BaseDataset
from textrank4zh import TextRank4Sentence
import re
import networkx
import numpy as np
import json
from cmekg.get_entity import has_relation

if os.path.exists("./cmekg/cmekg_graph.json"):
    cmekg_graph = json.load(open("./cmekg/cmekg_graph.json", 'r', encoding='utf-8'))
else:
    cmekg_graph = json.load(open("../cmekg/cmekg_graph.json", 'r', encoding='utf-8'))


class SummaryResponseDataset(BaseDataset):
    def __init__(
            self,
            vocab_path=None,
            data_path=None,
            data_type=None,
            config=None,
            data=None,
            summary_data=None,
            summary_data_path=None,
            create_one=False,
            summary_strategy='',
            with_entity=False,
            with_summary=False,
            decoder_with_entity=False,
            vmed=False,
            use_gat=False,
            dataset_name="MedDG",
            for_sim=False,
    ):
        super(SummaryResponseDataset, self).__init__(
            vocab_path=vocab_path,
            data_path=data_path,
            data_type=data_type,
            config=config,
            data=data,
        )
        if create_one:
            return
        self.max_input_len = 512 if dataset_name == 'MedDialog' else 512
        self.max_response_len = 256 if dataset_name == 'MedDialog' else 200
        self.max_sent_len = 128 if dataset_name == "MedDialog" else 80
        self.max_sent_num = 6 if dataset_name == 'MedDialog' else 15
        self.with_entity = with_entity
        self.with_summary = with_summary
        self.decoder_with_entity = decoder_with_entity
        self.use_gat = use_gat
        self.dataset_name = dataset_name

        self.summary = []

        self.entity_cls = self.token2idx['[ENTITY]']
        self.summary_cls = self.token2idx['[SUMMARY]']
        self.summary_end_idx = self.token2idx['[SummaryEnd]']
        self.summary_sep_idx = self.token2idx['[SummarySep]']
        self.entity_end_idx = self.token2idx['[EntityEnd]']

        self.summary_strategy = summary_strategy

        self.summary_data = summary_data

        # if dataset_name == "MedDialog":
        #     self.data = self.make_MedDialog_dataset()
        #     return

        if for_sim:
            return
        if summary_data_path is None:
            print("No Summary Data")
        else:
            self.build_summary(summary_data_path)
        if vmed or use_gat:
            self.build_sentence_graph()
        if dataset_name == "MedDialog":
            # self.data = self.make_MedDialog_dataset()
            if self.summary_data is not None:
                self.data = [(i, j) for i, j in zip(self.data, self.summary_data)]
            else:
                self.data = [(i, None) for i in self.data]
        else:
            if self.data is not None and self.summary_data is not None:
                self.data = [(i, j) for i, j in zip(self.data, self.summary_data)]
        self.vmed = vmed

    def make_MedDialog_dataset(self):
        # if os.path.exists(f"./MedDialog/{self.data_type}_pairs.pkl"):
        #     return self.data
        if self.data_type == "train":
            data_path = "./MedDialog/filtered_MedDialog/annotated_entity/train_data.json"
        elif self.data_type == "test":
            data_path = "./MedDialog/filtered_MedDialog/annotated_entity/test_data.json"
        else:
            data_path = "./MedDialog/filtered_MedDialog/annotated_entity/dev_data.json"
        # json_data = json.load(open(data_path, 'r', encoding='utf-8')) if self.data is None else self.data
        json_data = json.load(open(data_path, 'r', encoding='utf-8'))
        std_data = []

        """ 
        dialog = item['text'][0]
        response = item['text'][1]['Sentence']
        history_sent_ids = []
        for h in dialog:
            text_sentence = h['Sentence']
        """
        empty_entity_padding = {k: [] for k in self.entity_type}
        for dialog in tqdm(json_data):
            history_cache = []
            for uid, item in enumerate(dialog):
                utterance = item['Sentence']
                entity = item['Entity']
                spk_str = utterance[:3]
                utterance = utterance[3:]
                if spk_str == "病人：":
                    cur_spk = "Patients"
                elif spk_str == "医生：":
                    cur_spk = "Doctor"
                else:
                    raise ValueError("MedDialog Spk ERROR")
                if uid == 0:
                    if not cur_spk == "Patients":
                        raise ValueError("First Spk Must Be Patients")
                if cur_spk == "Doctor":
                    text_0 = deepcopy(history_cache)
                    text_1 = {"Sentence": utterance}
                    text_1.update(empty_entity_padding)
                    new_item = {"text": [text_0, text_1]}
                    std_data.append(new_item)
                history_cache.append({
                    "Sentence": utterance,
                    "Entity": entity,
                    "id": cur_spk,
                })
        pickle.dump(std_data, open(f"./MedDialog/filtered_MedDialog/{self.data_type}_pairs.pkl", 'wb'))
        return std_data

    def build_sentence_graph(self):
        all_graph_info = []
        if self.dataset_name != "MedDialog":
            if self.summary_strategy == "pcl_bert_sim":
                if os.path.exists("graph_info_pcl_sim_{}.pkl".format(self.data_type)):
                    all_graph_info = pickle.load(open("graph_info_pcl_sim_{}.pkl".format(self.data_type), 'rb'))
                    for cur_idx, cur_data in tqdm(enumerate(self.data), total=len(self.data)):
                        self.data[cur_idx].update(all_graph_info[cur_idx])
                    return
            else:
                if os.path.exists("graph_info_{}_{}.pkl".format(self.summary_strategy, self.data_type)):
                    all_graph_info = pickle.load(
                        open("graph_info_{}_{}.pkl".format(self.summary_strategy, self.data_type), 'rb'))
                    for cur_idx, cur_data in tqdm(enumerate(self.data), total=len(self.data)):
                        self.data[cur_idx].update(all_graph_info[cur_idx])
                    return

        # if self.dataset_name == "MedDialog":
        #     target_path = f"./MedDialog/{self.data_type}_graph.pkl"
        #     if self.config.get(f"{self.data_type}_graph_path") is not None:
        #         target_path = self.config[f"{self.data_type}_graph_path"]
        #     if os.path.exists(target_path):
        #         all_graph_info = pickle.load(open(target_path, 'rb'))
        #         for cur_idx, cur_data in tqdm(enumerate(self.data), total=len(self.data)):
        #             self.data[cur_idx].update(all_graph_info[cur_idx])
        #         return
        # else:
        #     print("graph file not exists!")
        #     exit(-1)

        for cur_idx, cur_data in tqdm(enumerate(self.data), total=len(self.data)):
            head_type = []
            dialog = cur_data['text'][0]
            temp_sentence_entity = []
            for h in dialog:
                text_sentence = h['Sentence']
                head_type.append(self.spk2idx[h['id']])
                if self.dataset_name == "MedDialog":
                    entity = h['Entity']
                else:
                    entity = []
                    for entity_type in self.entity_type:
                        for e in h[entity_type]:
                            entity.append(e)
                temp_sentence_entity.append((text_sentence, entity))

            if self.summary_strategy == "pcl_bert_sim":
                # max_sent_idx = max(cur_data['sim_target_recall']) + 1
                if self.dataset_name == 'MedDialog':
                    temp_sentence_entity = temp_sentence_entity[-self.max_sent_num:]
                    head_type = head_type[-self.max_sent_num:]
                else:
                    reserve_num = len(temp_sentence_entity) - min(cur_data['sim_target_recall'])
                    split_sent_idx = max(15, reserve_num)
                    temp_sentence_entity = temp_sentence_entity[-split_sent_idx:]
                    head_type = head_type[-split_sent_idx:]
            else:
                temp_sentence_entity = temp_sentence_entity[-15:]  # last 15
                head_type = head_type[-15:]

            edges_type_matrix = np.zeros((len(temp_sentence_entity), len(temp_sentence_entity)), dtype=int).tolist()
            for i in range(len(temp_sentence_entity) - 1):
                edges_type_matrix[i][i] = 1
                edges_type_matrix[i][i + 1] = 2
                edges_type_matrix[i + 1][i] = 3
            edges_type_matrix[-1][-1] = 1
            if self.dataset_name == 'MedDialog':
                for i in range(len(temp_sentence_entity)):
                    for j in range(i + 1, len(temp_sentence_entity)):
                        if edges_type_matrix[i][j] != 0:
                            continue
                        sent_i, ent_i = temp_sentence_entity[i]
                        sent_j, ent_j = temp_sentence_entity[j]
                        find_relation = False
                        for ei in ent_i:
                            for ej in ent_j:
                                if has_relation(ei, ej):
                                    edges_type_matrix[i][j] = 4
                                    edges_type_matrix[j][i] = 4
                                    find_relation = True
                                    break
                            if find_relation:
                                break
            else:
                for i in range(len(temp_sentence_entity)):
                    for j in range(i + 1, len(temp_sentence_entity)):
                        sent_i, ent_i = temp_sentence_entity[i]
                        sent_j, ent_j = temp_sentence_entity[j]
                        medicine = cmekg_graph['Medicine']
                        disease = cmekg_graph['Disease']
                        triple = None
                        medicine.update(disease)
                        all_triple = medicine
                        for mh in all_triple:
                            if mh in ent_i:
                                for rel in all_triple[mh]:
                                    mts = all_triple[mh][rel]
                                    if ent_j in mts:
                                        triple = (ent_i, rel, ent_j)
                                        break
                            elif mh in ent_j:
                                for rel in all_triple[mh]:
                                    mts = all_triple[mh][rel]
                                    if ent_i in mts:
                                        triple = (ent_j, rel, ent_i)
                                        break
                            if triple is not None:
                                break
                        if triple is not None:
                            edges_type_matrix[i][j] = 4
                            edges_type_matrix[j][i] = 4

            adjacent_matrix = (torch.tensor(edges_type_matrix) != 0).long()

            cur_graph_info = {
                "adjacent_matrix": adjacent_matrix,
                "edge_type_matrix": edges_type_matrix,
                "head_type": head_type,
            }
            all_graph_info.append(cur_graph_info)

            self.data[cur_idx].update({
                "adjacent_matrix": adjacent_matrix,
                "edge_type_matrix": edges_type_matrix,
                "head_type": head_type,
            })
        # pickle.dump(all_graph_info, open("./MedDialog/{}_graph.pkl".format(self.data_type), 'wb'))

    def build_summary(self, summary_data_path=None):
        if self.dataset_name == "MedDialog":
            sim_sentences = pickle.load(open(summary_data_path, 'rb'))
            self.summary_data = []
            for idx, (item, sim_target_recall) in enumerate(sim_sentences):
                item = item[-2:]
                sim_target_recall = sim_target_recall[-2:]
                cur_summary_data = []
                summary_sentence_ids = []
                for sent in item:
                    cur_sent = self.build_sentence(sent)[:self.max_sent_len]
                    summary_sentence_ids.append(cur_sent)
                for i in summary_sentence_ids:
                    cur_summary_data = cur_summary_data + i + [self.summary_sep_idx]
                self.summary_data.append(cur_summary_data[:-1])
                bit_list_target_recall = [0] * len(self.data[idx]['text'][0])
                if len(bit_list_target_recall) <= max(sim_target_recall):
                    print("")
                for select_id in sim_target_recall:
                    bit_list_target_recall[select_id] = 1
                if self.dataset_name == 'MedDialog':
                    bit_list_target_recall = bit_list_target_recall[-6:]
                self.data[idx]['sim_target_recall'] = bit_list_target_recall
            return
        print("Summary Strategy: {}".format(self.summary_strategy))
        if self.summary_strategy == '':
            if summary_data_path == './data/aaai/dialogue_summary_test.pkl':
                self.summary_data = [[]] * len(self.data)
                return
            if summary_data_path is not None:
                self.summary_data = pickle.load(open(summary_data_path, 'rb'))
                self.clean_summary()
        elif self.summary_strategy == 'last_3_utterance':
            self.summary_data = []
            for item in self.data:
                cur_summary_data = []
                history_sentence_ids = []
                dialog = item['text'][0][-3:]
                for h in dialog:
                    text_sentence = h['Sentence']
                    cur_sentence = self.build_sentence(text_sentence)[:self.max_sent_len]
                    history_sentence_ids.append(cur_sentence)
                for i in history_sentence_ids:
                    cur_summary_data = cur_summary_data + i + [self.summary_sep_idx]
                self.summary_data.append(cur_summary_data[:-1])
        elif self.summary_strategy == "text_rank":
            self.summary_data = []
            tr4s = TextRank4Sentence()
            if os.path.exists("{}-TextRank-summary-cache.pkl".format(self.data_type)):
                self.summary_data = pickle.load(open("{}-TextRank-summary-cache.pkl".format(self.data_type), 'rb'))
                return
            for item in tqdm(self.data, desc="text-rank for summary"):
                history_dict = item['text'][0]
                history_sentence = "\n".join([i['Sentence'] for i in history_dict])
                tr4s.analyze(text=history_sentence, lower=True, source='all_filters')
                summary_analyze_res = tr4s.get_key_sentences(num=3, sentence_min_len=1)
                sent_idx_sent = [(i.index, i.sentence) for i in summary_analyze_res]
                sent_idx_sent = sorted(sent_idx_sent, key=lambda x: x[0])
                summary_sent = [i[1] for i in sent_idx_sent]
                cur_summary_data = []
                for sent in summary_sent:
                    cur_sent = self.build_sentence(sent)[:self.max_sent_len]
                    cur_summary_data = cur_summary_data + cur_sent + [self.summary_sep_idx]
                self.summary_data.append(cur_summary_data[:-1])
                pickle.dump(self.summary_data, open("{}-TextRank-summary-cache.pkl".format(self.data_type), 'wb'))
        elif self.summary_strategy == "pcl_bert_sim":
            self.summary_data = []
            sim_sentences = pickle.load(
                open("{}-sim-based-summary-with-idx.pkl".format(self.data_type), 'rb')
            )[:len(self)]
            for idx, (item, sim_target_recall) in enumerate(sim_sentences):
                cur_summary_data = []
                summary_sentence_ids = []
                for sent in item:
                    cur_sent = self.build_sentence(sent)[:self.max_sent_len]
                    summary_sentence_ids.append(cur_sent)
                for i in summary_sentence_ids:
                    cur_summary_data = cur_summary_data + i + [self.summary_sep_idx]
                self.summary_data.append(cur_summary_data[:-1])
                bit_list_target_recall = [0] * len(self.data[idx]['text'][0])
                if len(bit_list_target_recall) <= max(sim_target_recall):
                    print("")
                for select_id in sim_target_recall:
                    bit_list_target_recall[select_id] = 1
                self.data[idx]['sim_target_recall'] = bit_list_target_recall
        else:
            raise ValueError

    def clean_summary(self):
        for idx, i in enumerate(self.summary_data):
            if i.startswith("摘要：。"):
                self.summary_data[idx] = i.replace("摘要：。", "")
            if i.endswith("建议：。"):
                self.summary_data[idx] = i.replace("建议：。", "")
            if i.startswith("摘要：建议"):
                self.summary_data[idx] = i.replace("摘要：", "")
            if i.endswith("建议："):
                self.summary_data[idx] = i.replace("建议：", "")
            if i.endswith("建议：。"):
                self.summary_data[idx] = i.replace("建议：。", "")

    def encoder_inputs(self, batch, with_entity=False):
        history_ids = []
        history_spk = []
        for item, _ in batch:
            cur_history_ids = []
            cur_history_spk = []
            history_sentence_ids = []
            history_sentence_spk = []
            dialog = item['text'][0]
            for h in dialog:
                text_sentence = h['Sentence']
                cur_sentence = self.build_sentence(text_sentence)[:self.max_sent_len]
                cur_spk = [self.spk2idx[h['id']]] * len(cur_sentence)
                history_sentence_ids.append(cur_sentence)
                history_sentence_spk.append(cur_spk)
            cur_history_ids += [self.cls_idx]
            cur_history_spk += [self.spk2idx['Patients']]
            for i, j in zip(history_sentence_ids, history_sentence_spk):
                cur_history_ids = cur_history_ids + i + [self.sep_idx]
                cur_history_ids = cur_history_ids[-self.max_input_len:]
                cur_history_spk = cur_history_spk + j + [j[0]]
                cur_history_spk = cur_history_spk[-self.max_input_len:]

            #
            if with_entity:
                target_entity = []
                for entity_type in self.entity_type:
                    for e in item['text'][1][entity_type]:
                        target_entity.append(e)
                target_entity = self.combine_entity_to_ids(target_entity)
                if len(target_entity) > 0:
                    cur_history_ids = (cur_history_ids + target_entity)[-self.max_input_len:]
                    cur_history_spk = (cur_history_spk + [2] * len(target_entity))[-self.max_input_len:]

            history_ids.append(torch.tensor(cur_history_ids))
            history_spk.append(torch.tensor(cur_history_spk))
        return history_ids, history_spk

    def decoder_inputs(self, batch, with_summary=False, decoder_with_entity=False):
        targets = []
        summary_end_pos = []
        response_start_pos = []
        prefix = []
        for item, summary in batch:
            response = item['text'][1]['Sentence']
            response = self.build_sentence(response)
            response = [_ for _ in response if _ != self.unk_idx][:self.max_response_len]
            target_ids = [self.cls_idx]

            if with_summary:
                # summary = self.build_sentence(summary)
                summary = [_ for _ in summary if _ != self.unk_idx]
                if len(summary) > 0:
                    summary_ids = summary + [self.summary_end_idx]
                else:
                    summary_ids = []
                target_ids = target_ids + summary_ids

                # if len(summary) > 0:
                #     summary_and_response = ([self.cls_idx] + summary + [self.summary_end_idx]
                #                             + response + [self.sep_idx])[:512]
                #     summary_end_pos.append(summary_and_response.index(self.summary_end_idx))
                # else:
                #     summary_and_response = ([self.cls_idx] + response + [self.sep_idx])[:512]
                #     summary_end_pos.append(0)
                # targets.append(torch.tensor(summary_and_response))
            # else:
            #     response = ([self.cls_idx] + response + [self.sep_idx])[:512]
            #     summary_end_pos.append(0)
            #     targets.append(torch.tensor(response))

            if decoder_with_entity:
                entity_ids = self.get_entity_sequence(item['text'][1])
                if len(entity_ids) > 0:
                    entity_ids = entity_ids + [self.entity_end_idx]
                    target_ids = target_ids + entity_ids

            prefix.append(torch.tensor(target_ids))

            response_start_pos.append(len(target_ids))
            target_ids = target_ids + response + [self.sep_idx]

            try:
                cur_summary_end_pos = target_ids.index(self.summary_end_idx)
                summary_end_pos.append(cur_summary_end_pos)
            except ValueError:
                summary_end_pos.append(-1)

            targets.append(torch.tensor(target_ids))
            # try:
            #     cur_summary_end_pos = target_ids.index(self.summary_end_idx)
            #     summary_end_pos.append(cur_summary_end_pos)
            # except ValueError:
            #     summary_end_pos.append(-1)

        return targets, summary_end_pos, response_start_pos, prefix

    def entity_inputs(self, batch):
        entity = []
        for item, _ in batch:
            target_entity = []
            for entity_type in self.entity_type:
                for e in item['text'][1][entity_type]:
                    target_entity.append(e)
            target_entity = self.combine_entity_to_ids(target_entity)
            target_entity = [self.entity_cls] + target_entity[:-1]
            entity.append(torch.tensor(target_entity))
        return entity

    def get_entity_sequence(self, type_and_entity):
        entity = []
        if self.dataset_name == "MedDialog":
            for e in type_and_entity["Entity"][:10]:
                entity.append(e)
        else:
            for entity_type in self.entity_type:
                for e in type_and_entity[entity_type]:
                    entity.append(e)
        target_entity = self.combine_entity_to_ids(entity)
        return target_entity[:-1]

    def summary_inputs(self, batch, use_summary_cls=True):
        summary = []
        for _, cur_summary in batch:
            cur_summary = self.build_sentence(cur_summary)
            if use_summary_cls:
                cur_summary = [self.summary_cls] + [_ for _ in cur_summary if _ != self.unk_idx]
            else:
                cur_summary = [self.cls_idx] + [_ for _ in cur_summary if _ != self.unk_idx]
            summary.append(torch.tensor(cur_summary))
        return summary

    def summary_entity_cross_inputs(self, batch):
        all_summary = []
        entity = []
        for item, summary in batch:
            summary = [_ for _ in summary if _ != self.unk_idx]
            summary = [self.cls_idx] + summary + [self.summary_end_idx]
            all_summary.append(torch.tensor(summary))
            entity_ids = self.get_entity_sequence(item['text'][1])
            entity_ids = [self.cls_idx] + entity_ids + [self.entity_end_idx]
            entity.append(torch.tensor(entity_ids))
        return all_summary, entity

    def for_sim_inputs(self, batch):
        assert len(batch) == 1
        item = batch[0]
        combine_sent_ids = []
        save_origin_sent = []
        start_end_pos = []
        dialog = item['text'][0]
        response = item['text'][1]['Sentence']
        history_sent_ids = []
        for h in dialog:
            text_sentence = h['Sentence']
            cur_sentence = self.build_sentence(text_sentence)[:self.max_sent_len]
            history_sent_ids.append(cur_sentence)
        combine_sent_ids += [self.cls_idx]
        for i in history_sent_ids:
            combine_sent_ids = combine_sent_ids + i + [self.sep_idx]
            combine_sent_ids = combine_sent_ids[-512:]
        response = self.build_sentence(response)[:self.max_sent_len]
        combine_sent_ids = combine_sent_ids + response + [self.sep_idx]
        combine_sent_ids = combine_sent_ids[-512:]

        combine_tokens = self.convert_ids_to_tokens(combine_sent_ids)

        if combine_tokens[0] == "[SEP]":
            combine_sent_ids = combine_sent_ids[1:]
            combine_tokens = combine_tokens[1:]

        # sep_pos = list(re.finditer('\\[SEP\\]', combine_tokens))
        cls_in_tokens = "[CLS]" in combine_tokens
        last_end = 1 if cls_in_tokens else 0

        # for idx, i in enumerate(sep_pos):
        #     start_end_pos.append((last_end, i.end()))
        #     save_origin_sent.append(combine_tokens[last_end: i.end()])
        #     last_end = i.end()

        for idx, token in enumerate(combine_tokens):
            if token == '[SEP]':
                save_origin_sent.append("".join(combine_tokens[last_end:idx]))
                start_end_pos.append((last_end, idx))
                last_end = idx + 1

        return torch.tensor([combine_sent_ids]), start_end_pos, save_origin_sent

    def get_batch_sentences_with_pad_sentence(self, batch):
        sentences_ids = []
        target_recall = []  # todo 只有最后三句才有效
        for item, _ in batch:
            history_sentence_ids = []
            dialog = item['text'][0]
            # temp_sentence_entity = []
            for h in dialog:
                text_sentence = h['Sentence']
                cur_sentence = [self.cls_idx] + self.build_sentence(text_sentence)[:self.max_sent_len] + [self.sep_idx]
                history_sentence_ids.append(cur_sentence)
                # entity = []
                # for entity_type in self.entity_type:
                #     for e in h[entity_type]:
                #         entity.append(e)
                # temp_sentence_entity.append((text_sentence, entity))
            if self.summary_strategy == "pcl_bert_sim":
                cur_recall = item['sim_target_recall']
                if self.dataset_name == 'MedDialog':
                    history_sentence_ids = history_sentence_ids[-self.max_sent_num:]
                else:
                    reserve_num = len(history_sentence_ids) - min(cur_recall)
                    split_sent_idx = max(15, reserve_num)
                    history_sentence_ids = history_sentence_ids[-split_sent_idx:]
                sentences_ids.append(history_sentence_ids)
                target_recall.append(cur_recall)
            else:
                history_sentence_ids = history_sentence_ids[-15:]
                sentences_ids.append(history_sentence_ids)  # 最多取最后15个句子
                if len(history_sentence_ids) >= 3:
                    cur_recall = [0] * (len(history_sentence_ids) - 3) + [1] * 3
                else:
                    cur_recall = [1] * len(history_sentence_ids)
                target_recall.append(cur_recall)
        sentence_num = [len(i) for i in sentences_ids]
        original_sentences = [list(i) for i in sentences_ids]
        max_sent_num = max(sentence_num)
        # sentences_ids = [i.extend([[0]] * (max_sent_num - len(i))) for i in sentences_ids]
        for idx in range(len(sentences_ids)):
            sentences_ids[idx].extend([[0]] * (max_sent_num - len(sentences_ids[idx])))
        combined_sentences = []
        for i in sentences_ids:
            combined_sentences.extend(i)
        combined_sentences = [torch.tensor(i) for i in combined_sentences]
        # sentences_ids = [torch.tensor(i) for i in sentences_ids]
        last_3_target_recall = pad_sequence([torch.tensor(i) for i in target_recall], batch_first=True,
                                            padding_value=0)
        # if len(combined_sentences) // 6 != last_3_target_recall.shape[1]:
        #     print("")

        return combined_sentences, sentence_num, original_sentences, last_3_target_recall

    def get_batch_sentence_graph(self, batch):
        # batch_edges_type_matrix = []
        all_edges_type = [i['edge_type_matrix'] for i, _ in batch]
        max_sent_num = max(len(i) for i in all_edges_type)
        for idx in range(len(batch)):
            expand_num = max_sent_num - len(all_edges_type[idx])
            if expand_num > 0:
                for raw_id in range(len(all_edges_type[idx])):
                    all_edges_type[idx][raw_id].extend([0] * expand_num)
            if expand_num > 0:
                for exp_idx in range(expand_num):
                    all_edges_type[idx].append([0] * max_sent_num)
                # all_edges_type[idx].extend([[0] * max_sent_num] * expand_num)
            # batch_edges_type_matrix.append(all_edges_type)
        batch_head_type = [torch.tensor(i['head_type']) for i, _ in batch]
        batch_edges_type_matrix = torch.tensor(all_edges_type)
        return batch_head_type, batch_edges_type_matrix

    def get_pretrain_dataloader(self, batch_size, shuffle):
        pad_idx = self.pad_idx

        def collate_fn(batch):
            history_ids, history_spk = self.encoder_inputs(batch, with_entity=False)
            history_ids = pad_sequence(history_ids, batch_first=True, padding_value=pad_idx)
            history_spk = pad_sequence(history_spk, batch_first=True, padding_value=0)
            target_ids, summary_end_pos, response_start_pos, prefix = self.decoder_inputs(
                batch, with_summary=False, decoder_with_entity=False
            )
            target_ids = pad_sequence(target_ids, batch_first=True, padding_value=pad_idx)

            history_mask = (history_ids != pad_idx).long()
            target_mask = (target_ids != pad_idx).long()

            ret_data = {
                "history_ids": history_ids,
                "history_mask": history_mask,
                "history_spk": history_spk,
                "target_ids": target_ids,
                "target_mask": target_mask,
                "response_start_pos": None,
            }
            return ret_data

        return data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=False
        )

    def get_dataloader(
            self, batch_size, shuffle=True, num_workers=0,
            use_hir_attention=False, use_bert_summary_gpt=False,
            for_sim=False,
    ):
        pad_idx = self.pad_idx
        with_entity = self.with_entity
        with_summary = self.with_summary
        decoder_with_entity = self.decoder_with_entity
        use_gat = self.use_gat

        separate_target = True if self.config.get("model_recall_strategy", None) is not None else False

        def collate_fn(batch):
            history_ids, history_spk = self.encoder_inputs(batch, with_entity=with_entity)
            history_ids = pad_sequence(history_ids, batch_first=True, padding_value=pad_idx)
            history_spk = pad_sequence(history_spk, batch_first=True, padding_value=0)
            target_ids, summary_end_pos, response_start_pos, prefix = self.decoder_inputs(
                batch, with_summary=with_summary, decoder_with_entity=decoder_with_entity
            )
            target_ids = pad_sequence(target_ids, batch_first=True, padding_value=pad_idx)

            response_start_pos = torch.tensor(response_start_pos)

            history_mask = (history_ids != pad_idx).long()
            target_mask = (target_ids != pad_idx).long()

            summary, entity = self.summary_entity_cross_inputs(batch)

            summary = pad_sequence(summary, batch_first=True, padding_value=pad_idx)
            entity = pad_sequence(entity, batch_first=True, padding_value=pad_idx)

            summary_mask = (summary != pad_idx).long()
            entity_mask = (entity != pad_idx).long()

            ret_data = {
                "history_ids": history_ids,
                "history_mask": history_mask,
                "history_spk": history_spk,
                "target_ids": target_ids,
                "target_mask": target_mask,
                "summary_end_pos": summary_end_pos,
                "response_start_pos": response_start_pos,
                "prefix": prefix,
                "summary_ids": summary,
                "entity_ids": entity,
                "summary_mask": summary_mask,
                "entity_mask": entity_mask,

            }
            if use_gat:
                sentences_ids, sentences_num, _, target_recall = self.get_batch_sentences_with_pad_sentence(batch)
                sentences_ids = pad_sequence(sentences_ids, batch_first=True, padding_value=0)
                sentences_mask = (sentences_ids != self.pad_idx).long()
                batch_head_type, batch_edges_type_matrix = self.get_batch_sentence_graph(batch)
                batch_head_type = pad_sequence(batch_head_type, batch_first=True, padding_value=0)
                sentence_adjacent_matrix = (batch_edges_type_matrix != pad_idx).long()
                ret_data.update({
                    "sentences_ids": sentences_ids,
                    "sentences_mask": sentences_mask,
                    "sentences_num": sentences_num,
                    "head_type": batch_head_type,
                    "edge_type": batch_edges_type_matrix,
                    "adjacent_matrix": sentence_adjacent_matrix,
                    "target_recall": target_recall,
                })

            if separate_target:
                cur_batch_size, pad_to_length = target_ids.shape
                target_list = target_ids.tolist()
                summary_target_ids = []
                response_target_ids = []
                for bid in range(cur_batch_size):
                    summary_target_ids.append(target_list[bid][:response_start_pos[bid]])
                    response_target_ids.append([pad_idx] * response_start_pos[bid] +
                                               target_list[bid][response_start_pos[bid]:])
                padded_summary_target_ids = [s + [pad_idx] * (pad_to_length - len(s)) for s in summary_target_ids]
                padded_response_target_ids = [r + [pad_idx] * (pad_to_length - len(r)) for r in response_target_ids]
                ret_data.update({
                    "padded_summary_target_ids": torch.tensor(padded_summary_target_ids),
                    "padded_response_target_ids": torch.tensor(padded_response_target_ids)
                })

            return ret_data

        def bert_summary_gpt_collate_fn(batch):
            history_ids, history_spk = self.encoder_inputs(batch, with_entity=False)
            history_ids = pad_sequence(history_ids, batch_first=True, padding_value=pad_idx)
            history_spk = pad_sequence(history_spk, batch_first=True, padding_value=0)
            target_ids, _, _, _ = self.decoder_inputs(batch, with_summary=False)
            target_ids = pad_sequence(target_ids, batch_first=True, padding_value=pad_idx)

            summary = self.summary_inputs(batch, use_summary_cls=False)
            summary = pad_sequence(summary, batch_first=True, padding_value=pad_idx)

            history_mask = (history_ids != pad_idx).long()
            target_mask = (target_ids != pad_idx).long()
            summary_mask = (summary != pad_idx).long()

            ret_data = {
                "history_ids": history_ids,
                "history_mask": history_mask,
                "history_spk": history_spk,
                "target_ids": target_ids,
                "target_mask": target_mask,
                "summary": summary,
                "summary_mask": summary_mask,
            }

            return ret_data

        def for_sim_collate_fn(batch):
            input_ids, start_end_pos, save_origin_sent = self.for_sim_inputs(batch)
            ret_data = {
                "input_ids": input_ids,
                "start_end_pos": start_end_pos,
                "save_origin_sent": save_origin_sent
            }
            return ret_data

        target_collate_fn = None
        if use_bert_summary_gpt:
            target_collate_fn = bert_summary_gpt_collate_fn
        else:
            target_collate_fn = collate_fn

        if for_sim:
            target_collate_fn = for_sim_collate_fn

        print("collate function: {}".format(target_collate_fn.__name__))

        return data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=target_collate_fn,
            pin_memory=False
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
