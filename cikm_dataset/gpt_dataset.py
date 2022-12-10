import os
from abc import ABC
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


# 用40表示spk1 用41表示spk2

class GPTDataset(BaseDataset):
    def __init__(self, vocab_path=None, data_path=None, data_type=None, config=None, data=None):
        super(GPTDataset, self).__init__(
            vocab_path=vocab_path, data_path=data_path, data_type=data_type, config=config, data=data
        )
        self.entity_end_idx = self.token2idx['[entities]']

    def combine_entity_to_ids(self, entities):
        if len(entities) == 0:
            return []
        res = []
        for i in entities:
            res.extend(
                [self.token2idx[c] for c in i] + [self.token2idx['、']]
            )
        # res[-1] = self.entity_end_idx
        res[-1] = self.sep_idx
        return res

    def build_GPT_input_ids(self, batch):
        input_ids = []
        token_type_ids = []
        for item in batch:
            dialog = item['text'][0]
            response = item['text'][1]['Sentence']
            cur_history_sents = []
            spks = []
            for h in dialog:
                text_sentence = h['Sentence']
                cur_sentence = self.build_sentence(text_sentence)[:65]
                cur_history_sents.append(cur_sentence)
                spks.append(self.spk2idx[h['id']])

            target_entity = []
            if self.config['use_entity_appendix']:
                for entity_type in self.entity_type:
                    for e in item['text'][1][entity_type]:
                        target_entity.append(e)
                target_entity = self.combine_entity_to_ids(target_entity)

            combine_cur_history_sents = []
            combine_history_spk = []
            for sent, spk in zip(cur_history_sents, spks):
                combine_cur_history_sents = combine_cur_history_sents + sent + [self.sep_idx]
                combine_history_spk = combine_history_spk + [spk] * (len(sent) + 1)

            if len(target_entity) > 0:
                combine_cur_history_sents = combine_cur_history_sents + target_entity
                combine_history_spk = combine_history_spk + [self.spk2idx['Doctor']] * len(target_entity)

            combine_cur_history_sents = combine_cur_history_sents + self.build_sentence(response)[:65] + [self.sep_idx]
            combine_history_spk = combine_history_spk + [self.spk2idx['Doctor']] * (len(
                self.build_sentence(response)[:65]) + 1)
            combine_cur_history_sents = [self.cls_idx] + combine_cur_history_sents[-299:]
            combine_history_spk = [self.spk2idx['Patients']] + combine_history_spk[-299:]

            input_ids.append(combine_cur_history_sents)
            token_type_ids.append(combine_history_spk)

        # for idx in range(len(input_ids)):
        #     print("".join(self.convert_ids_to_tokens(input_ids[idx])))

        input_ids = [torch.tensor(i) for i in input_ids]
        token_type_ids = [torch.tensor(i) for i in token_type_ids]
        return input_ids, token_type_ids

    def build_GPT_test_input_ids(self, batch):
        input_ids = []
        labels = []
        token_type_ids = []
        for item in batch:
            dialog = item['text'][0]
            response = item['text'][1]['Sentence']
            cur_history_sents = []
            spks = []
            for h in dialog:
                text_sentence = h['Sentence']
                cur_sentence = self.build_sentence(text_sentence)[:65]
                cur_history_sents.append(cur_sentence)
                spks.append(self.spk2idx[h['id']])
            target_entity = []
            if self.config['use_entity_appendix']:
                for entity_type in self.entity_type:
                    for e in item['text'][1][entity_type]:
                        target_entity.append(e)
                target_entity = self.combine_entity_to_ids(target_entity)

            combine_cur_history_sents = []
            combine_history_spk = []
            for sent, spk in zip(cur_history_sents, spks):
                combine_cur_history_sents = combine_cur_history_sents + sent + [self.sep_idx]
                combine_history_spk = combine_history_spk + [spk] * (len(sent) + 1)

            if len(target_entity) > 0:
                combine_cur_history_sents = combine_cur_history_sents + target_entity
                combine_history_spk = combine_history_spk + [self.spk2idx['Doctor']] * len(target_entity)

            combine_cur_history_sents = [self.cls_idx] + combine_cur_history_sents[-299:]
            combine_history_spk = [self.spk2idx['Patients']] + combine_history_spk[-299:]

            input_ids.append(combine_cur_history_sents)
            token_type_ids.append(combine_history_spk)
            label = self.build_sentence(response)[:65] + [self.sep_idx]
            labels.append(label)

        # for idx in range(len(input_ids)):
        #     print("".join(self.convert_ids_to_tokens(input_ids[idx])))
        #     print("".join(self.convert_ids_to_tokens(labels[idx])))

        input_ids = [torch.tensor(i) for i in input_ids]
        labels = [torch.tensor(i) for i in labels]
        token_type_ids = [torch.tensor(i) for i in token_type_ids]
        return input_ids, labels, token_type_ids

    def build_GPT_input_ids_special_spk(self, batch):
        input_ids = []

        for item in batch:
            dialog = item['text'][0]
            response = item['text'][1]['Sentence']
            cur_history_sents = []
            spks = []
            for h in dialog:
                text_sentence = h['Sentence']
                cur_sentence = self.build_sentence(text_sentence)[:65]
                cur_history_sents.append(cur_sentence)
                spks.append(self.spk2idx[h['id']])

            target_entity = []
            if self.config['use_entity_appendix']:
                for entity_type in self.entity_type:
                    for e in item['text'][1][entity_type]:
                        target_entity.append(e)
                target_entity = self.combine_entity_to_ids(target_entity)

            combine_cur_history_sents = [self.cls_idx]
            for sent, spk in zip(cur_history_sents, spks):
                combine_cur_history_sents = combine_cur_history_sents + [spk + 40] + sent + [self.sep_idx]

            if len(target_entity) > 0:
                combine_cur_history_sents = combine_cur_history_sents + target_entity

            combine_cur_history_sents = combine_cur_history_sents + [self.spk2idx['Doctor'] + 40] + \
                                        self.build_sentence(response)[:65] + [self.sep_idx]
            combine_cur_history_sents = combine_cur_history_sents[-299:]

            input_ids.append(torch.tensor(combine_cur_history_sents))

        return input_ids, None

    def build_GPT_test_input_ids_special_spk(self, batch):
        input_ids = []
        labels = []
        for item in batch:
            dialog = item['text'][0]
            response = item['text'][1]['Sentence']
            cur_history_sents = []
            spks = []
            for h in dialog:
                text_sentence = h['Sentence']
                cur_sentence = self.build_sentence(text_sentence)[:65]
                cur_history_sents.append(cur_sentence)
                spks.append(self.spk2idx[h['id']])
            target_entity = []
            if self.config['use_entity_appendix']:
                for entity_type in self.entity_type:
                    for e in item['text'][1][entity_type]:
                        target_entity.append(e)
                target_entity = self.combine_entity_to_ids(target_entity)

            combine_cur_history_sents = [self.cls_idx]
            for sent, spk in zip(cur_history_sents, spks):
                combine_cur_history_sents = combine_cur_history_sents + [spk + 40] + sent + [self.sep_idx]

            if len(target_entity) > 0:
                combine_cur_history_sents = combine_cur_history_sents + target_entity

            combine_cur_history_sents =  combine_cur_history_sents[-299:] + \
                                        [self.spk2idx['Doctor'] + 40]

            input_ids.append(combine_cur_history_sents)

            label = self.build_sentence(response)[:65] + [self.sep_idx]
            labels.append(label)

        # for idx in range(len(input_ids)):
        #     print("".join(self.convert_ids_to_tokens(input_ids[idx])))
        #     print("".join(self.convert_ids_to_tokens(labels[idx])))

        input_ids = [torch.tensor(i) for i in input_ids]
        labels = [torch.tensor(i) for i in labels]

        return input_ids, labels, None

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0):
        pad_idx = self.pad_idx
        if self.config['use_entity_appendix']:
            print("use entity appendix")

        def GPT_collate_fn(batch):
            input_ids, token_type_ids = self.build_GPT_input_ids_special_spk(batch)
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_idx)

            if token_type_ids is not None:
                token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=pad_idx)

            ret_data = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
            }
            return ret_data

        def GPT_for_test_collate_fn(batch):
            input_ids, labels, token_type_ids = self.build_GPT_test_input_ids_special_spk(batch)
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_idx)
            labels = pad_sequence(labels, batch_first=True, padding_value=pad_idx)
            if token_type_ids is not None:
                token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=pad_idx)

            ret_data = {
                "input_ids": input_ids,
                "response_ids": labels,
                "token_type_ids": token_type_ids,
            }
            return ret_data

        # if self.data_type == "test":
        #     target_collate_fn = GPT_for_test_collate_fn
        # else:
        #     target_collate_fn = GPT_collate_fn

        if self.data_type == "test":
            target_collate_fn = GPT_for_test_collate_fn
        else:
            target_collate_fn = GPT_collate_fn

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
