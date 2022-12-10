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


class ResponseSelectDataset(data.Dataset, ABC):
    def __init__(self, vocab_path, data_path=None, data_type=None, config=None,
                 tokenizer=None, neg_sample_num=4,
                 all_doctor_sentence=None,
                 used_data=None):
        super(ResponseSelectDataset, self).__init__()
        self.config = dict(config) if config is not None else dict()
        self.data_type = data_type
        assert self.data_type in ['train', 'test', 'dev']
        self.tokenizer = tokenizer
        self.token2idx = dict()
        self.idx2token = dict()
        with open(vocab_path, 'r', encoding='utf-8') as reader:
            for idx, token in enumerate(list(reader.readlines())):
                token = token.strip()
                self.token2idx[token] = idx
                self.idx2token[idx] = token
        self.unk_idx = self.token2idx['[UNK]']
        self.pad_idx = self.token2idx['[PAD]']
        self.sep_idx = self.token2idx['[SEP]']
        self.cls_idx = self.token2idx['[CLS]']
        self.spk2idx = {"Patients": 0, "Doctor": 1}

        self.origin_data = pickle.load(open(data_path, 'rb')) if used_data is None else None
        self.all_doctor_sentence = self.get_all_doctor_sentence() if all_doctor_sentence is None else all_doctor_sentence

        self.data: List[Dict] = self.origin_data if used_data is None else used_data

        self.neg_sample_num = neg_sample_num

    def get_all_doctor_sentence(self):
        sentence = []
        for i in self.origin_data:
            dialog = i['text'][0]
            for turn in dialog:
                if turn['id'] == "Doctor":
                    sentence.append(turn['Sentence'])
            reply = i['text'][1]
            sentence.append(reply['Sentence'])
        return sentence

    def negative_sample(self, positive_sentence, sample_num=None):
        sample_sentence = random.sample(self.all_doctor_sentence, sample_num)
        # 去重 算了
        # eq2pos_num = [i == positive_sentence for i in sample_sentence]
        # if eq2pos_num != 0:
        #     re
        sample_sentence = [self._build_sentence(i)[-300:] for i in sample_sentence]
        return sample_sentence

    def _build_sentence(self, text_sentence):
        text_sentence = text_sentence.replace(' ', '').replace("\n", ''). \
            replace('\t', '').replace('“', '"').replace('”', '"').replace('\u3000', '').replace('\u00A0', '')
        text_sentence = text_sentence.lower()

        return [self.token2idx.get(i, self.unk_idx) for i in text_sentence]

    def convert_ids_to_tokens(self, ids):
        return [self.idx2token[i] for i in ids]

    def get_history(self, batch):
        history = []
        for item in batch:
            cur_history = []
            for turn in item['text'][0]:
                cur_history.extend(
                    self._build_sentence(turn['Sentence'])[-50:] + [self.sep_idx]
                )
            history.append(cur_history[-300:])
        return history

    def get_positive_sentence(self, batch):
        return [
            self._build_sentence(i['text'][1]['Sentence'])[-300:] for i in batch
        ]

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0):
        pad_idx = self.pad_idx
        data_type = self.data_type

        def collate_fn(batch):
            # 300 history 300 reply
            # 0           1
            cur_batch_size = len(batch)
            sentence_num = self.neg_sample_num + 1

            history = self.get_history(batch)
            positive_sentence = self.get_positive_sentence(batch)
            negative_sentence = [self.negative_sample(ps, sample_num=self.neg_sample_num) for ps in positive_sentence]

            input_ids = []
            labels = []
            token_type_ids = []
            for cur_history, cur_pos_sent, cur_neg_sent in zip(history, positive_sentence, negative_sentence):
                input_ids.append(
                    [self.cls_idx] + (cur_history + cur_pos_sent)[-510:] + [self.sep_idx]
                )
                token_type_ids.append(
                    [0] + ([0] * len(cur_history) + [1] * len(cur_pos_sent))[-510:] + [1]
                )
                labels.append(1)

                for neg in cur_neg_sent:
                    input_ids.append(
                        [self.cls_idx] + (cur_history + neg)[-510:] + [self.sep_idx]
                    )
                    token_type_ids.append(
                        [0] + ([0] * len(cur_history) + [1] * len(neg))[-510:] + [1]
                    )
                    labels.append(0)
            input_ids = [torch.tensor(i) for i in input_ids]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_idx)
            input_ids = input_ids.view(cur_batch_size, sentence_num, -1)

            token_type_ids = [torch.tensor(i) for i in token_type_ids]
            token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
            token_type_ids = token_type_ids.view(cur_batch_size, sentence_num, -1)

            labels = torch.tensor(labels).view(cur_batch_size, sentence_num)

            attention_mask = (input_ids != pad_idx).long()

            ret_data = {
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'labels': labels,
                'attention_mask': attention_mask
            }

            if data_type == 'test':
                raise NotImplementedError
            return ret_data

        target_collate_fn = collate_fn

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
