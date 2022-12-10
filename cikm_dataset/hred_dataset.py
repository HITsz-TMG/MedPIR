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


class HREDDataset(data.Dataset, ABC):
    def __init__(self, vocab_path=None, data_path=None, data_type=None, config=None, data=None):
        super(HREDDataset, self).__init__()
        self.config = dict(config) if config is not None else dict()
        self.data_type = data_type
        assert self.data_type in ['train', 'test', 'dev']
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
        self.data: List[Dict] = []
        self.entity_type = ['Symptom', 'Medicine', 'Test', 'Attribute', 'Disease']

        if self.config.get("entity", None) is not None:
            self.entity2idx = {e: idx for idx, e in enumerate(self.config['entity'])}
            self.idx2entity = {idx: e for idx, e in enumerate(self.config['entity'])}
        else:
            self.entity2idx, self.idx2entity = None, None

        if data_path is not None:
            self.data = pickle.load(open(data_path, 'rb'))

        if data is not None:
            self.data = data

        print(self.data_type + ": " + str(len(self.data)))
        # self.data = self.data[:10] + self.data[-10:]

    def build_sentence(self, text_sentence):
        text_sentence = text_sentence.replace(' ', '').replace("\n", ''). \
            replace('\t', '').replace('“', '"').replace('”', '"').replace('\u3000', '').replace('\u00A0', '')
        text_sentence = text_sentence.lower()
        return [self.token2idx.get(i, self.unk_idx) for i in text_sentence]

    def change_to_lower(self, temp):
        Symptom, Medicine, Test, Attribute, Disease, = temp["Symptom"], temp["Medicine"], temp["Test"], temp[
            "Attribute"], temp["Disease"]
        for i in range(len(Symptom)):
            Symptom[i] = Symptom[i].lower()
        for i in range(len(Medicine)):
            Medicine[i] = Medicine[i].lower()
        for i in range(len(Test)):
            Test[i] = Test[i].lower()
        for i in range(len(Attribute)):
            Attribute[i] = Attribute[i].lower()
        for i in range(len(Disease)):
            Disease[i] = Disease[i].lower()
        return temp

    def convert_ids_to_tokens(self, ids):
        return [self.idx2token[i] for i in ids]

    def combine_entity_to_ids(self, item):
        entities = []
        for entity_type in self.entity_type:
            for e in item['text'][1][entity_type]:
                entities.append(e)
        if len(entities) == 0:
            return False
        res = []
        for i in entities:
            res.extend(
                [self.token2idx[c] for c in i] + [self.token2idx['、']]
            )
        return res

    def history_for_HRED(self, batch):
        # bs * sen_num * seq_len history
        history_ids = []
        for item in batch:
            cur_history_sents = []
            dialog = item['text'][0]

            for h in dialog:
                text_sentence = h['Sentence']
                cur_sentence = self.build_sentence(text_sentence)[:65]
                cur_history_sents.append(cur_sentence)
            if self.config['use_entity_appendix']:
                entity_appendix_sentence = self.combine_entity_to_ids(item)
                if entity_appendix_sentence is not False:
                    cur_history_sents.append(entity_appendix_sentence)
            history_ids.append(cur_history_sents)

        utterance_num = [len(i) for i in history_ids]
        max_sent_num = max(utterance_num)
        for idx in range(len(history_ids)):
            history_ids[idx].extend([[]] * (max_sent_num - len(history_ids[idx])))
            assert len(history_ids[idx]) == max_sent_num

        res_history_ids = []  # bs*sent, seq_len
        for i in range(len(history_ids)):
            for j in range(max_sent_num):
                res_history_ids.append(torch.tensor(history_ids[i][j]))

        return res_history_ids, utterance_num, max_sent_num

    def process_response(self, batch):
        response_ids = []
        for item in batch:
            response = item['text'][1]['Sentence']
            response = self.build_sentence(response)[:65]
            response = [_ for _ in response if _ != self.unk_idx]
            response = [self.cls_idx] + response + [self.sep_idx]
            response_ids.append(torch.tensor(response))
        return response_ids

    def get_entities_label(self, batch):
        batch_entity_label = []
        for i in batch:
            entity_label = len(self.entity2idx) * [0]
            for entity_type in self.entity_type:
                for e in i['text'][1][entity_type]:
                    e_id = self.entity2idx[e]
                    entity_label[e_id] = 1
            batch_entity_label.append(entity_label)
        return batch_entity_label

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0):
        pad_idx = self.pad_idx
        if self.config['use_entity_appendix']:
            print("use entity appendix")

        def HRED_collate_fn(batch):
            cur_batch_size = len(batch)
            history_ids, utterance_num, max_sent_num = self.history_for_HRED(batch)
            history_ids = pad_sequence(history_ids, batch_first=True, padding_value=pad_idx)
            history_ids = history_ids.view(cur_batch_size, max_sent_num, -1)
            history_mask = (history_ids != pad_idx).long()

            ret_data = {
                "history_ids": history_ids,
                "history_mask": history_mask,
                "utterance_num": utterance_num
            }

            response_ids = self.process_response(batch)
            response_ids = pad_sequence(response_ids, batch_first=True, padding_value=pad_idx)
            ret_data['response_ids'] = response_ids
            response_mask = (response_ids != pad_idx).long()
            ret_data['response_mask'] = response_mask

            return ret_data

        target_collate_fn = HRED_collate_fn
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
