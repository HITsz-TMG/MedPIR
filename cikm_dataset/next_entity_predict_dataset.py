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


class CIKMNextEntityPredictDataset(BaseDataset):
    def __init__(self, original_data_path=None, vocab_path=None, data_path=None, data_type=None,
                 preprocess=False, config=None, data=None):
        super(CIKMNextEntityPredictDataset, self).__init__(
            original_data_path=original_data_path, vocab_path=vocab_path, data_path=data_path, data_type=data_type,
            preprocess=preprocess, config=config, data=data
        )

    def get_history_ids(self, batch):
        history_ids = []
        for item in batch:
            cur_history_ids = []
            history_sentence_ids = []
            dialog = item['text'][0]
            for h in dialog:
                text_sentence = h['Sentence']
                cur_sentence = self.build_sentence(text_sentence)
                history_sentence_ids.append(cur_sentence)
            cur_history_ids += [self.cls_idx]
            for i in history_sentence_ids:
                cur_history_ids = cur_history_ids + i + [self.sep_idx]
                cur_history_ids = cur_history_ids[-512:]
            history_ids.append(torch.tensor(cur_history_ids))
        return history_ids

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

        def collate_fn(batch):
            history_ids = self.get_history_ids(batch)
            history_ids = pad_sequence(history_ids, batch_first=True, padding_value=pad_idx)
            history_mask = (history_ids != pad_idx).long()

            ret_data = {
                "input_ids": history_ids,
                "attention_mask": history_mask,
            }

            if self.data_type != "test":
                entity_label = self.get_entities_label(batch)
                entity_label = torch.tensor(entity_label)
                ret_data["label"] = entity_label

            return ret_data

        return data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
