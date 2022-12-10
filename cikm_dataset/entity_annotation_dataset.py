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


class EntityAnnotationDataset(BaseDataset):
    def __init__(self, original_data_path=None, vocab_path=None, data_path=None, data_type=None,
                 preprocess=False, config=None, data=None):
        super(EntityAnnotationDataset, self).__init__(
            original_data_path=original_data_path, vocab_path=vocab_path, data_path=data_path, data_type=data_type,
            preprocess=preprocess, config=config, data=data
        )

    def get_input_ids(self, batch):
        input_ids = []
        labels = []
        for item in batch:
            sentence = item[0]
            sentence = self.convert_to_input_ids_for_annotation(sentence)
            input_ids.append(sentence)
            labels.append(item[1])

        input_ids = [torch.tensor(i) for i in input_ids]
        labels = torch.tensor(labels)
        return input_ids, labels

    def convert_to_input_ids_for_annotation(self, sentence):
        sentence = [self.cls_idx] + self.build_sentence(sentence)[:300] + [self.sep_idx]
        return sentence

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0):
        pad_idx = self.pad_idx

        def collate_fn(batch):
            input_ids, label = self.get_input_ids(batch)
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_idx)
            attention_mask = (input_ids != pad_idx).long()
            ret_data = {
                "input_ids": input_ids,
                "label": label,
                "attention_mask": attention_mask
            }
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


def build_entity_dataset():
    import sys
    sys.path.append("..")
    from cikm_config import config
    train_data = pickle.load(open("../data/original/new_train.pk", 'rb'))
    sentence_and_entities = []
    for i in tqdm(train_data):
        for t in i:
            if t['id'] != 'Doctor':
                continue
            assert t['id'] == 'Doctor'
            sentence = t['Sentence']
            entity_label = [0] * len(config['entity'])
            for et in config['entity_type']:
                for e in t[et]:
                    e = e.lower()
                    eid = config['entity2eid'][e]
                    entity_label[eid] = 1
            sentence_and_entities.append((sentence, entity_label))
    print(len(sentence_and_entities))
    random.shuffle(sentence_and_entities)
    random.shuffle(sentence_and_entities)
    random.shuffle(sentence_and_entities)
    random.shuffle(sentence_and_entities)
    random.shuffle(sentence_and_entities)
    pickle.dump(sentence_and_entities, open("../data/cikm/entity-annotation/sentence-and-entities.pkl", 'wb'))


if __name__ == '__main__':
    build_entity_dataset()
