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
import json
from cikm_entity_annotation import get_entity_type
from cmekg.get_entity import replace_dict, reverse_replace_dict

graph = json.load(open("./cmekg/cmekg_graph.json", 'r', encoding='utf-8'))


def find_entity(e):
    triples = []
    subgraph = []
    if e in replace_dict:
        e = replace_dict[e]
    for g in graph:
        for h in graph[g]:
            if h == e:
                subgraph.append({e: graph[g][e]})
            else:
                for r in graph[g][h]:
                    if e in graph[g][h][r]:
                        triples.append(
                            (h, r, e)
                        )
    return triples, subgraph


def add_entity_filter(ents):
    fil_ents = []
    for e in ents:
        if e in ['肠炎',
                 '胃炎',
                 '莫沙必利',
                 '奥美',
                 '蒙脱石散',
                 '整肠生',
                 '便秘',
                 '吗丁啉', '阑尾炎', '达喜', '思密达', '乳果糖'
                 ]:
            fil_ents.append(e)
    return fil_ents


class TripleSelectorPredictDataset(BaseDataset):
    def __init__(self, original_data_path=None, vocab_path=None, data_path=None, data_type=None,
                 preprocess=False, config=None, data=None):
        super(TripleSelectorPredictDataset, self).__init__(
            original_data_path=original_data_path, vocab_path=vocab_path, data_path=data_path, data_type=data_type,
            preprocess=preprocess, config=config, data=data
        )

    def triple_build(self, path):
        items = pickle.load(open(path, 'rb'))
        positive_items = []
        negative_items = []
        bar = tqdm(range(len(items)))
        for idx in bar:
            sample = items[idx]
            history_entities = set()
            for i in sample['text'][0]:
                history_entities.update(
                    i['Symptom'] +
                    i['Medicine'] +
                    i['Test'] +
                    i['Attribute'] +
                    i['Disease']
                )
            target_entities = set(
                sample['text'][1]['Medicine'] +
                sample['text'][1]['Disease']
            )

            ths = []
            trans_triples = []

            for he in history_entities:
                t, s = find_entity(he)
                for (th, tr, tt) in t:
                    ths.append(th)
                    if th in reverse_replace_dict:
                        for r in reverse_replace_dict[th]:
                            trans_triples.append((r, tr, tt))
                    else:
                        trans_triples.append((th, tr, tt))
            ths_t = []
            for tmp in ths:
                if tmp in reverse_replace_dict:
                    ths_t += add_entity_filter(reverse_replace_dict[tmp])
                else:
                    ths_t += add_entity_filter([tmp])

            for te in target_entities:
                for triple in trans_triples:
                    if triple[0] == te:
                        positive_items.append({
                            "text": sample['text'],
                            "triple": triple,
                            "label": 1,
                            "triple_src": idx,
                        })
                    else:
                        negative_items.append({
                            "text": sample['text'],
                            "triple": triple,
                            "label": 0,
                            "triple_src": idx,
                        })

            # ths = set(ths_t)
            # in_num += sum([1 for i in ths if i in target_entities])
            # all_num += len(target_entities)
            # pred_num += len(ths)
            # all_test_ths.append(ths)
            # all_target_entities.append(target_entities)

            bar.set_description(desc="pos: {}, neg:{}".format(len(positive_items), len(negative_items)))
        pos_num = len(positive_items)
        neg_num = len(negative_items)
        print(pos_num)
        print(neg_num)
        # sample_neg = random.sample(negative_items, 5 * len(positive_items))
        # neg_and_pos = positive_items + sample_neg
        neg_and_pos = positive_items + negative_items
        random.shuffle(neg_and_pos)
        pickle.dump(neg_and_pos, open("../data/cikm/triples_test.pkl", 'wb'))

    def build_inputs(self, batch):
        history_ids_with_triples = []
        token_type_ids = []
        labels = []

        for item in batch:
            cur_triple = item['triple']
            cur_history_ids = []

            history_sentence_ids = []
            dialog = item['text'][0]
            for h in dialog:
                text_sentence = h['Sentence']
                cur_sentence = self.build_sentence(text_sentence)[:100]  # 限制长度
                history_sentence_ids.append(cur_sentence)

            cur_history_ids += [self.cls_idx]
            for i in history_sentence_ids:
                cur_history_ids = cur_history_ids + i + [self.sep_idx]
                cur_history_ids = cur_history_ids[-512:]

            triple_seq = ",".join(cur_triple)
            triple_seq = [self.token2idx[_] for _ in triple_seq] + [self.sep_idx]

            cur_history_with_triple = (cur_history_ids + triple_seq)[-512:]
            cur_token_type_ids = ([0] * len(cur_history_ids) + [1] * len(triple_seq))[-512:]

            token_type_ids.append(torch.tensor(cur_token_type_ids))
            history_ids_with_triples.append(torch.tensor(cur_history_with_triple))

            labels.append(item['label'])

        return token_type_ids, history_ids_with_triples, torch.tensor(labels)

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0):
        pad_idx = self.pad_idx

        def collate_fn(batch):
            token_type_ids, input_ids, labels = self.build_inputs(batch)
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_idx)
            attention_mask = (input_ids != pad_idx).long()
            token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)

            ret_data = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

            return ret_data

        def predict_collate_fn(batch):
            token_type_ids, input_ids, labels = self.build_inputs(batch)
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_idx)
            attention_mask = (input_ids != pad_idx).long()
            token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)

            triples = [item['triple'] for item in batch]
            triples_src = [item['triple_src'] for item in batch]

            ret_data = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "triples": triples,
                "triples_src": triples_src
            }

            return ret_data

        if self.data_type == 'test':
            target_fn = predict_collate_fn
        else:
            target_fn = collate_fn

        return data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=target_fn,
            pin_memory=False
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

if __name__ == '__main__':
    dataset = TripleSelectorPredictDataset(
        data_type='train',
        vocab_path="../data/vocab.txt"
    )
    dataset.triple_build(path="../data/cikm/process_test.pkl")
