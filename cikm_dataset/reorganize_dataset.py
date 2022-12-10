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


class ReorganizeDataset(BaseDataset):
    def __init__(
            self,
            vocab_path=None,
            data_path=None,
            data_type=None,
            config=None,
            data=None,
            ref_data_path=None,
            ref_data=None,
            create_one=False
    ):
        super(ReorganizeDataset, self).__init__(
            vocab_path=vocab_path, data_path=data_path, data_type=data_type, config=config, data=data
        )

        if create_one:
            return

        if ref_data_path is not None:
            self.ref_data = pickle.load(open(ref_data_path, 'rb'))
        if ref_data is not None:
            self.ref_data = ref_data

        if self.ref_data is not None and self.data is not None:
            assert len(self.ref_data) == len(self.data)
            self.data = [(ref_list, test_item) for ref_list, test_item in zip(self.ref_data, self.data)]

        # 数据的形式 self.data = [(references_list, test_data_item)]

    def create_one_sample(self, history=None, response=None, entities=None, raw_response=None):
        # history: [{'id':, 'Sentence': }, {}]
        # entities: {"Symptom":[]....}
        self.ref_data = [raw_response]

        item = dict()
        item['text'] = []
        item['text'].append(history)
        item_response = {
            "id": "Doctor",
            'Sentence': response,
        }
        item_response.update(entities)
        item['text'].append(item_response)

        self.data = [(raw_response, item)]

    def add_noise_entities(self, entities):
        # 减少
        result_entities = []
        for i in entities:
            if random.random() < 0.9:
                result_entities.append(i)

        # 增加
        sample_entities = random.sample(self.config['entity'], 3)
        add_prob = random.random()
        if add_prob < 0.08:
            result_entities.extend(sample_entities[:1])
        elif 0.08 < add_prob < 0.1:
            result_entities.extend(sample_entities[:2])
        result_entities = list(set(result_entities))

        # 乱序
        random.shuffle(result_entities)
        return result_entities

    def build_inputs_method_3(self, batch, add_entity_noise):
        history_ids = []
        combined_references_with_entities = []
        token_type_ids = []  # 用来区分reference 和 entities

        for references_list, item in batch:
            cur_history_ids = []
            cur_combined_references_with_entities = []
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

            target_entity = []
            for entity_type in self.entity_type:
                for e in item['text'][1][entity_type]:
                    target_entity.append(e)
            if add_entity_noise:  # train 和 dev 都用来训练实体预测了
                target_entity = self.add_noise_entities(target_entity)  # 加入噪音减少exposure bias
            target_entity = self.combine_entity_to_ids(target_entity)

            reference_sentence_ids = []
            random.shuffle(references_list)  # 避免references的顺序偏差
            for r in references_list:
                cur_sentence = self.build_sentence(r)
                reference_sentence_ids.append(cur_sentence)
            cur_combined_references_with_entities += [self.cls_idx]
            for i in reference_sentence_ids:
                cur_combined_references_with_entities = cur_combined_references_with_entities + i + [self.sep_idx]
                cur_combined_references_with_entities = cur_combined_references_with_entities[-512:]

            cur_token_type_ids = [0] * len(cur_combined_references_with_entities) + [1] * len(target_entity)
            cur_combined_references_with_entities += target_entity

            if len(target_entity) > 0:
                cur_combined_references_with_entities = cur_combined_references_with_entities[-512:]
                cur_token_type_ids = cur_token_type_ids[-512:]
            if len(cur_token_type_ids) != len(cur_combined_references_with_entities):
                print("\n err \n")
            history_ids.append(torch.tensor(cur_history_ids))
            token_type_ids.append(torch.tensor(cur_token_type_ids))
            combined_references_with_entities.append(torch.tensor(cur_combined_references_with_entities))

        return history_ids, token_type_ids, combined_references_with_entities

    def process_response(self, batch):
        response_ids = []
        for _, item in batch:
            response = item['text'][1]['Sentence']
            response = self.build_sentence(response)[:510]
            response = [_ for _ in response if _ != self.unk_idx]
            response = [self.cls_idx] + response + [self.sep_idx]
            response_ids.append(torch.tensor(response))
        return response_ids

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0):
        pad_idx = self.pad_idx

        reorganize_method_id = self.config['reorganize_method_id']

        def collate_fn_3(batch):
            history_ids, token_type_ids, combined_references_with_entities = self.build_inputs_method_3(
                batch, self.data_type == "train" and self.config['add_entity_noise'])
            history_ids = pad_sequence(history_ids, batch_first=True, padding_value=pad_idx)
            combined_references_with_entities = pad_sequence(combined_references_with_entities, batch_first=True,
                                                             padding_value=pad_idx)
            token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)

            history_mask = (history_ids != pad_idx).long()
            references_mask = (combined_references_with_entities != pad_idx).long()

            ret_data = {
                "history_ids": history_ids,
                "history_mask": history_mask,
                "references_with_entities": combined_references_with_entities,
                "references_mask": references_mask,
                "token_type_ids": token_type_ids,
            }

            response_ids = self.process_response(batch)
            response_ids = pad_sequence(response_ids, batch_first=True, padding_value=pad_idx)
            ret_data['response_ids'] = response_ids
            response_mask = (response_ids != pad_idx).long()
            ret_data['response_mask'] = response_mask

            if self.data_type == 'train' and self.config['entity_kl']:
                _, kl_target_token_type_ids, kl_target_combined_references_with_entities = self.build_inputs_method_3(
                    batch, False
                )
                kl_target_combined_references_with_entities = pad_sequence(
                    kl_target_combined_references_with_entities, batch_first=True, padding_value=pad_idx
                )
                kl_target_token_type_ids = pad_sequence(kl_target_token_type_ids, batch_first=True, padding_value=0)
                kl_target_references_mask = (kl_target_combined_references_with_entities != pad_idx).long()
                ret_data.update({
                    "kl_target_token_type_ids": kl_target_token_type_ids,
                    "kl_target_combined_references_with_entities": kl_target_combined_references_with_entities,
                    "kl_target_references_mask": kl_target_references_mask
                })
            return ret_data

        target_collate_fn = collate_fn_3

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
