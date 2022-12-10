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


class MedDGDataset(data.Dataset, ABC):
    def __init__(self, original_data_path, vocab_path, data_path=None, data_type=None,
                 preprocess=False, config=None, tokenizer=None):
        super(MedDGDataset, self).__init__()
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
        self.response = self.token2idx['[response]']
        self.spk2idx = {"Patients": 0, "Doctor": 1}
        self.data: List[Dict] = []
        self.entity_type = ['Symptom', 'Medicine', 'Test', 'Attribute', 'Disease']

        if config.get("entity", None) is not None:
            self.entity2idx = {e: idx for idx, e in enumerate(config['entity'])}
            self.idx2entity = {idx: e for idx, e in enumerate(config['entity'])}
        else:
            self.entity2idx, self.idx2entity = None, None

        if preprocess:
            self._preprocessing(original_data_path)
            pickle.dump(self.data, open(data_path, 'wb'))
        else:
            self.data = pickle.load(open(data_path, 'rb'))[:100]
            data_type = self.data_type
            add_entity_lm = self.config.get('add_entity_lm', False)
            if data_type != 'test':
                if add_entity_lm is True:
                    response_ids = self.add_lm_entity_predict(self.data)
                    for i in range(len(self.data)):
                        self.data[i]["response_ids"] = response_ids[i]

    def _build_sentence(self, text_sentence):
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

    def _preprocessing(self, original_data_path):
        pkl_data = pickle.load(open(original_data_path, 'rb'))
        for dialog in tqdm(pkl_data, desc="data preprocessing"):
            if self.data_type == "test":
                if 'id' in dialog.keys():
                    history = [self.cls_idx]
                    history_speaker = [self.spk2idx['Patients']]
                    for turn in dialog:
                        cur_sentence = turn['Sentence']
                        cur_sentence = self._build_sentence(cur_sentence) + [self.sep_idx]
                        cur_speaker = [self.spk2idx[turn['id']]] * len(cur_sentence)

                        history = history + cur_sentence
                        history_speaker = history_speaker + cur_speaker
                    history = history[-512:]
                    history_speaker = history_speaker[-512:]
                    self.data.append({
                        "history_ids": list(history),
                        "history_speaker": list(history_speaker)
                    })
                else:

                    history = [self.cls_idx]
                    history_speaker = [self.spk2idx['Patients']]
                    for h_id, h in enumerate(dialog['history']):
                        cur_h = self._build_sentence(h)
                        cut_pos = len(history) + len(cur_h) - 511
                        if cut_pos > 0:
                            history = history[cut_pos:]
                            history_speaker = history_speaker[cut_pos:]
                        history = history + cur_h + [self.sep_idx]
                        spk_id = self.spk2idx['Patients'] if h_id % 2 == 0 else self.spk2idx['Doctor']
                        history_speaker = history_speaker + [spk_id] * (len(cur_h) + 1)
                    self.data.append({
                        "history_speaker": list(history_speaker)[-512:],
                        "history_ids": list(history)[-512:],
                        "text": dialog['history']
                    })
            else:
                history_text = [self.change_to_lower(dialog[0])]
                history = [self.cls_idx] + self._build_sentence(dialog[0]['Sentence'])[-510:] + [self.sep_idx]
                history_speaker = [self.spk2idx['Patients']] * len(history)
                for turn in dialog[1:]:
                    response = self._build_sentence(turn["Sentence"]) + [self.sep_idx]
                    response_text = turn
                    item = {
                        "history_speaker": history_speaker[-512:],
                        "history_ids": history[-512:],
                        "response_ids": [self.cls_idx] + response[:511],
                        "text": [deepcopy(history_text), response_text]
                    }

                    if self.entity2idx is not None:
                        entity_label = len(self.entity2idx) * [0]
                        for entity_type in self.entity_type:
                            for e in turn[entity_type]:
                                e_id = self.entity2idx[e]
                                entity_label[e_id] = 1
                        item['entity_label'] = entity_label
                    if turn["id"] == "Doctor":
                        self.data.append(item)
                        # print("".join(self.convert_ids_to_tokens(self.data[-1])))
                    speaker = [self.spk2idx[turn["id"]]] * len(response)
                    history_speaker = list(history_speaker + speaker)
                    history = list(history + response)
                    history_text.append(self.change_to_lower(turn))

    def convert_ids_to_tokens(self, ids):
        return [self.idx2token[i] for i in ids]

    def add_lm_entity_predict(self, batch):
        response_ids = []
        for i in batch:
            entity = i["text"][1]["Symptom"] + i["text"][1]["Medicine"] + i["text"][1]["Test"] + i["text"][1][
                "Attribute"] + i["text"][1]["Disease"]
            entity_str = "、".join(entity)
            entity_ids = self._build_sentence(entity_str)
            jisuan = 512 - len([self.cls_idx] + entity_ids + [self.response])
            temp = i['response_ids'][1:]
            response_ids.append([self.cls_idx] + entity_ids + [self.response] + temp[-jisuan:])

        return response_ids

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0):
        pad_idx = self.pad_idx
        data_type = self.data_type
        use_token_type_ids = self.config.get('use_token_type_ids', False)
        entity_attention = self.config.get('entity_attention', False)

        def collate_fn(batch):
            history_ids = [torch.tensor(i['history_ids']) for i in batch]
            history_ids = pad_sequence(history_ids, batch_first=True, padding_value=pad_idx)
            history_mask = (history_ids != pad_idx).long()
            ret_data = {
                "history_ids": history_ids,
                "history_mask": history_mask,
            }
            if use_token_type_ids is not False:
                history_speaker = [torch.tensor(i['history_speaker']) for i in batch]
                history_speaker = pad_sequence(history_speaker, batch_first=True, padding_value=0)
                ret_data["history_speaker"] = history_speaker

            if entity_attention and data_type != 'test':
                ret_data["entity_label"] = torch.tensor([i['entity_label'] for i in batch], dtype=torch.float)

            if data_type != 'test':
                response_ids = [torch.tensor(i['response_ids']) for i in batch]
                response_ids = pad_sequence(response_ids, batch_first=True, padding_value=pad_idx)
                response_mask = (response_ids != pad_idx).long()
                ret_data['response_ids'] = response_ids
                ret_data['response_mask'] = response_mask
                # else:
                #     response_ids = self.add_lm_entity_predict(batch)
                #     response_ids = pad_sequence(response_ids, batch_first=True, padding_value=pad_idx)
                #     response_mask = (response_ids != pad_idx).long()
                #     ret_data['response_ids'] = response_ids
                #     ret_data['response_mask'] = response_mask
            return ret_data

        return data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class AddEntityDataset(data.Dataset, ABC):
    def __init__(self, original_data_path, vocab_path, data_path=None, data_type=None,
                 preprocess=False, config=None):
        super(AddEntityDataset, self).__init__()
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
        if preprocess:
            self._preprocessing(original_data_path)
            pickle.dump(self.data, open(data_path, 'wb'))
        else:
            self.data = pickle.load(open(data_path, 'rb'))

    def _build_sentence(self, text_sentence):
        text_sentence = text_sentence.replace(' ', '').replace("\n", ''). \
            replace('\t', '').replace('\u3000', '').replace('\u00A0', '')
        return [self.token2idx.get(i, self.unk_idx) for i in text_sentence]

    def _preprocessing(self, original_data_path):
        pkl_data = pickle.load(open(original_data_path, 'rb'))
        for dialog in tqdm(pkl_data, desc="data preprocessing"):
            if self.data_type == "test":
                if 'id' in dialog[0].keys():
                    history = [self.cls_idx]
                    history_speaker = [self.spk2idx['Patients']]
                    for turn in dialog:
                        cur_sentence = turn['Sentence']
                        cur_sentence = self._build_sentence(cur_sentence) + [self.sep_idx]
                        cur_speaker = [self.spk2idx[turn['id']]] * len(cur_sentence)

                        history = history + cur_sentence
                        history_speaker = history_speaker + cur_speaker
                    history = history[-512:]
                    history_speaker = history_speaker[-512:]
                    self.data.append({
                        "history_ids": list(history),
                        "history_speaker": list(history_speaker)
                    })
                else:
                    history = [self.cls_idx]
                    history_speaker = [self.spk2idx['Patients']]
                    for h_id, h in enumerate(dialog['history']):
                        cur_h = self._build_sentence(h)
                        cut_pos = len(history) + len(cur_h) - 511
                        if cut_pos > 0:
                            history = history[cut_pos:]
                            history_speaker = history_speaker[cut_pos:]
                        history = history + cur_h + [self.sep_idx]
                        spk_id = self.spk2idx['Patients'] if h_id % 2 == 0 else self.spk2idx['Doctor']
                        history_speaker = history_speaker + [spk_id] * (len(cur_h) + 1)
                    self.data.append({
                        "history_speaker": list(history_speaker)[-512:],
                        "history_ids": list(history)[-512:]
                    })
            else:
                history = [self.cls_idx] + self._build_sentence(dialog[0]['Sentence'])[-510:] + [self.sep_idx]
                for turn in dialog[1:]:
                    history_len = len(history)
                    entity = turn['Symptom'] + turn['Medicine'] + turn['Test'] + turn['Attribute'] + turn['Disease']
                    entity = self._build_sentence(','.join(entity)) + [self.sep_idx]
                    entity_len = len(entity)
                    history_entity = history + entity
                    token_type_ids = [0] * history_len + [1] * entity_len
                    response = self._build_sentence(turn["Sentence"]) + [self.sep_idx]
                    item = {
                        "token_type_ids": token_type_ids[-512:],
                        "history_ids": history_entity[-512:],
                        "response_ids": [self.cls_idx] + response[:511],
                    }
                    if turn["id"] == "Doctor":
                        self.data.append(item)
                        # print("".join(self.convert_ids_to_tokens(self.data[-1])))
                    history = list(history + response)

    def convert_ids_to_tokens(self, ids):
        return [self.idx2token[i] for i in ids]

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0):
        pad_idx = self.pad_idx
        data_type = self.data_type
        use_token_type_ids = self.config.get('use_token_type_ids', True)

        def collate_fn(batch):
            history_ids = [torch.tensor(i['history_ids']) for i in batch]
            token_type_ids = [torch.tensor(i['token_type_ids']) for i in batch]
            history_ids = pad_sequence(history_ids, batch_first=True, padding_value=pad_idx)
            token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
            history_mask = (history_ids != pad_idx).long()
            ret_data = {
                "history_ids": history_ids,
                "history_mask": history_mask,
            }
            if use_token_type_ids is not False:
                ret_data["token_type_ids"] = token_type_ids

            if data_type != 'test':
                response_ids = [torch.tensor(i['response_ids']) for i in batch]
                response_ids = pad_sequence(response_ids, batch_first=True, padding_value=pad_idx)
                response_mask = (response_ids != pad_idx).long()
                ret_data['response_ids'] = response_ids
                ret_data['response_mask'] = response_mask

            return ret_data

        return data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

