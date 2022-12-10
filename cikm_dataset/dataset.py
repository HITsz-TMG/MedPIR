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
import json


class BaseDataset(data.Dataset, ABC):
    def __init__(self, original_data_path=None, vocab_path=None, data_path=None, data_type=None,
                 preprocess=False, config=None, data=None):
        super(BaseDataset, self).__init__()
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

        if preprocess:
            self.new_test_preprocessing(original_data_path)
            pickle.dump(self.data, open(data_path, 'wb'))
        else:
            if data_path is not None:
                if data_path.endswith(".json"):
                    self.data = json.load(open(data_path, 'r', encoding='utf-8'))
                else:
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
        examination = copy.deepcopy(temp['Examination'])
        del temp['Examination']
        temp['Test'] = examination
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

    def new_test_preprocessing(self, new_test_path):
        pkl_data = pickle.load(open(new_test_path, 'rb'))
        for dialog in tqdm(pkl_data, desc="data preprocessing"):
            history_text = [self.change_to_lower(dialog[0])]
            history = [self.cls_idx] + self.build_sentence(dialog[0]['Sentence'])[-510:] + [self.sep_idx]
            history_speaker = [self.spk2idx['Patients']] * len(history)
            for turn in dialog[1:]:
                turn = self.change_to_lower(turn)
                response = self.build_sentence(turn["Sentence"]) + [self.sep_idx]
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
                history_text.append(turn)

    def preprocessing(self, original_data_path):
        pkl_data = pickle.load(open(original_data_path, 'rb'))
        res_data = []
        for item in tqdm(pkl_data, desc="data preprocessing"):
            dialog = item['history']
            history_ids_list = []
            history_text = []
            for turn in dialog:
                history_ids_list.append(self.build_sentence(turn))
                history_text.append({
                    "id": None,
                    "Sentence": turn,
                    "Symptom": None,
                    "Medicine": None,
                    "Test": None,
                    "Attribute": None,
                    "Disease": None,
                })
            response_text = {
                "id": "Doctor",
                "Sentence": item['response'],
                "Symptom": None,
                "Medicine": None,
                "Test": None,
                "Attribute": None,
                "Disease": None,
            }
            history_ids = [self.cls_idx]
            for h in history_ids_list:
                history_ids = history_ids + [self.sep_idx] + h
            history_ids += [self.sep_idx]
            response = [self.cls_idx] + self.build_sentence(item["response"])[:510] + [self.sep_idx]
            item = {
                "history_ids": history_ids,
                "response_ids": response,
                "text": [deepcopy(history_text), response_text]
            }
            res_data.append(item)
        return res_data

    def convert_ids_to_tokens(self, ids):
        return [self.idx2token[i] for i in ids]

    def clip_doctor_long_sentence(self, sentence):
        expect_len = 50  # 长度1~50涵盖了95%的情况
        if len(sentence) > 50:
            all_dot_pos = []
            for idx, c in enumerate(sentence):
                if c in ['。', '！', '？', '；', ';', '!', '?']:
                    all_dot_pos.append(idx)
            if len(all_dot_pos) == 0:
                for idx, c in enumerate(sentence):
                    if c in ['，', ',']:
                        all_dot_pos.append(idx)
            if len(all_dot_pos) != 0:
                from_dot_to_expect = [(math.fabs(pos - expect_len), pos) for idx, pos in enumerate(all_dot_pos)]
                from_dot_to_expect = sorted(from_dot_to_expect, key=lambda x: x[0], reverse=False)
                split_pos = from_dot_to_expect[0][1]
            else:
                split_pos = expect_len

            if split_pos >= expect_len * 1.5:
                split_pos = int(expect_len * 1.5)

            sentence = sentence[:split_pos]
        # if len(sentence) > self.l:
        #     self.l = len(sentence)
        return sentence

    def history_clip_long_sentence(self, batch):
        history_ids = []
        history_spk = []
        for item in batch:
            cur_history_ids = []
            cur_history_spk = []
            history_sentence_ids = []
            history_sentence_spk = []
            dialog = item['text'][0]
            for h in dialog:
                text_sentence = h['Sentence']
                if h['id'] == 'Doctor':
                    text_sentence = self.clip_doctor_long_sentence(text_sentence)
                else:
                    text_sentence = text_sentence[-100:]
                cur_sentence = self.build_sentence(text_sentence)
                cur_spk = [self.spk2idx[h['id']]] * len(cur_sentence)
                history_sentence_ids.append(cur_sentence)
                history_sentence_spk.append(cur_spk)
            cur_history_ids += [self.cls_idx]
            cur_history_spk += [self.spk2idx['Patients']]
            for i, j in zip(history_sentence_ids, history_sentence_spk):
                cur_history_ids = cur_history_ids + i + [self.sep_idx]
                cur_history_ids = cur_history_ids[-512:]
                cur_history_spk = cur_history_spk + j + [j[0]]
                cur_history_spk = cur_history_spk[-512:]
            history_ids.append(torch.tensor(cur_history_ids))
            history_spk.append(torch.tensor(cur_history_spk))
        return history_ids, history_spk

    def history_base_sentence(self, batch):
        if self.config['use_token_type_ids'] is True:
            history_ids = []
            history_spk = []
            for item in batch:
                cur_history_ids = []
                cur_history_spk = []
                history_sentence_ids = []
                history_sentence_spk = []
                dialog = item['text'][0]
                for h in dialog:
                    text_sentence = h['Sentence']
                    cur_sentence = self.build_sentence(text_sentence)[-80:]
                    cur_spk = [self.spk2idx[h['id']]] * len(cur_sentence)
                    history_sentence_ids.append(cur_sentence)
                    history_sentence_spk.append(cur_spk)
                cur_history_ids += [self.cls_idx]
                cur_history_spk += [self.spk2idx['Patients']]
                for i, j in zip(history_sentence_ids, history_sentence_spk):
                    cur_history_ids = cur_history_ids + i + [self.sep_idx]
                    cur_history_ids = cur_history_ids[-512:]
                    cur_history_spk = cur_history_spk + j + [j[0]]
                    cur_history_spk = cur_history_spk[-512:]
                history_ids.append(torch.tensor(cur_history_ids))
                history_spk.append(torch.tensor(cur_history_spk))
            return history_ids, history_spk
        else:
            history_ids = []
            for item in batch:
                cur_history_ids = []
                history_sentence_ids = []
                dialog = item['text'][0]
                for h in dialog:
                    text_sentence = h['Sentence']
                    cur_sentence = self.build_sentence(text_sentence)[-80:]
                    history_sentence_ids.append(cur_sentence)
                cur_history_ids += [self.cls_idx]
                for i in history_sentence_ids:
                    cur_history_ids = cur_history_ids + i + [self.sep_idx]
                    cur_history_ids = cur_history_ids[-512:]
                history_ids.append(torch.tensor(cur_history_ids))
            return history_ids, None

    def combine_entity_to_ids(self, entities):
        if len(entities) == 0:
            return []
        res = []
        for i in entities:
            res.extend(
                [self.token2idx.get(c.lower(), self.unk_idx) for c in i] + [self.token2idx['、']]
            )
        res[-1] = self.sep_idx
        return res

    def history_with_entity_appendix(self, batch):
        history_ids = []
        token_type_ids = []
        for item in batch:
            cur_history_ids = []
            history_sentence_ids = []
            dialog = item['text'][0]
            for h in dialog:
                text_sentence = h['Sentence']
                if h['id'] == 'Doctor':
                    text_sentence = self.clip_doctor_long_sentence(text_sentence)
                else:
                    text_sentence = text_sentence[-100:]
                cur_sentence = self.build_sentence(text_sentence)
                history_sentence_ids.append(cur_sentence)

            cur_history_ids += [self.cls_idx]

            for i in history_sentence_ids:
                cur_history_ids = cur_history_ids + i + [self.sep_idx]
                cur_history_ids = cur_history_ids[-512:]

            target_entity = []
            for entity_type in self.entity_type:
                for e in item['text'][1][entity_type]:
                    target_entity.append(e)
            target_entity = self.combine_entity_to_ids(target_entity)

            cur_token_type_ids = [0] * len(cur_history_ids)
            if len(target_entity) > 0:
                cur_token_type_ids = (cur_token_type_ids + [1] * len(target_entity))[-512:]
                cur_history_ids = (cur_history_ids + target_entity)[-512:]

            history_ids.append(torch.tensor(cur_history_ids))
            token_type_ids.append(torch.tensor(cur_token_type_ids))

        return history_ids, token_type_ids

    def process_response(self, batch):
        response_ids = []
        for item in batch:
            response = item['text'][1]['Sentence']
            response = self.build_sentence(response)[:100]
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
        use_token_type_ids = self.config.get('use_token_type_ids', False)

        def base_collate_fn(batch):
            history_ids, history_spk = self.history_base_sentence(batch)
            history_ids = pad_sequence(history_ids, batch_first=True, padding_value=pad_idx)

            if history_spk is not None:
                history_spk = pad_sequence(history_spk, batch_first=True, padding_value=0)

            history_mask = (history_ids != pad_idx).long()

            ret_data = {
                "history_ids": history_ids,
                "history_mask": history_mask,
            }

            if use_token_type_ids is not False:
                ret_data["history_speaker"] = history_spk

            response_ids = self.process_response(batch)
            response_ids = pad_sequence(response_ids, batch_first=True, padding_value=pad_idx)
            ret_data['response_ids'] = response_ids
            response_mask = (response_ids != pad_idx).long()
            ret_data['response_mask'] = response_mask

            return ret_data

        def entity_appendix_collate_fn(batch):
            history_ids, history_spk = self.history_with_entity_appendix(batch)
            history_ids = pad_sequence(history_ids, batch_first=True, padding_value=pad_idx)
            # sentence and entities segments
            history_spk = pad_sequence(history_spk, batch_first=True, padding_value=0)
            history_mask = (history_ids != pad_idx).long()
            ret_data = {
                "history_ids": history_ids,
                "history_mask": history_mask,
                "history_speaker": history_spk
            }
            response_ids = self.process_response(batch)
            response_ids = pad_sequence(response_ids, batch_first=True, padding_value=pad_idx)
            ret_data['response_ids'] = response_ids
            response_mask = (response_ids != pad_idx).long()
            ret_data['response_mask'] = response_mask
            return ret_data

        if self.config['use_entity_appendix']:
            print("use entity appendix")
            target_collate_fn = entity_appendix_collate_fn
        else:
            target_collate_fn = base_collate_fn
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


def build_test():
    data_path = "../data/original/new_test.pk"
    # data_path = "./test.pk"
    test_data = pickle.load(open(data_path, 'rb'))

    history_combine = []
    for i in test_data:
        c_h = ""
        for h in i['history']:
            c_h += h
        history_combine.append(c_h)

    def get_reply(long_dialog, short_dialog):
        short_dialog_len = len(short_dialog['history'])
        if short_dialog_len >= len(long_dialog['history']):
            print("")
        reply = long_dialog['history'][short_dialog_len]
        return reply

    reply_dict = dict()
    from tqdm import tqdm

    for i in tqdm(range(len(history_combine))):
        hc_i = history_combine[i]
        for j in range(i + 1, len(history_combine)):
            hc_j = history_combine[j]
            if hc_i == hc_j:
                continue

            if hc_i.startswith(hc_j):
                reply_dict[j] = get_reply(
                    test_data[i], test_data[j]
                )
            elif hc_j.startswith(hc_i):
                reply_dict[i] = get_reply(
                    test_data[j], test_data[i]
                )

    print(len(reply_dict))

    leak_test_data = []
    leak_idx = sorted(list(reply_dict.keys()))
    print(leak_idx)
    for idx in leak_idx:
        item = test_data[idx]
        item['response'] = reply_dict[idx]
        leak_test_data.append(item)

        print(item)

    print(len(leak_test_data))
    pickle.dump(leak_test_data, open("../data/cikm/test-plain-4-25.pkl", 'wb'))

# if __name__ == '__main__':
# build_test()
# test_dataset = BaseDataset(vocab_path="../data/vocab.txt", data_type='test')
#
# tt = test_dataset.preprocessing("../data/cikm/test-plain-4-25.pkl")
#
# print(len(tt))
#
# pickle.dump(tt, open("../data/cikm/test-4-25-input.pkl", 'wb'))
