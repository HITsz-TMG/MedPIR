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


class MedDGDataset(data.Dataset, ABC):
    def __init__(self, original_data_path, vocab_path, data_path=None, data_type=None,
                 preprocess=False, config=None, tokenizer=None):
        super(MedDGDataset, self).__init__()

        # self.l = 0

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
            tmp_data = pickle.load(open(data_path, 'rb'))

            if self.config.get('model_name') == "NextEntityPredict":
                if data_type == 'train':
                    for i in tmp_data:
                        t_l = 0
                        for t in self.entity_type:
                            t_l += len(i['text'][1][t])
                        if t_l != 0 or random.random() < 0.3:
                            self.data.append(i)
                else:
                    self.data = tmp_data
            else:
                self.data = []
                if data_type == 'train':
                    #
                    if self.config.get('data_argument', False):
                        argument_data1 = pickle.load(open("./data/4-18-data/input/test2_for_train.pkl", 'rb'))
                        argument_data1 = [ad for ad in argument_data1 if self.select_to_argument(ad)]
                        random.shuffle(argument_data1)
                        argument_data1 = argument_data1[:int(len(argument_data1) / 2)]

                        argument_data2 = pickle.load(open("./data/4-18-data/test_old_for_train.pkl", 'rb'))
                        argument_data2 = [ad for ad in argument_data2 if self.select_to_argument(ad)]
                        random.shuffle(argument_data2)
                        argument_data2 = argument_data2[:int(len(argument_data2) / 2)]

                        argument_data3 = pickle.load(open("./data/4-28-data/spring.pkl", 'rb'))

                        tmp_data = tmp_data + argument_data1 + argument_data2 + argument_data3

                    pass_n = 0
                    for i in tmp_data:
                        if self.select_to_train(i):
                            self.data.append(i)
                        else:
                            pass_n += 1

                    print(pass_n)

                elif data_type == "dev" or self.config.get("running_task", "eval") == "eval":
                    self.data = tmp_data + pickle.load(open("./data/4-18-data/input/test2_for_dev.pkl", 'rb'))
                else:
                    self.data = tmp_data

        print(self.data_type + ": " + str(len(self.data)))
        # self.data = self.data[:10] + self.data[-10:]

    def select_to_train(self, dialog):
        turn_num = len(dialog['text'][0])
        if turn_num > 30:
            return False
        if turn_num <= 2:
            return False
        return True

    def select_to_argument(self, dialog):
        turn_num = len(dialog['text'][0])
        response_len = len(dialog['text'][1])
        if turn_num > 25:
            return False
        if turn_num <= 2:
            return False
        if response_len < 5:
            return False
        return True

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
                if isinstance(dialog[0], dict) and 'id' in dialog[0].keys():
                    history = [self.cls_idx]
                    history_speaker = [self.spk2idx['Patients']]
                    history_text = []
                    for turn in dialog:
                        cur_sentence = turn['Sentence']
                        cur_sentence = self._build_sentence(cur_sentence) + [self.sep_idx]
                        cur_speaker = [self.spk2idx[turn['id']]] * len(cur_sentence)
                        history = history + cur_sentence
                        history_speaker = history_speaker + cur_speaker
                        history_text.append(turn)
                    history = history[-512:]
                    history_speaker = history_speaker[-512:]
                    self.data.append({
                        "history_ids": list(history),
                        "history_speaker": list(history_speaker),
                        "text": [history_text, None]
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
                    turn = self.change_to_lower(turn)
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
                    history_text.append(turn)

    def convert_ids_to_tokens(self, ids):
        return [self.idx2token[i] for i in ids]

    def history_keep_first_sentence(self, batch):
        history_ids = []
        for item in batch:
            cur_history_ids = []
            history_sentence_ids = []
            dialog = item['text'][0]
            # convert each sentence to ids
            for h in dialog:
                cur_sentence = self._build_sentence(h['Sentence'])
                history_sentence_ids.append(cur_sentence)

            for h in history_sentence_ids[1:]:
                cur_history_ids = (cur_history_ids + h[-250:])[-511:] + [self.sep_idx]

            first_sentence = history_sentence_ids[0][-300:]
            first_sentence_len = len(first_sentence)
            other_len = 512 - 2 - first_sentence_len

            cur_history_ids = [self.cls_idx] + first_sentence + [self.sep_idx] + cur_history_ids[-other_len:]

            history_ids.append(torch.tensor(cur_history_ids))
        return history_ids

    def clip_doctor_long_sentence(self, sentence):
        expect_len = 50

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
            # convert each sentence to ids
            # estimate_len = len("".join([i['Sentence'] for i in dialog]))
            for h in dialog:
                text_sentence = h['Sentence']
                if h['id'] == 'Doctor':
                    text_sentence = self.clip_doctor_long_sentence(text_sentence)
                else:
                    text_sentence = text_sentence[-100:]
                cur_sentence = self._build_sentence(text_sentence)
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

    def combine_entity_to_ids(self, entities):
        if len(entities) == 0:
            return []
        res = []
        for i in entities:
            res.extend(
                [self.token2idx[c] for c in i] + [self.token2idx['、']]
            )
        res[-1] = self.sep_idx
        return res

    def history_clip_long_sentence_with_entity(self, batch):
        history_ids = []
        history_spk = []  # with entity 2
        for item in batch:
            cur_history_ids = []
            cur_history_spk = []
            history_sentence_ids = []
            history_sentence_spk = []
            history_sentence_entity = []
            dialog = item['text'][0]
            for h in dialog:
                text_sentence = h['Sentence']
                if h['id'] == 'Doctor':
                    text_sentence = self.clip_doctor_long_sentence(text_sentence)
                else:
                    text_sentence = text_sentence[-100:]
                cur_sentence_entity = []
                for entity_type in self.entity_type:
                    for e in h[entity_type]:
                        if e == "时长":
                            continue
                        cur_sentence_entity.append(e)
                cur_sentence_entity_ids = self.combine_entity_to_ids(cur_sentence_entity)
                cur_sentence = self._build_sentence(text_sentence)
                cur_spk = [self.spk2idx[h['id']]] * len(cur_sentence)
                history_sentence_ids.append(cur_sentence)
                history_sentence_spk.append(cur_spk)
                history_sentence_entity.append(cur_sentence_entity_ids)

            cur_history_ids += [self.cls_idx]
            cur_history_spk += [self.spk2idx['Patients']]
            for i, j, ent in zip(history_sentence_ids, history_sentence_spk, history_sentence_entity):
                cur_history_ids = cur_history_ids + i + [self.sep_idx]
                cur_history_spk = cur_history_spk + j + [j[0]]
                if len(ent) != 0:
                    cur_history_ids = cur_history_ids + ent  # ent 是自带sep的
                    cur_history_spk = cur_history_spk + [2] * len(ent)
                cur_history_ids = cur_history_ids[-512:]
                cur_history_spk = cur_history_spk[-512:]
            history_ids.append(torch.tensor(cur_history_ids))
            history_spk.append(torch.tensor(cur_history_spk))
        return history_ids, history_spk

    def history_clip_long_sentence_kl_div_entity(self, batch):
        history_ids = []
        history_spk = []  # with entity 2
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
                cur_sentence = self._build_sentence(text_sentence)
                cur_spk = [self.spk2idx[h['id']]] * len(cur_sentence)
                history_sentence_ids.append(cur_sentence)
                history_sentence_spk.append(cur_spk)

            target_entity = []
            for entity_type in self.entity_type:
                for e in item['text'][1][entity_type]:
                    if e == "时长":
                        continue
                    target_entity.append(e)
            target_entity = self.combine_entity_to_ids(target_entity)

            cur_history_ids += [self.cls_idx]
            cur_history_spk += [self.spk2idx['Patients']]

            for i, j in zip(history_sentence_ids, history_sentence_spk):
                cur_history_ids = cur_history_ids + i + [self.sep_idx]
                cur_history_spk = cur_history_spk + j + [j[0]]
                cur_history_ids = cur_history_ids[-512:]
                cur_history_spk = cur_history_spk[-512:]

            if len(target_entity) > 0:
                cur_history_ids = (cur_history_ids + target_entity)[-512:]
                cur_history_spk = (cur_history_spk + [self.spk2idx['Doctor']] * len(target_entity))[-512:]

            # print("".join(self.convert_ids_to_tokens(cur_history_ids)))
            # print(item['text'][1]['Sentence'])
            history_ids.append(torch.tensor(cur_history_ids))
            history_spk.append(torch.tensor(cur_history_spk))
        return history_ids, history_spk

    def history_for_rewrite_entity(self, batch):
        history_ids = []
        history_spk = []  # with entity 2
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
                cur_sentence = self._build_sentence(text_sentence)
                cur_spk = [self.spk2idx[h['id']]] * len(cur_sentence)
                history_sentence_ids.append(cur_sentence)
                history_sentence_spk.append(cur_spk)

            target_entity = []
            for entity_type in self.entity_type:
                for e in item['text'][1][entity_type]:
                    target_entity.append(e)

            # 加入噪音
            for e in self.idx2entity:
                if random.random() < 0.05:
                    target_entity.append(e)
                    print(e)
            target_entity = list(set(target_entity))
            random.shuffle(target_entity)

            target_entity = self.combine_entity_to_ids(target_entity)

            cur_history_ids += [self.cls_idx]
            cur_history_spk += [self.spk2idx['Patients']]

            for i, j in zip(history_sentence_ids, history_sentence_spk):
                cur_history_ids = cur_history_ids + i + [self.sep_idx]
                cur_history_spk = cur_history_spk + j + [j[0]]
                cur_history_ids = cur_history_ids[-512:]
                cur_history_spk = cur_history_spk[-512:]

            if len(target_entity) > 0:
                cur_history_ids = (cur_history_ids + target_entity)[-512:]
                cur_history_spk = (cur_history_spk + [self.spk2idx['Doctor']] * len(target_entity))[-512:]

            # print("".join(self.convert_ids_to_tokens(cur_history_ids)))
            # print(item['text'][1]['Sentence'])
            history_ids.append(torch.tensor(cur_history_ids))
            history_spk.append(torch.tensor(cur_history_spk))
        return history_ids, history_spk

    def two_stage_entity_to_history(self, batch):
        history_ids = []
        history_spk = []  # with entity 2
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
                cur_sentence = self._build_sentence(text_sentence)
                cur_spk = [self.spk2idx[h['id']]] * len(cur_sentence)
                history_sentence_ids.append(cur_sentence)
                history_sentence_spk.append(cur_spk)

            target_entity = []
            if self.data_type == "train":
                for entity_type in self.entity_type:
                    for e in item['text'][1][entity_type]:
                        if random.random() > 0.4:
                            target_entity.append(e)
            else:
                for entity_type in self.entity_type:
                    for e in item['text'][1][entity_type]:
                        target_entity.append(e)

            target_entity = self.combine_entity_to_ids(target_entity)

            cur_history_ids += [self.cls_idx]
            cur_history_spk += [self.spk2idx['Patients']]

            for i, j in zip(history_sentence_ids, history_sentence_spk):
                cur_history_ids = cur_history_ids + i + [self.sep_idx]
                cur_history_spk = cur_history_spk + j + [j[0]]
                cur_history_ids = cur_history_ids[-512:]
                cur_history_spk = cur_history_spk[-512:]

            if len(target_entity) > 0:
                cur_history_ids = (cur_history_ids + target_entity)[-512:]
                cur_history_spk = (cur_history_spk + [self.spk2idx['Doctor']] * len(target_entity))[-512:]

            # print("".join(self.convert_ids_to_tokens(cur_history_ids)))
            # print(item['text'][1]['Sentence'])
            history_ids.append(torch.tensor(cur_history_ids))
            history_spk.append(torch.tensor(cur_history_spk))
        return history_ids, history_spk

    def process_response(self, batch):
        response_ids = []
        add_entity_lm = self.config.get('add_entity_lm', False)
        if add_entity_lm:
            for item in batch:
                entity = item["text"][1]["Symptom"] + item["text"][1]["Medicine"] + item["text"][1]["Test"] + \
                         item["text"][1]["Attribute"] + item["text"][1]["Disease"]
                entity_str = "、".join(entity)
                entity_ids = self._build_sentence(entity_str)
                response = item['text'][1]['Sentence']
                response = self._build_sentence(response)
                response = [_ for _ in response if _ != self.unk_idx]
                entity_response = (entity_ids + [self.response] + response)[:510]
                entity_response = [self.cls_idx] + entity_response + [self.sep_idx]

                response_ids.append(torch.tensor(entity_response))
        else:
            for item in batch:
                response = item['text'][1]['Sentence']
                response = self._build_sentence(response)[:510]
                response = [_ for _ in response if _ != self.unk_idx]
                response = [self.cls_idx] + response + [self.sep_idx]
                response_ids.append(torch.tensor(response))
        return response_ids

    def data_argument(self):
        test_data = pickle.load(open("../data/change/test_add_spk.pkl", 'rb'))
        # test_data = pickle.load(open(self.config['test_data_path'], 'rb'))
        # assert isinstance(test_data[0][0], dict) and 'id' in test_data[0]['text'][0].keys()

        clean_test_data = []

        print(len(test_data))
        idx_history = []
        for idx, item in enumerate(test_data):

            item_history = ""
            for h in item['text'][0]:
                item_history += h['Sentence']

            idx_history.append((idx, item_history))

        idx_history = sorted(idx_history, key=lambda x: len(x[1]))

        for i, (idx, history) in tqdm(enumerate(idx_history), desc='去重', total=len(idx_history)):
            add_this = True
            for j in range(i + 1, len(idx_history)):
                if idx_history[j][1].startswith(history):
                    add_this = False
                    break
            if add_this:
                clean_test_data.append(test_data[idx])

        print(len(clean_test_data))

        test_data = clean_test_data

        additional_data = []

        def add_empty_entity(d):
            d['Symptom'] = []
            d["Medicine"] = []
            d["Test"] = []
            d["Attribute"] = []
            d["Disease"] = []
            return d

        for dialog in tqdm(test_data):
            dialog = dialog['text'][0]
            history_text = [add_empty_entity(dialog[0])]
            history = [self.cls_idx] + self._build_sentence(dialog[0]['Sentence'])[-510:] + [self.sep_idx]
            history_speaker = [self.spk2idx['Patients']] * len(history)
            for turn in dialog[1:]:
                turn = add_empty_entity(turn)
                response = self._build_sentence(turn["Sentence"]) + [self.sep_idx]
                response_text = turn
                item = {
                    "history_speaker": history_speaker[-512:],
                    "history_ids": history[-512:],
                    "response_ids": [self.cls_idx] + response[:511],
                    "text": [deepcopy(history_text), response_text],
                    # "mask_entity": True,
                }
                if self.entity2idx is not None:
                    entity_label = len(self.entity2idx) * [0]
                    for entity_type in self.entity_type:
                        for e in turn[entity_type]:
                            e_id = self.entity2idx[e]
                            entity_label[e_id] = 1
                    item['entity_label'] = entity_label
                if turn["id"] == "Doctor":
                    additional_data.append(item)
                speaker = [self.spk2idx[turn["id"]]] * len(response)
                history_speaker = list(history_speaker + speaker)
                history = list(history + response)
                history_text.append(turn)

        print(len(additional_data))
        pickle.dump(additional_data, open("../data/4-18-data/test_old_for_train.pkl", 'wb'))

        # random.shuffle(additional_data)
        # pickle.dump(additional_data[:500], open("../data/4-18-data/input/test2_for_dev.pkl", 'wb'))
        # pickle.dump(additional_data[500:], open("../data/4-18-data/input/test2_for_train.pkl", 'wb'))

        self.data.extend(additional_data)
        print("data argument")
        print("")

    def chunks_clip(self, chunk, start_end):
        start, end = start_end
        if len(chunk) >= 512:
            if end > 512:
                print("!!!")
            else:
                chunk = chunk[:512]
        return chunk, start_end

    def sliding_window_one_sample(self, history_sentence_ids, win_size=5):
        chunks = []
        start_end = []
        if len(history_sentence_ids) <= win_size:
            cur_chunk = []
            cur_chunk.append(self.cls_idx)
            for h in history_sentence_ids:
                cur_chunk = cur_chunk + h + [self.sep_idx]
            start_end.append((0, len(cur_chunk)))
            chunks.append(cur_chunk)
        else:
            cur_chunk = []
            half_win_size = math.ceil(win_size / 2)
            first_chunk_needed_turn_num = half_win_size
            cur_chunk.append(self.cls_idx)
            first_chunk_end = 1  # 1: cls
            for idx in range(0, win_size):
                cur_chunk = cur_chunk + history_sentence_ids[idx] + [self.sep_idx]
            for idx in range(0, first_chunk_needed_turn_num):
                first_chunk_end = first_chunk_end + len(history_sentence_ids[idx]) + 1  # 1: sep
            chunks.append(list(cur_chunk))
            start_end.append((0, first_chunk_end))

            for idx in range(1, len(history_sentence_ids) - win_size):
                start, end, cur_len = 0, 0, 0
                cur_chunk = []
                for n in range(win_size):
                    if n == win_size // 2:
                        start = cur_len
                        end = cur_len + len(history_sentence_ids[idx + n]) + 1
                    cur_len = cur_len + len(history_sentence_ids[idx + n]) + 1
                    cur_chunk = cur_chunk + history_sentence_ids[idx + n] + [self.sep_idx]
                chunks.append(list(cur_chunk))
                start_end.append((start, end))

            cur_chunk = []
            start, end, cur_len = 0, 0, 0
            for e_idx, idx in enumerate(range(len(history_sentence_ids) - win_size, len(history_sentence_ids))):
                if e_idx == win_size // 2:
                    start = cur_len
                cur_len = cur_len + len(history_sentence_ids[idx]) + 1
                cur_chunk = cur_chunk + history_sentence_ids[idx] + [self.sep_idx]
            chunks.append(list(cur_chunk))
            start_end.append((start, cur_len))

        for idx in range(len(chunks)):
            if len(chunks[idx]) > 512:
                chunks[idx] = chunks[idx][512:]
                if start_end[idx][1] > 512:
                    start_end[idx] = (start_end[idx][0], 512)
                    assert start_end[idx][1] > start_end[idx][0]

        return chunks, start_end

    def split_text_sentence(self, sentence, recursion_time=0):
        if len(sentence) > 150:
            all_dot_pos = []
            half_len = len(sentence) / 2
            for idx, c in enumerate(sentence):
                if c in ['。', '！', '？', '；', ';', '!', '?']:
                    all_dot_pos.append(idx)

            if len(all_dot_pos) == 0:
                for idx, c in enumerate(sentence):
                    if c in ['，', ',']:
                        all_dot_pos.append(idx)

            from_dot_to_half = [(math.fabs(pos - half_len), pos) for idx, pos in enumerate(all_dot_pos)]
            from_dot_to_half = sorted(from_dot_to_half, key=lambda x: x[0], reverse=False)
            dis_to_split_pos = from_dot_to_half[0][0]
            half_split_pos = from_dot_to_half[0][1]

            if dis_to_split_pos > 100:
                for idx, c in enumerate(sentence):
                    if c in ['，', ',']:
                        all_dot_pos.append(idx)
                from_dot_to_half = [(math.fabs(pos - half_len), pos) for idx, pos in enumerate(all_dot_pos)]
                from_dot_to_half = sorted(from_dot_to_half, key=lambda x: x[0], reverse=False)
                half_split_pos = from_dot_to_half[0][1]

            if recursion_time > 7:
                return [sentence[:half_split_pos + 1] + sentence[half_split_pos + 1:]]
            else:
                return self.split_text_sentence(sentence[:half_split_pos + 1], recursion_time=recursion_time + 1) + \
                       self.split_text_sentence(sentence[half_split_pos + 1:], recursion_time=recursion_time + 1)
        else:
            return [sentence]

    def history_self_attention_by_sliding_window(self, batch, window_size=7):
        all_chunks, all_start_end, combine_num = [], [], []
        for item in batch:
            history_sentence_ids = []
            dialog = item['text'][0]
            # convert each sentence to ids

            for idx, h in enumerate(dialog):
                sentence = h['Sentence'][:100]
                if h['id'] == 'Doctor':
                    sentence = self.clip_doctor_long_sentence(sentence)
                sentences = self.split_text_sentence(sentence)
                for tmp_s in sentences:
                    cur_sentence = self._build_sentence(tmp_s)
                    history_sentence_ids.append(cur_sentence)
                # spk = h['id']
                # if idx == 0:
                #     history_sentence_ids.append(cur_sentence[:300])
                # else:
                #     if spk == 'Patients':
                #         history_sentence_ids.append(cur_sentence[:200])
                #     else:
                #         history_sentence_ids.append(cur_sentence[:150])

            if sum(len(sent) for sent in history_sentence_ids) < self.config.get('slide_threshold', 400):
                single_chunk = [self.cls_idx]
                for sent_ids in history_sentence_ids:
                    single_chunk = single_chunk + sent_ids + [self.sep_idx]
                all_chunks.append([single_chunk])
                all_start_end.append([(0, len(single_chunk))])
            else:
                chunks, start_end = self.sliding_window_one_sample(history_sentence_ids, win_size=window_size)
                all_chunks.append(chunks)
                all_start_end.append(start_end)

        return all_chunks, all_start_end

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
        data_type = self.data_type
        use_token_type_ids = self.config.get('use_token_type_ids', False)
        entity_attention = self.config.get('entity_attention', False)
        entity_predict = self.config.get("entity_predict", False)
        use_random_remove_short = True if self.data_type == "train" and self.config[
            'model_name'] != 'NextEntityPredict' else False
        print("use random_remove_short_response: " + str(use_random_remove_short))

        # if self.config.get('data_argument', False):
        #     self.data_argument()

        def random_remove_short_dialog(batch):
            estimate_response_len = [len(i['text'][1]['Sentence']) for i in batch]
            new_batch = []
            for idx in range(len(batch)):
                if estimate_response_len[idx] <= 5:
                    if random.random() < 0.6:
                        new_batch.append(batch[idx])
                elif estimate_response_len[idx] <= 7:
                    if random.random() < 0.8:
                        new_batch.append(batch[idx])
                else:
                    new_batch.append(batch[idx])
            batch = new_batch if len(new_batch) > 0 else batch
            return batch

        def slide_window_collate_fn(batch):
            if data_type == 'train':
                batch = random_remove_short_dialog(batch)

            bs = len(batch)
            all_chunks, all_start_end = self.history_self_attention_by_sliding_window(
                batch, window_size=self.config.get('slide_window_size')
            )
            assert len(all_chunks) == len(all_start_end)
            assert all(len(all_chunks[i]) == len(all_start_end[i]) for i in range(len(batch)))

            each_sample_chunks_num = [len(i) for i in all_chunks]

            combine_chunks = []
            for i in all_chunks:
                combine_chunks.extend(i)

            combine_start_end = []
            for i in all_start_end:
                combine_start_end.extend(i)

            chunks_expand_as_batch_size = []
            start_end_expand_as_batch_size = []
            for i in range(math.ceil(len(combine_chunks) / bs)):
                chunks_expand_as_batch_size.append(combine_chunks[i * bs:(i + 1) * bs])
                start_end_expand_as_batch_size.append(combine_start_end[i * bs:(i + 1) * bs])

            group_history_ids = []
            group_history_mask = []
            for i in chunks_expand_as_batch_size:
                cur_group = [torch.tensor(j) for j in i]
                cur_group = pad_sequence(cur_group, batch_first=True, padding_value=pad_idx)
                group_history_ids.append(cur_group)
                cur_mask = (cur_group != pad_idx).long()
                group_history_mask.append(cur_mask)

            ret_data = {
                "group_history_ids": group_history_ids,
                "group_history_mask": group_history_mask,
                "start_end_expand_as_batch_size": start_end_expand_as_batch_size,
                "each_sample_chunks_num": each_sample_chunks_num,
            }
            if self.data_type != "train":
                history_ids = [torch.tensor(i['history_ids']) for i in batch]
                history_ids = pad_sequence(history_ids, batch_first=True, padding_value=pad_idx)
                ret_data['history_ids'] = history_ids

            if entity_attention and data_type != 'test':
                # ret_data["entity_label"] = torch.tensor([i['entity_label'] for i in batch], dtype=torch.float)
                ret_data["entity_label"] = torch.tensor(
                    self.get_entities_label(batch), dtype=torch.float
                )

            if data_type != 'test':
                response_ids = self.process_response(batch)
                response_ids = pad_sequence(response_ids, batch_first=True, padding_value=pad_idx)
                ret_data['response_ids'] = response_ids
                response_mask = (response_ids != pad_idx).long()
                ret_data['response_mask'] = response_mask

            return ret_data

        def clip_long_sentence_collate_fn(batch):
            if use_random_remove_short:
                batch = random_remove_short_dialog(batch)

            if self.config.get('sentence_add_entity', False) is True:
                history_ids, history_spk = self.history_clip_long_sentence_with_entity(batch)
            elif self.config.get('two_stage', False) is True:
                history_ids, history_spk = self.two_stage_entity_to_history(batch)
            else:
                history_ids, history_spk = self.history_clip_long_sentence(batch)
            history_ids = pad_sequence(history_ids, batch_first=True, padding_value=pad_idx)
            history_spk = pad_sequence(history_spk, batch_first=True, padding_value=0)

            history_mask = (history_ids != pad_idx).long()
            ret_data = {
                "history_ids": history_ids,
                "history_mask": history_mask,
            }

            if use_token_type_ids is not False:
                ret_data["history_speaker"] = history_spk

            if (entity_attention or entity_predict) and data_type != 'test':
                # ret_data["entity_label"] = torch.tensor([i['entity_label'] for i in batch], dtype=torch.float)
                ret_data["entity_label"] = torch.tensor(
                    self.get_entities_label(batch), dtype=torch.float
                )

            if data_type != 'test':
                response_ids = self.process_response(batch)
                response_ids = pad_sequence(response_ids, batch_first=True, padding_value=pad_idx)
                ret_data['response_ids'] = response_ids
                response_mask = (response_ids != pad_idx).long()
                ret_data['response_mask'] = response_mask

            if self.config.get("kl_div", False) is True:
                kl_history_ids, kl_history_spk = self.history_clip_long_sentence_kl_div_entity(batch)
                kl_history_ids = pad_sequence(kl_history_ids, batch_first=True, padding_value=pad_idx)
                kl_history_spk = pad_sequence(kl_history_spk, batch_first=True, padding_value=0)
                kl_history_mask = (kl_history_ids != pad_idx).long()
                ret_data['kl_history_ids'] = kl_history_ids
                ret_data['kl_history_spk'] = kl_history_spk
                ret_data['kl_history_mask'] = kl_history_mask

            return ret_data

        if self.config.get("slide_window", False):
            target_collate_fn = slide_window_collate_fn
        elif self.config.get("clip_long_sentence", False):
            target_collate_fn = clip_long_sentence_collate_fn
        else:
            raise NotImplementedError

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


class SpkPredictDataset(data.Dataset, ABC):
    def __init__(self, original_data_path, vocab_path, data_path=None, data_type=None, preprocess=False, config=None):
        super(SpkPredictDataset, self).__init__()
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
        self.idx2spk = {0: "Patients", 1: "Doctor"}
        self.data: List[Dict] = []
        if preprocess:
            self._preprocessing_2(original_data_path)
            pickle.dump(self.data, open(data_path, 'wb'))
        else:
            self.data = pickle.load(open(data_path, 'rb'))

        self.config = dict(config) if config is not None else None

    def _build_sentence(self, text_sentence):
        text_sentence = text_sentence.replace(' ', '').replace("\n", ''). \
            replace('\t', '').replace('\u3000', '').replace('\u00A0', '')
        return [self.token2idx.get(i, self.unk_idx) for i in text_sentence]

    def build_sentence(self, text_sentence):
        text_sentence = text_sentence.replace(' ', '').replace("\n", ''). \
            replace('\t', '').replace('\u3000', '').replace('\u00A0', '')
        return [self.token2idx.get(i, self.unk_idx) for i in text_sentence]

    def _preprocessing(self, original_data_path):
        pkl_data = pickle.load(open(original_data_path, 'rb'))

        for dialog in tqdm(pkl_data, desc="data preprocessing"):
            last_sentence = self._build_sentence(dialog[0]['Sentence'])[-250:]
            last_spk = dialog[0]["id"]
            for idx, turn in enumerate(dialog[1:]):
                sentence = self._build_sentence(turn["Sentence"])[-250:]
                input_ids = [self.cls_idx] + last_sentence + \
                            [self.sep_idx] + sentence + [self.sep_idx]
                token_type_ids = [0] * (len(last_sentence) + 2) + [1] * (len(sentence) + 1)
                cur_spk = turn["id"]
                label = 1 if cur_spk == last_spk else 0
                item = {
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "label": label
                }
                self.data.append(item)
                last_spk = str(cur_spk)
                last_sentence = list(sentence)

    def _preprocessing_2(self, original_data_path):
        pkl_data = pickle.load(open(original_data_path, 'rb'))
        for dialog in tqdm(pkl_data, desc="data preprocessing"):
            history = self._build_sentence(dialog[0]['Sentence'])
            for idx, turn in enumerate(dialog[1:]):
                sentence = self._build_sentence(turn["Sentence"])[-250:]

                input_ids = [self.cls_idx] + sentence + \
                            [self.sep_idx] + history[-(512 - len(sentence) - 3):] + [self.sep_idx]
                token_type_ids = [0] * (len(sentence) + 2)
                token_type_ids = token_type_ids + [1] * (len(input_ids) - len(token_type_ids))
                item = {
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "label": self.spk2idx[turn["id"]]
                }
                self.data.append(item)

                history = list(history + [self.sep_idx] + sentence)

    def convert_ids_to_tokens(self, ids):
        return [self.idx2token[i] for i in ids]

    def get_dataloader(self, batch_size, shuffle=True, num_workers=4):
        pad_idx = self.pad_idx
        data_type = self.data_type

        def collate_fn(batch):
            input_ids = [torch.tensor(i['input_ids']) for i in batch]
            token_type_ids = [torch.tensor(i['token_type_ids']) for i in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_idx)
            token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=pad_idx)
            attention_mask = (input_ids != pad_idx).long()
            label = torch.tensor([i['label'] for i in batch])

            return {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "label": label
            }

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


class EntityPredictDataset(data.Dataset, ABC):
    def __init__(self, original_data_path, vocab_path, entity_path=None,
                 data_path=None, data_type=None, preprocess=False, config=None):
        super(EntityPredictDataset, self).__init__()
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
        self.symptom = ['乏力', '体重下降', '便血', '反流', '发热', '口苦', '吞咽困难', '呕吐', '呕血', '呼吸困难', '咳嗽', '咽部灼烧感', '咽部痛', '喷嚏',
                        '嗜睡', '四肢麻木', '头晕', '头痛', '寒战', '尿急', '尿频', '心悸', '恶心', '打嗝', '排气', '月经紊乱', '有痰', '气促', '水肿',
                        '消化不良', '淋巴结肿大', '烧心', '焦躁', '痉挛', '痔疮', '稀便', '粘便', '精神不振', '细菌感染', '肌肉酸痛', '肛周疼痛', '肠梗阻',
                        '肠鸣', '胃痛', '胃肠不适', '胃肠功能紊乱', '背痛', '胸痛', '脱水', '腹泻', '腹痛', '腹胀', '菌群失调', '螺旋杆菌感染', '贫血', '过敏',
                        '里急后重', '食欲不振', '饥饿感', '黄疸', '黑便', '鼻塞']
        self.medicine = ['三九胃泰', '乳果糖', '乳酸菌素', '人参健脾丸', '健胃消食片', '健脾丸', '克拉霉素', '兰索拉唑', '吗丁啉', '嗜酸乳杆菌',
                         '四磨汤口服液', '培菲康', '复方消化酶', '多潘立酮', '多酶片', '奥美', '山莨菪碱', '左氧氟沙星', '布洛芬', '康复新液',
                         '开塞露', '得舒特', '思密达', '思连康', '抗生素', '整肠生', '斯达舒', '曲美布汀', '泌特', '泮托拉唑', '消炎利胆片',
                         '瑞巴派特', '甲硝唑', '益生菌', '硫糖铝', '磷酸铝', '耐信', '肠溶胶囊', '肠胃康', '胃复安', '胃苏颗粒',
                         '胶体果胶铋', '莫沙比利', '蒙脱石散', '藿香正气丸', '补脾益肠丸', '诺氟沙星胶囊', '谷氨酰胺肠溶胶囊', '达喜', '金双歧', '铝碳酸镁', '阿莫西林',
                         '雷贝拉唑', '颠茄片', '香砂养胃丸', '马来酸曲美布丁']
        self.test = ['b超', 'ct', '便常规', '呼气实验', '小肠镜', '尿常规', '尿检', '糖尿病', '结肠镜',
                     '肛门镜', '肝胆胰脾超声', '肠镜', '胃蛋白酶', '胃镜', '胶囊内镜', '腹腔镜', '腹部彩超', '血常规',
                     '转氨酶', '钡餐']
        self.attribute = ['位置', '性质', '时长', '诱因']
        self.disease = ['便秘', '感冒', '肝硬化', '肠易激综合征', '肠炎', '肺炎', '胃溃疡', '胃炎', '胆囊炎', '胰腺炎', '阑尾炎', '食管炎']
        self.entity_type_num = {
            'Symptom': len(self.symptom),
            'Medicine': len(self.medicine),
            'Test': len(self.test),
            'Attribute': len(self.attribute),
            'Disease': len(self.disease)
        }

        self.symptom2idx = {k: v for v, k in enumerate(self.symptom)}
        self.medicine2idx = {k: v for v, k in enumerate(self.medicine)}
        self.test2idx = {k: v for v, k in enumerate(self.test)}
        self.attribute2idx = {k: v for v, k in enumerate(self.attribute)}
        self.disease2idx = {k: v for v, k in enumerate(self.disease)}

        # self.ent2idx = dict()
        # self.idx2ent = dict()
        # with open(entity_path, 'r', encoding='utf-8') as reader:
        #     for idx, token in enumerate(reader.readlines()):
        #         entity = token.strip()
        #         self.ent2idx[entity] = idx
        #         self.idx2ent[idx] = entity
        # self.entity_label_keys = ["label{}".format(e_idx) for e_idx in range(len(self.ent2idx))]
        # print("entity num: {}".format(len(self.entity_label_keys)))

        if preprocess:
            self._preprocessing(original_data_path)
            pickle.dump(self.data, open(data_path, 'wb'))
        else:
            self.data = pickle.load(open(data_path, 'rb'))

        self.config = dict(config) if config is not None else None

    def _build_sentence(self, text_sentence):
        text_sentence = text_sentence.replace(' ', '').replace("\n", ''). \
            replace('\t', '').replace('“', '"').replace('”', '"').replace('\u3000', '').replace('\u00A0', '')
        text_sentence = text_sentence.lower()

        return [self.token2idx.get(i, self.unk_idx) for i in text_sentence]

    def _preprocessing(self, original_data_path):
        pkl_data = pickle.load(open(original_data_path, 'rb'))

        for dialog in tqdm(pkl_data, desc="data preprocessing"):

            all_turn = dialog['text'][0] + [dialog['text'][1]]

            for idx, turn in enumerate(all_turn):
                sentence = self._build_sentence(turn["Sentence"])[-510:]
                input_ids = [self.cls_idx] + sentence + [self.sep_idx]

                symptom_label = [0] * len(self.symptom)
                for symptom in turn['Symptom']:
                    symptom_id = self.symptom2idx[symptom]
                    symptom_label[symptom_id] = 1

                medicine_label = [0] * len(self.medicine)
                for medicine in turn['Medicine']:
                    medicine_id = self.medicine2idx[medicine]
                    medicine_label[medicine_id] = 1

                test_label = [0] * len(self.test)
                for test in turn['Test']:
                    test_id = self.test2idx[test]
                    test_label[test_id] = 1

                attribute_label = [0] * len(self.attribute)
                for attribute in turn['Attribute']:
                    attribute_id = self.attribute2idx[attribute]
                    attribute_label[attribute_id] = 1

                disease_label = [0] * len(self.disease)
                for disease in turn['Disease']:
                    disease_id = self.disease2idx[disease]
                    disease_label[disease_id] = 1

                item = {
                    "input_ids": input_ids,
                    # "label": label,
                    "symptom_label": symptom_label,
                    "medicine_label": medicine_label,
                    "test_label": test_label,
                    "attribute_label": attribute_label,
                    "disease_label": disease_label
                }

                self.data.append(item)

    def convert_ids_to_tokens(self, ids):
        return [self.idx2token[i] for i in ids]

    def get_dataloader(self, batch_size, shuffle=True, num_workers=4):
        pad_idx = self.pad_idx
        data_type = self.data_type

        # entity_label_keys = self.entity_label_keys

        def collate_fn(batch):
            input_ids = [torch.tensor(i['input_ids']) for i in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_idx)

            attention_mask = (input_ids != pad_idx).long()

            # entity_labels = dict((k, []) for k in entity_label_keys)
            # for k in entity_label_keys:
            #     for i in batch:
            #         entity_labels[k].append(i[k])
            #     entity_labels[k] = torch.tensor(entity_labels[k], dtype=torch.float)
            ret_data = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            symptom_label = torch.stack([torch.tensor(i['symptom_label']) for i in batch])
            medicine_label = torch.stack([torch.tensor(i['medicine_label']) for i in batch])
            test_label = torch.stack([torch.tensor(i['test_label']) for i in batch])
            attribute_label = torch.stack([torch.tensor(i['attribute_label']) for i in batch])
            disease_label = torch.stack([torch.tensor(i['disease_label']) for i in batch])

            label = torch.cat([symptom_label, medicine_label, test_label, attribute_label, disease_label], dim=-1)

            ret_data.update({
                "label": label,
                "symptom_label": symptom_label,
                "medicine_label": medicine_label,
                "test_label": test_label,
                "attribute_label": attribute_label,
                "disease_label": disease_label
            })

            # ret_data.update(entity_labels)
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


class NextEntityPredictDataset(data.Dataset, ABC):
    def __init__(self, original_data_path, vocab_path, entity_path=None,
                 data_path=None, data_type=None, preprocess=False, config=None):
        super(NextEntityPredictDataset, self).__init__()
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
        self.entity_type_num = {'Symptom': 62, 'Medicine': 57, 'Test': 20, 'Attribute': 4, 'Disease': 12}

        self.symptom = ['月经紊乱', '呕血', '肠梗阻', '淋巴结肿大', '吞咽困难', '胸痛', '咽部痛', '咳嗽', '脱水', '乏力', '肛周疼痛', '菌群失调', '螺旋杆菌感染',
                        '嗜睡', '食欲不振', '腹泻', '尿频', '胃肠功能紊乱', '有痰', '气促', '打嗝', '胃痛', '烧心', '鼻塞', '腹痛', '反流', '肠鸣', '水肿',
                        '痔疮', '发热', '精神不振', '贫血', '稀便', '消化不良', '尿急', '焦躁', '里急后重', '细菌感染', '喷嚏', '恶心', '呼吸困难', '黄疸',
                        '呕吐', '粘便', '寒战', '胃肠不适', '便血', '头晕', '口苦', '体重下降', '腹胀', '黑便', '过敏', '背痛', '痉挛', '头痛', '咽部灼烧感',
                        '排气', '肌肉酸痛', '四肢麻木', '心悸', '饥饿感']

        self.medicine = ['肠溶胶囊', '藿香正气丸', '阿莫西林', '消炎利胆片', '思连康', '人参健脾丸', '乳酸菌素', '诺氟沙星', '山莨菪碱', '达喜', '得舒特', '瑞巴派特',
                         '吗丁啉', '金双歧', '肠胃康', '马来酸曲美布丁', '泌特', '乳果糖', '磷酸铝', '左氧氟沙星', '康复新液', '多潘立酮', '胶体果胶铋', '莫沙比利',
                         '泮托拉唑', '思密达', '颠茄片', '复方消化酶', '斯达舒', '香砂养胃丸', '铝碳酸镁', '多酶片', '曲美布汀', '克拉霉素', '开塞露', '健脾丸',
                         '硫糖铝', '蒙脱石散', '嗜酸乳杆菌', '胃苏', '益生菌', '抗生素', '布洛芬', '耐信', '谷氨酰胺肠溶胶囊', '补脾益肠丸', '健胃消食片', '兰索拉唑',
                         '果胶铋', '雷呗', '四磨汤', '胃复安', '甲硝唑', '培菲康', '三九胃泰', '奥美', '整肠生']

        self.test = ['肠镜', 'CT', '腹部彩超', '糖尿病', 'B超', '尿检', '转氨酶', '小肠镜', '血常规', '胃镜', '肝胆胰脾超声', '胶囊内镜', '结肠镜', '胃蛋白酶',
                     '呼气实验', '肛门镜', '钡餐', '尿常规', '腹腔镜', '便常规']

        self.attribute = ['位置', '性质', '时长', '诱因']

        self.disease = ['胃溃疡', '食管炎', '阑尾炎', '肝硬化', '胰腺炎', '肠易激综合征', '肺炎', '肠炎', '胆囊炎', '感冒', '胃炎', '便秘']
        self.symptom2idx = {k: v for v, k in enumerate(self.symptom)}
        self.medicine2idx = {k: v for v, k in enumerate(self.medicine)}
        self.test2idx = {k: v for v, k in enumerate(self.test)}
        self.attribute2idx = {k: v for v, k in enumerate(self.attribute)}
        self.disease2idx = {k: v for v, k in enumerate(self.disease)}

        if preprocess:
            self._preprocessing(original_data_path)
            pickle.dump(self.data, open(data_path, 'wb'))
        else:
            self.data = pickle.load(open(data_path, 'rb'))

        self.config = dict(config) if config is not None else None

    def _build_sentence(self, text_sentence):
        text_sentence = text_sentence.replace(' ', '').replace("\n", ''). \
            replace('\t', '').replace('\u3000', '').replace('\u00A0', '')
        return [self.token2idx.get(i, self.unk_idx) for i in text_sentence]

    def _preprocessing(self, original_data_path):
        pkl_data = pickle.load(open(original_data_path, 'rb'))

        for dialog in tqdm(pkl_data, desc="data preprocessing"):
            history = [self.cls_idx] + self._build_sentence(dialog[0]['Sentence'])[-510:] + [self.sep_idx]
            history_speaker = [self.spk2idx['Patients']] * len(history)
            for idx, turn in enumerate(dialog[1:]):
                response = self._build_sentence(turn["Sentence"]) + [self.sep_idx]
                symptom_label = [0] * len(self.symptom)
                for symptom in turn['Symptom']:
                    symptom_id = self.symptom2idx[symptom]
                    symptom_label[symptom_id] = 1

                medicine_label = [0] * len(self.medicine)
                for medicine in turn['Medicine']:
                    medicine_id = self.medicine2idx[medicine]
                    medicine_label[medicine_id] = 1

                test_label = [0] * len(self.test)
                for test in turn['Test']:
                    test_id = self.test2idx[test]
                    test_label[test_id] = 1

                attribute_label = [0] * len(self.attribute)
                for attribute in turn['Attribute']:
                    attribute_id = self.attribute2idx[attribute]
                    attribute_label[attribute_id] = 1

                disease_label = [0] * len(self.disease)
                for disease in turn['Disease']:
                    disease_id = self.disease2idx[disease]
                    disease_label[disease_id] = 1

                item = {
                    "input_ids": history[-512:],
                    "token_type_ids": history_speaker[-512:],
                    # "label": label,
                    "symptom_label": symptom_label,
                    "medicine_label": medicine_label,
                    "test_label": test_label,
                    "attribute_label": attribute_label,
                    "disease_label": disease_label
                }

                if turn['id'] == "Doctor":
                    self.data.append(item)
                speaker = [self.spk2idx[turn["id"]]] * len(response)
                history_speaker = list(history_speaker + speaker)
                history = list(history + response)

    def convert_ids_to_tokens(self, ids):
        return [self.idx2token[i] for i in ids]

    def get_dataloader(self, batch_size, shuffle=True, num_workers=4):
        pad_idx = self.pad_idx
        data_type = self.data_type

        def collate_fn(batch):
            input_ids = [torch.tensor(i['input_ids']) for i in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_idx)
            token_type_ids = [torch.tensor(i['token_type_ids']) for i in batch]
            token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
            attention_mask = (input_ids != pad_idx).long()

            symptom_label = torch.stack([torch.tensor(i['symptom_label']) for i in batch])
            medicine_label = torch.stack([torch.tensor(i['medicine_label']) for i in batch])
            test_label = torch.stack([torch.tensor(i['test_label']) for i in batch])
            attribute_label = torch.stack([torch.tensor(i['attribute_label']) for i in batch])
            disease_label = torch.stack([torch.tensor(i['disease_label']) for i in batch])

            label = torch.cat([symptom_label, medicine_label, test_label, attribute_label, disease_label], dim=-1)

            ret_data = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "label": label,

                "symptom_label": symptom_label,
                "medicine_label": medicine_label,
                "test_label": test_label,
                "attribute_label": attribute_label,
                "disease_label": disease_label
            }
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


class ResultEntityPredictDataset(data.Dataset, ABC):
    def __init__(self, predict_data_path, vocab_path, entity_path=None,
                 dev_data_path=None, data_path=None, data_type=None, preprocess=False, config=None):
        super(ResultEntityPredictDataset, self).__init__()
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
        self.symptom = ['乏力', '体重下降', '便血', '反流', '发热', '口苦', '吞咽困难', '呕吐', '呕血', '呼吸困难', '咳嗽', '咽部灼烧感', '咽部痛', '喷嚏',
                        '嗜睡', '四肢麻木', '头晕', '头痛', '寒战', '尿急', '尿频', '心悸', '恶心', '打嗝', '排气', '月经紊乱', '有痰', '气促', '水肿',
                        '消化不良', '淋巴结肿大', '烧心', '焦躁', '痉挛', '痔疮', '稀便', '粘便', '精神不振', '细菌感染', '肌肉酸痛', '肛周疼痛', '肠梗阻',
                        '肠鸣', '胃痛', '胃肠不适', '胃肠功能紊乱', '背痛', '胸痛', '脱水', '腹泻', '腹痛', '腹胀', '菌群失调', '螺旋杆菌感染', '贫血', '过敏',
                        '里急后重', '食欲不振', '饥饿感', '黄疸', '黑便', '鼻塞']
        self.medicine = ['三九胃泰', '乳果糖', '乳酸菌素', '人参健脾丸', '健胃消食片', '健脾丸', '克拉霉素', '兰索拉唑', '吗丁啉', '嗜酸乳杆菌',
                         '四磨汤口服液', '培菲康', '复方消化酶', '多潘立酮', '多酶片', '奥美', '山莨菪碱', '左氧氟沙星', '布洛芬', '康复新液',
                         '开塞露', '得舒特', '思密达', '思连康', '抗生素', '整肠生', '斯达舒', '曲美布汀', '泌特', '泮托拉唑', '消炎利胆片',
                         '瑞巴派特', '甲硝唑', '益生菌', '硫糖铝', '磷酸铝', '耐信', '肠溶胶囊', '肠胃康', '胃复安', '胃苏颗粒',
                         '胶体果胶铋', '莫沙比利', '蒙脱石散', '藿香正气丸', '补脾益肠丸', '诺氟沙星胶囊', '谷氨酰胺肠溶胶囊', '达喜', '金双歧', '铝碳酸镁', '阿莫西林',
                         '雷贝拉唑', '颠茄片', '香砂养胃丸', '马来酸曲美布丁']
        self.test = ['b超', 'ct', '便常规', '呼气实验', '小肠镜', '尿常规', '尿检', '糖尿病', '结肠镜',
                     '肛门镜', '肝胆胰脾超声', '肠镜', '胃蛋白酶', '胃镜', '胶囊内镜', '腹腔镜', '腹部彩超', '血常规',
                     '转氨酶', '钡餐']
        self.attribute = ['位置', '性质', '时长', '诱因']
        self.disease = ['便秘', '感冒', '肝硬化', '肠易激综合征', '肠炎', '肺炎', '胃溃疡', '胃炎', '胆囊炎', '胰腺炎', '阑尾炎', '食管炎']
        self.entity_type_num = {
            'Symptom': len(self.symptom),
            'Medicine': len(self.medicine),
            'Test': len(self.test),
            'Attribute': len(self.attribute),
            'Disease': len(self.disease)
        }

        self.symptom2idx = {k: v for v, k in enumerate(self.symptom)}
        self.medicine2idx = {k: v for v, k in enumerate(self.medicine)}
        self.test2idx = {k: v for v, k in enumerate(self.test)}
        self.attribute2idx = {k: v for v, k in enumerate(self.attribute)}
        self.disease2idx = {k: v for v, k in enumerate(self.disease)}

        # self.ent2idx = dict()
        # self.idx2ent = dict()
        # with open(entity_path, 'r', encoding='utf-8') as reader:
        #     for idx, token in enumerate(reader.readlines()):
        #         entity = token.strip()
        #         self.ent2idx[entity] = idx
        #         self.idx2ent[idx] = entity
        # self.entity_label_keys = ["label{}".format(e_idx) for e_idx in range(len(self.ent2idx))]
        # print("entity num: {}".format(len(self.entity_label_keys)))

        if preprocess:
            self._preprocessing(predict_data_path, dev_data_path)
            pickle.dump(self.data, open(data_path, 'wb'))
        else:
            self.data = pickle.load(open(data_path, 'rb'))

        self.config = dict(config) if config is not None else None

    def _build_sentence(self, text_sentence):
        text_sentence = text_sentence.replace(' ', '').replace("\n", ''). \
            replace('\t', '').replace('“', '"').replace('”', '"').replace('\u3000', '').replace('\u00A0', '')
        text_sentence = text_sentence.lower()

        return [self.token2idx.get(i, self.unk_idx) for i in text_sentence]

    def _preprocessing(self, predict_data_path, dev_data_path):
        predict_data = pickle.load(open(predict_data_path, 'rb'))
        dev_data = pickle.load(open(dev_data_path, 'rb'))

        for data_ in tqdm(zip(predict_data, dev_data), desc="data preprocessing"):

            pre_sentence = data_[0]
            sentence = self._build_sentence(pre_sentence)[-510:]
            input_ids = [self.cls_idx] + sentence + [self.sep_idx]
            Symptom = data_[1]["text"][1]['Symptom']
            Medicine = data_[1]["text"][1]['Medicine']
            Test = data_[1]["text"][1]['Test']
            Attribute = data_[1]["text"][1]['Attribute']
            Disease = data_[1]["text"][1]['Disease']

            symptom_label = [0] * len(self.symptom)
            for symptom in Symptom:
                symptom_id = self.symptom2idx[symptom]
                symptom_label[symptom_id] = 1

            medicine_label = [0] * len(self.medicine)
            for medicine in Medicine:
                medicine_id = self.medicine2idx[medicine]
                medicine_label[medicine_id] = 1

            test_label = [0] * len(self.test)
            for test in Test:
                test_id = self.test2idx[test]
                test_label[test_id] = 1

            attribute_label = [0] * len(self.attribute)
            for attribute in Attribute:
                attribute_id = self.attribute2idx[attribute]
                attribute_label[attribute_id] = 1

            disease_label = [0] * len(self.disease)
            for disease in Disease:
                disease_id = self.disease2idx[disease]
                disease_label[disease_id] = 1
            item = {
                "input_ids": input_ids,
                # "label": label,
                "symptom_label": symptom_label,
                "medicine_label": medicine_label,
                "test_label": test_label,
                "attribute_label": attribute_label,
                "disease_label": disease_label
            }

            self.data.append(item)

    def convert_ids_to_tokens(self, ids):
        return [self.idx2token[i] for i in ids]

    def get_dataloader(self, batch_size, shuffle=True, num_workers=4):
        pad_idx = self.pad_idx
        data_type = self.data_type

        # entity_label_keys = self.entity_label_keys

        def collate_fn(batch):
            input_ids = [torch.tensor(i['input_ids']) for i in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_idx)

            attention_mask = (input_ids != pad_idx).long()

            # entity_labels = dict((k, []) for k in entity_label_keys)
            # for k in entity_label_keys:
            #     for i in batch:
            #         entity_labels[k].append(i[k])
            #     entity_labels[k] = torch.tensor(entity_labels[k], dtype=torch.float)
            ret_data = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            symptom_label = torch.stack([torch.tensor(i['symptom_label']) for i in batch])
            medicine_label = torch.stack([torch.tensor(i['medicine_label']) for i in batch])
            test_label = torch.stack([torch.tensor(i['test_label']) for i in batch])
            attribute_label = torch.stack([torch.tensor(i['attribute_label']) for i in batch])
            disease_label = torch.stack([torch.tensor(i['disease_label']) for i in batch])

            label = torch.cat([symptom_label, medicine_label, test_label, attribute_label, disease_label], dim=-1)

            ret_data.update({
                "label": label,
                "symptom_label": symptom_label,
                "medicine_label": medicine_label,
                "test_label": test_label,
                "attribute_label": attribute_label,
                "disease_label": disease_label
            })

            # ret_data.update(entity_labels)
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

