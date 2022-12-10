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


class DialogueSummaryDataset(BaseDataset):
    def __init__(
            self,
            vocab_path=None,
            data_path=None,
            data_type=None,
            config=None,
            data=None,
            create_one=False,
    ):
        super(DialogueSummaryDataset, self).__init__(
            vocab_path=vocab_path,
            data_path=data_path,
            data_type=data_type,
            config=config,
            data=data,
        )

        if create_one:
            return

        self.summary = []
        self.summary_end_idx = self.token2idx['[SummaryEnd]']

        # self.preprocessing_dialogue_and_summary()
        # print(len(self.data))

    def preprocessing_dialogue_and_summary(self):
        all_data = []
        raw_data = pickle.load(open("../data/aaai/filtered_with_summary.pkl", 'rb'))

        for item in raw_data:
            cur_data = {"dialogue": [], 'summary': None}
            if '病情描述' in item['description']:
                cur_data['dialogue'].append({
                    "id": "Patients",
                    "Sentence": item['description']['病情描述']
                })

            for h in item['dialogue']:
                if h['Speaker'] == "病人：":
                    spk = "Patients"
                elif h['Speaker'] == "医生：":
                    spk = 'Doctor'
                else:
                    print(h['Speaker'])
                    raise ValueError
                if len(h['Sentence']) > 0:
                    cur_data['dialogue'].append({
                        "id": spk,
                        "Sentence": h['Sentence']
                    })

            summary = ""
            if "病情摘要及初步印象" in item['summary'] and item['summary']['病情摘要及初步印象'] != '':
                last_chr = item['summary']['病情摘要及初步印象'][-1]
                if last_chr in ['？', '?', '！', '!', '。', '.']:
                    s = item['summary']['病情摘要及初步印象']
                else:
                    s = item['summary']['病情摘要及初步印象'] + "。"
                summary += "摘要：" + s
            if "总结建议" in item['summary'] and item['summary']['总结建议'] != '':
                summary += "建议：" + item['summary']['总结建议']
            cur_data['summary'] = summary

            if len(cur_data['summary']) == 0 or len(cur_data['dialogue']) == 0:
                continue
            else:
                all_data.append(cur_data)
        self.data = all_data
        pickle.dump(all_data, open("../data/aaai/dialogue_summary.pkl", 'wb'))
        return all_data

    def encoder_inputs(self, batch):
        history_ids = []
        history_spk = []
        for item in batch:
            cur_history_ids = []
            cur_history_spk = []
            history_sentence_ids = []
            history_sentence_spk = []
            if 'dialogue' in item.keys():
                dialog = item['dialogue']
            else:
                dialog = item['text'][0]
            for h in dialog:
                text_sentence = h['Sentence']
                cur_sentence = self.build_sentence(text_sentence)[:80]
                cur_spk = [self.spk2idx[h['id']]] * len(cur_sentence)
                history_sentence_ids.append(cur_sentence)
                history_sentence_spk.append(cur_spk)
            cur_history_ids += [self.cls_idx]
            cur_history_spk += [self.spk2idx[dialog[0]['id']]]
            assert len(history_sentence_ids) == len(history_sentence_spk)
            for i, j in zip(history_sentence_ids, history_sentence_spk):
                cur_history_ids = cur_history_ids + i + [self.sep_idx]
                cur_history_ids = cur_history_ids[-512:]
                cur_history_spk = cur_history_spk + j + [j[0]]
                cur_history_spk = cur_history_spk[-512:]
            history_ids.append(torch.tensor(cur_history_ids))
            history_spk.append(torch.tensor(cur_history_spk))
        return history_ids, history_spk

    def decoder_inputs(self, batch):
        targets = []
        for item in batch:
            target = item['summary']
            target = self.build_sentence(target)
            target = [_ for _ in target if _ != self.unk_idx]
            target = [self.cls_idx] + target[:510] + [self.sep_idx]
            targets.append(torch.tensor(target))
        return targets

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0):
        pad_idx = self.pad_idx
        data_type = self.data_type

        def collate_fn(batch):
            history_ids, history_spk = self.encoder_inputs(batch)
            history_ids = pad_sequence(history_ids, batch_first=True, padding_value=pad_idx)
            history_spk = pad_sequence(history_spk, batch_first=True, padding_value=0)
            history_mask = (history_ids != pad_idx).long()
            ret_data = {
                "history_ids": history_ids,
                "history_mask": history_mask,
                "history_spk": history_spk,
            }
            if data_type != 'test':
                target_ids = self.decoder_inputs(batch)
                target_ids = pad_sequence(target_ids, batch_first=True, padding_value=pad_idx)
                target_mask = (target_ids != pad_idx).long()
                ret_data.update({
                    "target_ids": target_ids,
                    "target_mask": target_mask,
                })
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


if __name__ == '__main__':
    a = DialogueSummaryDataset(
        vocab_path="../data/vocab.txt",
        data_path="../data/aaai/dialogue_summary.pkl",
        data_type="train",
    )
    random.shuffle(a.data)
    dev = a.data[:2000]
    train = a.data[2000:]
    pickle.dump(dev, open("../data/aaai/dialogue_summary_dev.pkl", 'wb'))
    pickle.dump(train, open("../data/aaai/dialogue_summary_train.pkl", 'wb'))

    # cnt = 0
    # for dd in a.get_dataloader(batch_size=2, shuffle=False):
    #     """
    #     ret_data = {
    #             "history_ids": history_ids,
    #             "history_mask": history_mask,
    #             "history_spk": history_spk,
    #             "target_ids": target_ids,
    #             "target_mask": target_mask,
    #         }
    #     """
    #     h = dd["history_ids"][0].tolist()
    #     print("".join(a.convert_ids_to_tokens(h)))
    #     print(str(dd['history_spk'][0].tolist()))
    #     print("".join(a.convert_ids_to_tokens(dd['target_ids'][0].tolist())))
    #     cnt += 1
