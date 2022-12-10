from abc import ABC
import torch
from src.utils.gpt_modeling import TransformerEncoder, TransformerDecoderLM
from torch import nn
from src.utils.hugging_face_bert import BertModel
from src.utils.hugging_face_gpt import GPT2Model
from src.utils.hugging_face_gpt import Attention
from transformers import BertConfig, GPT2Config
import torch.nn.functional as F
from src.model import BERTGPTEntity
import json
from typing import Any, Tuple
from cikm_dataset.summary_response_dataset import SummaryResponseDataset
from main_config import config
import pickle
from tqdm import tqdm


class Similar(nn.Module, ABC):
    def __init__(self, encoder=None):
        super().__init__()
        self.config = dict(config)

        if config.get('pcl_encoder_predict_entity', False) or encoder is None:
            self.encoder_config = BertConfig.from_json_file(config["pretrained_encoder_config_path"])
            self.encoder = BertModel(config=self.encoder_config,
                                     add_pooling_layer=False,
                                     decoder_like_gpt=False)
            res = self.encoder.load_state_dict(self._load_modified_state_dict(), strict=False)
            assert res.missing_keys == ['embeddings.position_ids'] and res.unexpected_keys == []
            self.usl_pcl_encoder = True
        else:
            self.usl_pcl_encoder = False
            self.encoder = encoder

        self.token2idx = dict()
        self.idx2token = dict()
        with open(config['vocab_path'], 'r', encoding='utf-8') as reader:
            for idx, token in enumerate(list(reader.readlines())):
                token = token.strip()
                self.token2idx[token] = idx
                self.idx2token[idx] = token

        self.all_linked_inputs = []
        self.unk_idx = self.token2idx['[UNK]']
        self.sep_idx = self.token2idx['[SEP]']
        self.cls_idx = self.token2idx['[CLS]']
        self.pad_idx = self.token2idx['[PAD]']
        self.entity_sep_id = self.token2idx['[entities]']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _load_modified_state_dict(self):
        state_dict = torch.load(self.config["pretrained_state_dict_path"], map_location='cpu')
        new_state_dict = []
        for k, v in state_dict.items():
            if k.startswith("bert.pooler") or k.startswith("cls"):
                continue
            elif k.startswith("bert."):
                new_state_dict.append((k[5:], v))
            else:
                new_state_dict.append((k, v))
        return dict(new_state_dict)

    def forward(self, input_ids=None, attention_mask=None, start_end_pos=None):
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        encoded_history = encoder_output[0]

        assert encoded_history.shape[0] == 1
        encoded_history = encoded_history[0]  # seq_len * hidden

        sent_hidden = []

        for start, end in start_end_pos:
            sent_hidden.append(encoded_history[start:end].sum(dim=0) / (end - start))

        return sent_hidden


def get_summary(dataset, device, model):
    # summary = []
    summary_and_idx = []
    for item in tqdm(dataset.get_dataloader(batch_size=1, shuffle=False, for_sim=True), total=len(dataset)):
        item['input_ids'] = item['input_ids'].to(device)
        sent_hidden = model(input_ids=item['input_ids'], start_end_pos=item['start_end_pos'])
        assert len(sent_hidden) >= 2
        response_vec = sent_hidden[-1]
        sim_score = []
        for sent_vec in sent_hidden[:-1]:
            sim_score.append(torch.cosine_similarity(sent_vec, response_vec, dim=0))
        sim_score = [(idx, i if len(sim_score) - idx <= 6 else -10000) for idx, i in enumerate(sim_score)]
        sim_score_sort = sorted(sim_score, key=lambda x: x[1], reverse=True)
        top_3 = sim_score_sort[:3]
        top_3_sort_idx_score = sorted(top_3, key=lambda x: x[0])
        top_3_idx = [i[0] for i in top_3_sort_idx_score]

        cur_summary = []
        for i in top_3_idx:
            cur_summary.append(item['save_origin_sent'][i])
        summary_and_idx.append((cur_summary, top_3_idx))
    #     summary.append(cur_summary)

    return summary_and_idx


@torch.no_grad()
def main():
    model = Similar()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    t = "test"

    test_dataset = SummaryResponseDataset(
        vocab_path=config['vocab_path'],
        data_path=config['{}_data_path'.format(t)],
        summary_data_path=config['dialogue_summary_{}_path'.format(t)],
        data_type="test",
        config=config,
        with_entity=False,
        with_summary=False,
        decoder_with_entity=False,
    )
    summary = get_summary(test_dataset, device, model)
    pickle.dump(summary, open("{}-sim-based-summary-with-idx.pkl".format(t), 'wb'))

    # train_dataset = SummaryResponseDataset(
    #     vocab_path=config['vocab_path'],
    #     data_path=config['train_data_path'],
    #     summary_data_path=config['dialogue_summary_train_path'],
    #     data_type="train",
    #     config=config,
    #     with_entity=False,
    #     with_summary=False,
    #     decoder_with_entity=False,
    # )
    # summary = get_summary(train_dataset, device, model)
    # pickle.dump(summary, open("dev-sim-based-summary.pkl", 'wb'))


@torch.no_grad()
def get_MedDialog_summary(t, prefix, data_path):
    model = Similar()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    test_dataset = SummaryResponseDataset(
        vocab_path=config['vocab_path'],
        data_path=data_path,
        data_type=t,
        config=config,
        with_entity=False,
        with_summary=False,
        decoder_with_entity=False,
        dataset_name="MedDialog",
        for_sim=True
    )
    summary = get_summary(test_dataset, device, model)
    pickle.dump(summary, open("./MedDialog/filtered_MedDialog/{}_summary.pkl".format(prefix), 'wb'))


if __name__ == '__main__':
    # main()
    # get_MedDialog_summary('dev', 'dev-last6', "./MedDialog/filtered_MedDialog/dev_pairs.pkl")
    get_MedDialog_summary('train', 'train-last6', "./MedDialog/filtered_MedDialog/train_pairs.pkl")
    # get_MedDialog_summary('test', 'test-last6', "./MedDialog/filtered_MedDialog/test_pairs.pkl")
