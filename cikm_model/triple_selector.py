from abc import ABC
import torch
from torch import nn
from src.utils.hugging_face_bert import BertModel
from transformers import BertConfig
import torch.nn.functional as F


class TripleSelector(nn.Module, ABC):
    def __init__(self, input_size, class_num, config):
        super(TripleSelector, self).__init__()
        self.config = config
        self.encoder_config = BertConfig.from_json_file(config["pretrained_encoder_config_path"])
        self.encoder = BertModel(config=self.encoder_config,
                                 add_pooling_layer=False,
                                 decoder_like_gpt=False)
        res = self.encoder.load_state_dict(self._load_modified_state_dict(), strict=False)
        assert res.missing_keys == ['embeddings.position_ids'] and res.unexpected_keys == []

        self.input_size = input_size
        self.class_num = class_num,
        self.classifier = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Tanh(),
            nn.Linear(input_size, 2)
        )

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

    def create_entity_query(self, encoded_history, history_mask):
        history_length = history_mask.sum(dim=1)
        mask_encoded_history = encoded_history * history_mask.unsqueeze(dim=-1)
        avg_pooled_history = mask_encoded_history.sum(dim=1) / history_length.unsqueeze(dim=-1)
        return avg_pooled_history

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        features = encoder_outputs[0]
        features = self.create_entity_query(features, attention_mask)
        logits = self.classifier(features)

        return logits,
