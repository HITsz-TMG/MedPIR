from abc import ABC
import torch
from src.utils.gpt_modeling import TransformerEncoder, TransformerDecoderLM
from torch import nn
from src.utils.hugging_face_bert import BertModel
from src.utils.hugging_face_gpt import GPT2Model
from transformers import BertConfig, GPT2Config


class SpkPredict(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder_config = BertConfig.from_json_file(config["pretrained_encoder_config_path"])
        self.encoder = BertModel(config=self.encoder_config,
                                 add_pooling_layer=False,
                                 decoder_like_gpt=False)
        res = self.encoder.load_state_dict(self._load_modified_state_dict(), strict=False)
        assert res.missing_keys == ['embeddings.position_ids'] and res.unexpected_keys == []

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_config.hidden_size, self.encoder_config.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.encoder_config.hidden_dropout_prob),
            nn.Linear(self.encoder_config.hidden_size, 2)
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, label=None):
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_hidden_states = encoder_output[0]
        first_token_tensor = sequence_hidden_states[:, 0]
        logits = self.classifier(first_token_tensor)
        return logits,
