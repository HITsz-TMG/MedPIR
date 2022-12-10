from abc import ABC
import torch
from src.utils.gpt_modeling import TransformerEncoder, TransformerDecoderLM
from torch import nn
from src.utils.hugging_face_bert import BertModel
from src.utils.hugging_face_gpt import GPT2Model
from transformers import BertConfig, GPT2Config
import torch.nn.functional as F


class Classifier(nn.Module, ABC):
    def __init__(self, input_size, class_num, dropout=0.1):
        super(Classifier, self).__init__()
        self.l1 = nn.Linear(input_size, input_size)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(input_size, class_num)

    def forward(self, features):
        hidden = self.l1(features)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        hidden = self.out(hidden)
        return torch.sigmoid(hidden)


class EntityPredict(nn.Module, ABC):
    def __init__(self, config, entity_type_num: dict):
        super().__init__()
        self.config = config

        self.encoder_config = BertConfig.from_json_file(config["pretrained_encoder_config_path"])
        self.encoder = BertModel(config=self.encoder_config,
                                 add_pooling_layer=False,
                                 decoder_like_gpt=False)
        res = self.encoder.load_state_dict(self._load_modified_state_dict(), strict=False)
        assert res.missing_keys == ['embeddings.position_ids'] and res.unexpected_keys == []

        # self.entity_classifiers = nn.ModuleList([
        #     Classifier(self.encoder_config.hidden_size, entity_num, self.encoder_config.hidden_dropout_prob, ) for
        #     entity_type, entity_num in entity_type_num.items()]
        # )
        self.entity_type_num = entity_type_num
        self.symptom_classifier = Classifier(self.encoder_config.hidden_size, self.entity_type_num["Symptom"])
        self.medicine_classifier = Classifier(self.encoder_config.hidden_size, self.entity_type_num["Medicine"])
        self.test_classifier = Classifier(self.encoder_config.hidden_size, self.entity_type_num["Test"])
        self.attribute_classifier = Classifier(self.encoder_config.hidden_size, self.entity_type_num["Attribute"])
        self.disease_classifier = Classifier(self.encoder_config.hidden_size, self.entity_type_num["Disease"])
        self.classifier = [
            self.symptom_classifier,
            self.medicine_classifier,
            self.test_classifier,
            self.attribute_classifier,
            self.disease_classifier
        ]

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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, label=None, **kwargs):
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_hidden_states = encoder_output[0]
        first_token_tensor = sequence_hidden_states[:, 0]

        # ent_predict = []
        #
        # for classifier, ent_label in zip(self.entity_classifiers, self.entity_label_keys):
        #     ent_predict.append(classifier(first_token_tensor).squeeze(dim=-1))

        five_topic_probs = [classify(first_token_tensor) for classify in self.classifier]
        topic_probs = torch.cat(five_topic_probs, -1)

        if label is not None:
            topic_weight = torch.ones_like(label) + 4 * label
            loss = F.binary_cross_entropy(topic_probs, label.to(topic_probs.dtype), topic_weight.float())
        else:
            loss = None

        return topic_probs, five_topic_probs, loss,


