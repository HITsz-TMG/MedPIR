from abc import ABC
import torch
from torch import nn
from src.utils.hugging_face_bert import BertModel
from transformers import BertConfig
import torch.nn.functional as F


class NextEntityClassifier(nn.Module, ABC):
    def __init__(self, input_size, class_num, dropout=0.1):
        super(NextEntityClassifier, self).__init__()
        self.input_size = input_size
        self.class_num = class_num,
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, input_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(input_size, class_num),
            nn.Sigmoid()
        )

    def forward(self, features):
        return self.classifier(features)


class CIKMNextEntityPredict(nn.Module, ABC):
    def __init__(self, config, encoder=None):
        super().__init__()
        self.config = config

        if config.get('pcl_encoder_predict_entity', False) or encoder is None:
            print("use PCL BERT -> next entity predict")
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

        self.eid2entity = config['entity']

        self.entity_type_num = config['entity_type_num']

        self.symptom_classifier = NextEntityClassifier(
            768, self.entity_type_num["Symptom"]
        )
        self.medicine_classifier = NextEntityClassifier(
            768, self.entity_type_num["Medicine"]
        )
        self.test_classifier = NextEntityClassifier(
            768, self.entity_type_num["Test"]
        )
        self.attribute_classifier = NextEntityClassifier(
            768, self.entity_type_num["Attribute"]
        )
        self.disease_classifier = NextEntityClassifier(
            768, self.entity_type_num["Disease"]
        )

        self.classifier = [
            self.symptom_classifier,
            self.medicine_classifier,
            self.test_classifier,
            self.attribute_classifier,
            self.disease_classifier
        ]

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

    def create_entity_query(self, encoded_history, history_mask):
        history_length = history_mask.sum(dim=1)
        mask_encoded_history = encoded_history * history_mask.unsqueeze(dim=-1)
        avg_pooled_history = mask_encoded_history.sum(dim=1) / history_length.unsqueeze(dim=-1)
        return avg_pooled_history

    def forward(self, input_ids=None, attention_mask=None, label=None):
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        encoded_history = encoder_output[0]
        # entity_query = sequence_hidden_states[:, 0]
        entity_query = self.create_entity_query(encoded_history, attention_mask)
        five_topic_probs = [classify(entity_query) for classify in self.classifier]
        topic_probs = torch.cat(five_topic_probs, -1)

        if label is not None:
            topic_weight = torch.ones_like(label) + 3 * label
            loss = F.binary_cross_entropy(topic_probs, label.to(topic_probs.dtype), topic_weight.float())
        else:
            loss = None

        return topic_probs, five_topic_probs, loss,
