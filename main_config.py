import os
import argparse

config = {
    "model_name": "BERTGPTEntity",  # "BERT2BERTEntity" "BERTGPTEntity"
    "device": "0",

    "use_past_key_and_values": True,

    "use_token_type_ids": False,

    "entity_attention": False,
    "entity_fused_way": "only_attention",  # entity_coarsely_select / entity_info_before_cls / only_attention
    "entity_query_model": "avg_pool_linear",
    "pretrained_select": "BertGPT",  # BertGPT / PCL

    "epoch": 20,
    "batch_size": 6,
    "batch_expand_times": 6,
    "warm_up": 0.1,
    "lr": 2e-5,

    "preprocessing": False,
    # "parallel": "data_parallel",
    "parallel": None,

    "vocab_path": "./data/vocab.txt",
    "bert_config_path": "./pretrained_model/config.json",
    "bertgpt_state_dict": "pretrained/bertGPT_pretrained_model.pth",
    "pretrained_state_dict_path": "./pretrained/PCL-MedBERT/pytorch_model.bin",
    "pretrained_encoder_config_path": "./pretrained/PCL-MedBERT/config.json",
    "pretrained_decoder_config_path": "./pretrained/PCL-MedBERT/config_for_decoder.json",
    "gpt2_config_path": "./pretrained/gpt2/config.json",
    "dialog_gpt_path": "./GPT2-chitchat-master/dialogue_model",

    "train_data_path": "./data/cikm/train-4-25.pkl",
    "dev_data_path": "./data/cikm/dev-4-25.pkl",
    # "test_data_path": "./data/cikm/response_entity_predict_new-0.35.pkl",
    "test_data_path": "./data/cikm/test_new_with_retrieval.pkl",
    "golden_test_data_path": "./data/cikm/process_test.pkl",


    "summary_train_path": "./data/aaai/dialogue_summary_train.pkl",
    "summary_dev_path": "./data/aaai/dialogue_summary_dev.pkl",
    "summary_test_path": "./data/aaai/dialogue_summary_test.pkl",
    "summary_state_dict": "./cikm_save/SummaryResponse/epoch3-B[0.23985]-B1[0.31058]-B4[0.16912].pt",
    # "dialogue_summary_train_path": "./data/summary/summary_for_train.pkl",
    "dialogue_summary_train_path": "./data/summary/summary_for_train2.pkl",
    "dialogue_summary_dev_path": "./data/summary/summary_for_dev.pkl",
    "dialogue_summary_test_path": "./data/aaai/dialogue_summary_test.pkl",

    "train_data_with_predict_next_entities": "./data/cikm/train_data_with_predict_next_entities-5-5.pkl",
    # "train_data_with_predict_next_entities": "./data/cikm/test_dataset_references-5-11.pkl",
    "top_k": 64,
    "top_p": 1,
    "min_len": 1,
    "max_len": 300,
    "beam_size": 4,
    "length_penalty": 1,
    "no_repeat_ngram_size": 5,
    "encoder_no_repeat_ngram_size": 0,
    "repetition_penalty": 1,
}

parser = argparse.ArgumentParser()
parser.add_argument("--device", default=config['device'], required=False, type=str)
parser.add_argument("--top_k", default=config['top_k'], required=False, type=int)
parser.add_argument("--beam_size", default=config['beam_size'], required=False, type=int)
parser.add_argument("--summary_strategy", default=None, required=False, type=str)
parser.add_argument("--beam_sample", action='store_true', required=False)
parser.add_argument("--start", default=None, required=False, type=int)
parser.add_argument("--end", default=None, required=False, type=int)

args = vars(parser.parse_args())
config.update(args)

print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = config['device']

test_with_next_entities_path = "./data/cikm/predict_and_retrieve.pkl"
config['test_data_path'] = test_with_next_entities_path

print(config)
print("")

entity_type = ['Symptom', 'Medicine', 'Test', 'Attribute', 'Disease']
symptom = ['??????', '??????', '??????', '??????', '????????????', '??????', '??????', '??????', '?????????', '??????', '??????', '????????????', '????????????',
           '??????', '??????', '??????',
           '??????', '???????????????', '??????', '??????', '????????????', '????????????', '??????', '????????????', '????????????', '??????', '????????????', '??????', '??????', '????????????',
           '?????????', '??????', '??????', '??????', '????????????', '??????????????????', '??????????????????', '??????', '????????????', '??????', '????????????', '??????', '??????', '??????', '??????',
           '????????????', '??????', '??????', '??????', '??????', '?????????', '??????', '??????', '??????', '??????', '??????', '??????', '????????????', '??????', '??????', '??????',
           '???????????????']
medicine = ['??????', '?????????', '????????????', '???????????????', '????????????', '?????????', '?????????', '???????????????', '?????????', '?????????', '????????????', '???????????????', '?????????',
            '????????????????????????', '????????????', '??????', '????????????', '???????????????', '?????????', '?????????????????????', '????????????', '??????????????????', '????????????????????????', '?????????',
            '???????????????', '????????????', '????????????', '?????????', '?????????????????????', '?????????', '????????????', '?????????', '????????????', '???????????????', '???????????????', '?????????', '??????',
            '??????', '????????????', '?????????', '?????????', '?????????', '?????????', '?????????', '????????????', '?????????', '???????????????', '?????????', '??????', '?????????', '?????????',
            '?????????', '???????????????', '?????????', '????????????', '????????????', '?????????', '??????', '???????????????', '????????????', '????????????', '????????????']
test = ['b???', '?????????', '??????', '????????????', '??????', '????????????', '????????????', '??????????????????', '??????', '????????????', 'ct', '?????????', '?????????', '?????????', '?????????',
        '?????????', '?????????', '?????????', '?????????', '??????']
attribute = ['??????', '??????', '??????', '??????']
disease = ['??????', '?????????', '?????????', '?????????', '??????', '?????????', '?????????', '??????????????????', '??????', '?????????', '??????', '??????']

entity_type_num = {
    'Symptom': len(symptom),
    'Medicine': len(medicine),
    'Test': len(test),
    'Attribute': len(attribute),
    'Disease': len(disease)
}

eid2entity = [
    '??????', '??????', '??????', '??????', '????????????', '??????', '??????', '??????', '?????????', '??????', '??????', '????????????', '????????????', '??????',
    '??????', '??????', '??????', '???????????????', '??????', '??????', '????????????', '????????????', '??????', '????????????', '????????????', '??????', '????????????',
    '??????', '??????', '????????????', '?????????', '??????', '??????', '??????', '????????????', '??????????????????', '??????????????????', '??????', '????????????', '??????',
    '????????????', '??????', '??????', '??????', '??????', '????????????', '??????', '??????', '??????', '??????', '?????????', '??????', '??????', '??????', '??????',
    '??????', '??????', '????????????', '??????', '??????', '??????', '???????????????', '??????', '?????????', '????????????', '???????????????', '????????????', '?????????',
    '?????????', '???????????????', '?????????', '?????????', '????????????', '???????????????', '?????????', '????????????????????????', '????????????', '??????', '????????????',
    '???????????????', '?????????', '?????????????????????', '????????????', '??????????????????', '????????????????????????', '?????????', '???????????????', '????????????', '????????????',
    '?????????', '?????????????????????', '?????????', '????????????', '?????????', '????????????', '???????????????', '???????????????', '?????????', '??????', '??????',
    '????????????', '?????????', '?????????', '?????????', '?????????', '?????????', '????????????', '?????????', '???????????????', '?????????', '??????', '?????????',
    '?????????', '?????????', '???????????????', '?????????', '????????????', '????????????', '?????????', '??????', '???????????????', '????????????', '????????????',
    '????????????', 'b???', '?????????', '??????', '????????????', '??????', '????????????', '????????????', '??????????????????', '??????', '????????????', 'ct', '?????????',
    '?????????', '?????????', '?????????', '?????????', '?????????', '?????????', '?????????', '??????', '??????', '??????', '??????', '??????', '??????', '?????????',
    '?????????', '?????????', '??????', '?????????', '?????????', '??????????????????', '??????', '?????????', '??????', '??????'
]

entity2eid = {e: idx for idx, e in enumerate(eid2entity)}

config['symptom'] = symptom
config['medicine'] = medicine
config['test'] = test
config['attribute'] = attribute
config['disease'] = disease

config['entity_type'] = entity_type
config['entity'] = eid2entity
config['entity_type_num'] = entity_type_num
config['entity2eid'] = entity2eid

config['entity_predict'] = False
