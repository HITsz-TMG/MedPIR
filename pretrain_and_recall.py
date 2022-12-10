from src.model import BERTGPTEntity
from main_config import config
import torch
from cikm_dataset.summary_response_dataset import SummaryResponseDataset
import pickle
from cikm_model.supervised_summary import GenSummaryEntityResponse
from cikm_trainer.supervised_summary_trainer import SupervisedSummaryTrainer
import time, os
import json
from collections import OrderedDict
from copy import deepcopy
from tqdm import tqdm


def sort_dataset_by_length(dataset):
    decode_len = []
    for i in dataset.data:
        target_ids, _, _, _ = dataset.decoder_inputs([i], with_summary=False, decoder_with_entity=False)
        decode_len.append(len(target_ids[0]))
    data_with_len = [(i, j) for i, j in zip(dataset.data, decode_len)]
    sorted(data_with_len, key=lambda x: x[1], reverse=True)
    sorted_data = [i for i, _ in data_with_len]
    return sorted_data


def filter_short_dialogue(dataset, n):
    filtered_data = []
    for i in dataset.data:
        if len(i['text'][0]) <= n:
            continue
        filtered_data.append(i)
    return filtered_data


def make_filtered_MedDialog_pairs():
    #   data_path = "./MedDialog/filtered_MedDialog/annotated_entity/train_data.json"
    #   data_path = "./MedDialog/filtered_MedDialog/annotated_entity/test_data.json"
    #   data_path = "./MedDialog/filtered_MedDialog/annotated_entity/dev_data.json"
    data_paths = [
        "./MedDialog/filtered_MedDialog/annotated_entity/train_data.json",
        "./MedDialog/filtered_MedDialog/annotated_entity/dev_data.json"
    ]
    data_types = ["train", "test", "dev"]
    ns = [4, 4]
    for data_path, data_type, n in zip(data_paths, data_types, ns):
        json_data = json.load(open(data_path, 'r', encoding='utf-8'))
        std_data = []
        empty_entity_padding = {k: [] for k in config['entity_type']}
        for dialog in tqdm(json_data):
            if len(dialog) <= n:
                continue
            history_cache = []
            for uid, item in enumerate(dialog):
                utterance = item['Sentence']
                entity = item['Entity']
                spk_str = utterance[:3]
                utterance = utterance[3:]
                if spk_str == "病人：":
                    cur_spk = "Patients"
                elif spk_str == "医生：":
                    cur_spk = "Doctor"
                else:
                    raise ValueError("MedDialog Spk ERROR")
                if uid == 0:
                    if not cur_spk == "Patients":
                        raise ValueError("First Spk Must Be Patients")
                if cur_spk == "Doctor":
                    text_0 = deepcopy(history_cache)
                    text_1 = {"Sentence": utterance}
                    text_1.update(empty_entity_padding)
                    new_item = {"text": [text_0, text_1]}
                    std_data.append(new_item)
                history_cache.append({
                    "Sentence": utterance,
                    "Entity": entity,
                    "id": cur_spk,
                })
        pickle.dump(std_data, open(f"./MedDialog/full_MedDialog/pretrain_{data_type}_pairs.pkl", 'wb'))


def pretrain_on_full_MedDialog(train_data_path, dev_data_path, test_data_path, epoch, lr, state_dict):
    config.update({
        # require to set
        'summary_gate_open': False,
        'entity_gate_open': False,
        'rsep_as_associate': False,
        'recall_gate_network': None,
        'model_name': 'GenSummaryEntityResponse',

        'pretrain_on_MedDialog': True,
        'epoch': epoch,
        'lr': lr,
        'warm_up': 0.1,
        'batch_size': 16,
        'batch_expand_times': 4,
        'start_eval_epoch': 1,
        'eval_times_each_epoch': 2,
        'use_token_type_ids': True,

        'state_dict': state_dict
    })

    train_data = pickle.load(open(train_data_path, 'rb'))
    dev_data = pickle.load(open(dev_data_path, 'rb'))
    test_data = pickle.load(open(test_data_path, 'rb'))

    print(f"{len(train_data)}, {len(dev_data)}, {len(test_data)}")

    model = GenSummaryEntityResponse(config)
    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))

    train_dataset = SummaryResponseDataset(
        vocab_path=config['vocab_path'],
        data=train_data,
        data_type="train",
        config=config,
        with_entity=False,
        with_summary=False,
        use_gat=False,
        dataset_name="MedDialog"
    )

    dev_dataset = SummaryResponseDataset(
        vocab_path=config['vocab_path'],
        data=dev_data,
        data_type="dev",
        config=config,
        with_entity=False,
        with_summary=False,
        use_gat=False,
        dataset_name="MedDialog"
    )
    dev_dataset.data = sort_dataset_by_length(dev_dataset)
    trainer = SupervisedSummaryTrainer(
        train_dataset,
        model,
        dev_dataset=dev_dataset,
        config=config,
        save_root="./cikm_save/MedDialog_pretrain",
    )

    trainer.train()
    return trainer.save_name_list[-1]


def model_load_pretrain_parameters(model, pretrain_path):
    state_dict = torch.load(pretrain_path, map_location='cpu')
    new_params = OrderedDict()
    for key in state_dict:
        if key.startswith("encoder_for_crossattention"):
            new_key = key.replace("encoder_for_crossattention", "skr_summary_encoder")
            new_params[new_key] = deepcopy(state_dict[key])
            new_key = key.replace("encoder_for_crossattention", "skr_entity_encoder")
            new_params[new_key] = deepcopy(state_dict[key])
        if "references_crossattention" in key:
            new_key = key.replace("references_crossattention", "summary_crossattention")
            new_params[new_key] = deepcopy(state_dict[key])
            new_key = key.replace("references_crossattention", "entity_crossattention")
            new_params[new_key] = deepcopy(state_dict[key])

        if "lm_linear.weight" == key:
            new_key = key.replace("lm_linear", "rc_linear")
            new_params[new_key] = deepcopy(state_dict[key])

    state_dict.update(new_params)
    # "cross_gate" "rgat"
    err_params = model.load_state_dict(state_dict, strict=False)
    assert all("cross_gate" in i or "recall_gate_network" in i
               or "gat_embed_project" in i for i in err_params.missing_keys)
    return model


def recall_with_strategy():
    # train_data_path = "./MedDialog/filtered_MedDialog/train_pairs.pkl"
    train_data_path = "./MedDialog/filtered_MedDialog/pairs_with_entity/train_pairs.pkl"
    train_summary_path = "./MedDialog/filtered_MedDialog/train-last6_summary.pkl"

    # dev_data_path = "./MedDialog/filtered_MedDialog/dev_pairs.pkl"
    dev_data_path = "./MedDialog/filtered_MedDialog/pairs_with_entity/dev_pairs.pkl"
    dev_summary_path = "./MedDialog/filtered_MedDialog/dev-last6_summary.pkl"

    # train_data_path = dev_data_path
    # train_summary_path = dev_summary_path

    # strategy: use two linear linear_rc and linear_rp
    # 1. separately: predict recall and response separately, only use linear_rp at test time.
    # 2. jointly: predict recall and response jointly, use different output mask to compute loss.

    # w/o recall generation: GAT and summary_gate_open but without summary
    config.update({
        # require to set
        'summary_gate_open': True,  # --
        'entity_gate_open': True,
        'rsep_as_associate': False,
        'recall_gate_network': "GAT",  # --
        'model_name': 'GenSummaryEntityResponse',

        'model_recall_strategy': "jointly",  # "jointly"
        'epoch': 10,
        'lr': 1e-5,
        'warm_up': 0.1,
        'batch_size': 4,
        'batch_expand_times': 8,
        'start_eval_epoch': 1,
        'eval_times_each_epoch': 2,
        'use_token_type_ids': True,
    })
    generate_summary = True

    summary_strategy = "pcl_bert_sim"
    save_root = "./cikm_save/MedDialog_save/with_entity"

    pretrain_path = "./cikm_save/MedDialog_pretrain/epoch3_1-B2[0.07605]-B4[0.01364].pt"
    model = GenSummaryEntityResponse(config)
    model = model_load_pretrain_parameters(model, pretrain_path)
    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))

    dev_dataset = SummaryResponseDataset(
        vocab_path=config['vocab_path'],
        data_path=dev_data_path,
        data=None,
        summary_data_path=dev_summary_path,
        data_type="dev",
        config=config,
        with_entity=False,
        with_summary=generate_summary,
        summary_strategy=summary_strategy,
        use_gat=True if config['recall_gate_network'] == "GAT" else False,
        dataset_name="MedDialog"
    )
    # not use all dev data to evaluate
    select_dev_index = pickle.load(open("./MedDialog/filtered_MedDialog/select_dev_index.pkl", 'rb'))
    dev_dataset.data = [i for idx, i in enumerate(dev_dataset.data) if idx in select_dev_index]
    # dev_dataset.data = dev_dataset.data[:10]
    print(f"select {len(dev_dataset.data)} dev data")
    decode_len = []
    for i in dev_dataset.data:
        target_ids, summary_end_pos, response_start_pos, prefix = dev_dataset.decoder_inputs(
            [i], with_summary=True, decoder_with_entity=False)
        decode_len.append(len(target_ids[0]))
    data_with_len = [(i, j) for i, j in zip(dev_dataset.data, decode_len)]
    sorted(data_with_len, key=lambda x: x[1], reverse=True)
    dev_dataset.data = [i for i, _ in data_with_len]

    test_dataset = None

    train_dataset = SummaryResponseDataset(
        vocab_path=config['vocab_path'],
        data_path=train_data_path,
        summary_data_path=train_summary_path,
        data_type="train",
        config=config,
        with_entity=False,
        with_summary=generate_summary,
        summary_strategy=summary_strategy,
        use_gat=True if config['recall_gate_network'] == "GAT" else False,
        dataset_name="MedDialog"
    )

    trainer = SupervisedSummaryTrainer(
        train_dataset,
        model,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        config=config,
        save_root=save_root,
    )

    trainer.train()


def main():
    # Full MedDialog Pre-Train
    train_data_path = "./MedDialog/full_MedDialog/train_pairs.pkl"
    dev_data_path = "./MedDialog/full_MedDialog/dev_pairs.pkl"
    test_data_path = "./MedDialog/full_MedDialog/test_pairs.pkl"
    # state_dict =  "./cikm_save/MedDialog_pretrain/epoch2-B2[0.07327]-B4[0.02377].pt"
    state_dict = "./cikm_save/MedDialog_pretrain/epoch5-1221-0130.pt"
    new_state_dict = pretrain_on_full_MedDialog(train_data_path, dev_data_path, test_data_path,
                                                epoch=1, lr=1.5e-5, state_dict=state_dict)

    # Only Pre-train with Long Dialogue
    train_data_path = "./MedDialog/full_MedDialog/pretrain_train_pairs.pkl"
    dev_data_path = "./MedDialog/full_MedDialog/pretrain_dev_pairs.pkl"
    test_data_path = "./MedDialog/full_MedDialog/pretrain_test_pairs.pkl"
    pretrain_on_full_MedDialog(train_data_path, dev_data_path, test_data_path,
                               epoch=3, lr=1.5e-5, state_dict=new_state_dict)


if __name__ == '__main__':
    # main()
    # make_filtered_MedDialog_pairs()
    recall_with_strategy()
