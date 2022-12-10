from cikm_config import config
from cikm_trainer.trainer import CIKMTrainer
from cikm_trainer.reorganize_trainer import ReorganizeTrainer
from cikm_dataset.dataset import BaseDataset
from cikm_dataset.reorganize_dataset import ReorganizeDataset
from src.model import BERTGPTEntity
from cikm_model.reorganize_model import Reorganize
from cikm_entity_annotation import annotate_predict_result_entity_from_k_fold_model
from cikm_entity_annotation import calculate_F1
import torch
import random
import pickle
import importlib
from tqdm import tqdm
from cikm_generate_utils import prepare_input_utils
from cikm_generate_utils.generator import BeamSample, Greedy
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import os
import json
import Levenshtein
from collections import Counter
import time
from ccks_evaluate import KD_Metric
from cikm_entity_annotation import get_entity_type, input_entities_cal_F1

DialogModel = getattr(importlib.import_module("src.model"), config['model_name'])
config['model_class'] = DialogModel


def train_Reorganize_G1():
    print("Train reorganize G1")
    model = BERTGPTEntity(config)
    if config.get('state_dict', None) is not None:
        print("load G1 state dict")
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))

    all_train_data = pickle.load(open(config['train_data_path'], 'rb'))
    split_pos = int(len(all_train_data) / 2)
    train_G1_data = all_train_data[:split_pos]

    train_dataset = BaseDataset(
        vocab_path=config['vocab_path'],
        data_type="train",
        config=config,
        data=train_G1_data
    )

    dev_dataset = BaseDataset(
        vocab_path=config['vocab_path'],
        data_path=config['dev_data_path'],
        data_type="dev",
        config=config
    )

    if config['debug']:
        print("Debug...")
        train_dataset.data = train_dataset.data[:10]
        dev_dataset.data = dev_dataset.data[:10]

    print(len(train_dataset.data))
    print(len(dev_dataset.data))

    trainer = CIKMTrainer(
        train_dataset,
        model,
        dev_dataset=dev_dataset,
        config=config,
        save_root="./cikm_save/Reorganize_model",
    )

    trainer.train()


def sentence_similarity(str1, str2):
    # sim = Levenshtein.ratio(str1, str2)
    sim = (len(str1) + len(str2) - Levenshtein.distance(str1, str2)) / (len(str1) + len(str2))
    return sim


def judge_add_to_references(ref_list, cur_str):
    if all(sentence_similarity(r, cur_str) < 0.6 for r in ref_list):
        # if all(r != cur_str for r in ref_list):
        return True
    else:
        return False


@torch.no_grad()
def build_G2_train_dataset(start=None, end=None, model=None, train_data=None, dump_file_name=None):
    assert config['use_entity_appendix'] is False
    print("Build G2 train-dataset through reorganize-G1")

    if model is None:
        # config['state_dict'] = "./cikm_save/Reorganize_model/G1-epoch11-B-0.11400.pt"
        config['state_dict'] = "./cikm_save/BertGPT-baseline/BertGPTepoch14-B[0.11407]-B1[0.16539]-B4[0.06275].pt"
        model = BERTGPTEntity(config)
        if config.get('state_dict', None) is not None:
            model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

    if train_data is None:
        train_data = pickle.load(open(config['train_data_path'], 'rb'))

    if start is not None and end is not None:
        print("train_data[{}:{}]".format(start, end))
        train_data = train_data[start:end]

    train_dataset = BaseDataset(
        vocab_path=config['vocab_path'],
        data_type="train",
        config=config,
        data=train_data
    )
    # train: 142828

    # train_dataset.data = train_dataset.data[:5]
    references_list = [[] for _ in range(len(train_dataset))]
    reference_num = 3
    generator = BeamSample(config)

    idx_map = {i: i for i in range(len(train_dataset))}

    max_gen_times = 5
    gen_times = 0

    while gen_times < max_gen_times:

        gen_times += 1

        iterator = train_dataset.get_dataloader(batch_size=1, shuffle=False)
        input_process_func = prepare_input_utils.prepare_input_for_encode_step_BertGPT

        predict_result = generator.generate(
            early_stopping=False,
            prefix_allowed_tokens_fn=None,
            model=model,
            data_iterator=iterator,
            prepare_input_for_encode_step=input_process_func
        )

        for idx in range(len(train_dataset)):
            if judge_add_to_references(references_list[idx_map[idx]], predict_result['predict'][idx]):
                references_list[idx_map[idx]].append(predict_result['predict'][idx])

        # update unfinished
        new_idx_map = dict()
        unfinished_data = []
        for idx in range(len(train_dataset)):
            if len(references_list[idx_map[idx]]) < reference_num:
                new_idx_map[len(unfinished_data)] = idx_map[idx]
                unfinished_data.append(train_dataset.data[idx])
        train_dataset.data = unfinished_data
        idx_map = new_idx_map

        if len(idx_map) == 0:
            break

    if dump_file_name is None:
        dump_file_name = "./reference_data/{}-{}.pkl".format(start, end)
    pickle.dump(
        references_list,
        open(dump_file_name, 'wb')
    )
    # assert all(len(i) == reference_num for i in references_list)


@torch.no_grad()
def AIJ_build_G2_test_dev_dataset(train_data=None, dump_path=None):
    assert config['use_entity_appendix'] is False
    print("Build G2 train-dataset through reorganize-G1")

    raw_response = {str(idx): [] for idx in range(len(train_data))}

    config['state_dict'] = "./cikm_save/BertGPT-baseline/BertGPTepoch14-B[0.11407]-B1[0.16539]-B4[0.06275].pt"
    model = BERTGPTEntity(config)
    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    train_dataset = BaseDataset(
        vocab_path=config['vocab_path'],
        data_type="train",
        config=config,
        data=train_data
    )

    reference_num = 3
    generator = BeamSample(config)

    max_gen_times = 2 * reference_num

    iterator = train_dataset.get_dataloader(batch_size=1, shuffle=False)
    input_process_func = prepare_input_utils.prepare_input_for_encode_step_BertGPT

    for cur_gen_time in range(max_gen_times):

        for idx, item in enumerate(tqdm(iterator, total=len(iterator), ncols=50)):
            if len(raw_response[str(idx)]) >= reference_num:
                continue
            cur_res = generator.generate_one_item(item, model, input_process_func)
            if judge_add_to_references(raw_response[str(idx)], cur_res):
                raw_response[str(idx)].append(cur_res)

    json.dump(raw_response, open(dump_path.replace(".pkl", '.json'), 'w', encoding='utf-8'), ensure_ascii=False)

    pkl_raw = []
    for i in range(len(train_data)):
        pkl_raw.append(raw_response[str(i)])

    pickle.dump(raw_response, open(dump_path, 'wb'))


@torch.no_grad()
def AIJ_build_G2_train_dataset(start=None, end=None, model=None, train_data=None):
    assert config['use_entity_appendix'] is False
    print("Build G2 train-dataset through reorganize-G1")
    raw_path = "./raw_response3/{}-{}.json".format(start, end)
    if os.path.exists(raw_path):
        raw_response = json.load(open(raw_path, 'r', encoding='utf-8'))
    else:
        raw_response = {str(idx): [] for idx in range(start, end)}

    if model is None:
        # config['state_dict'] = "./cikm_save/Reorganize_model/G1-epoch11-B-0.11400.pt"
        config['state_dict'] = "./cikm_save/BertGPT-baseline/BertGPTepoch14-B[0.11407]-B1[0.16539]-B4[0.06275].pt"
        model = BERTGPTEntity(config)
        if config.get('state_dict', None) is not None:
            model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

    if train_data is None:
        train_data = pickle.load(open(config['train_data_path'], 'rb'))

    if start is not None and end is not None:
        print("train_data[{}:{}]".format(start, end))
        train_data = train_data[start:end]

    train_dataset = BaseDataset(
        vocab_path=config['vocab_path'],
        data_type="train",
        config=config,
        data=train_data
    )
    # train: 142828

    reference_num = 3
    generator = BeamSample(config)

    max_gen_times = 2 * reference_num

    iterator = train_dataset.get_dataloader(batch_size=1, shuffle=False)
    input_process_func = prepare_input_utils.prepare_input_for_encode_step_BertGPT

    for cur_gen_time in range(max_gen_times):

        for idx, item in enumerate(tqdm(iterator, total=len(iterator), ncols=50), start):
            if len(raw_response[str(idx)]) >= reference_num:
                continue
            cur_res = generator.generate_one_item(item, model, input_process_func)
            if judge_add_to_references(raw_response[str(idx)], cur_res):
                raw_response[str(idx)].append(cur_res)

            if idx % 100 == 0:
                json.dump(raw_response, open(raw_path, 'w', encoding='utf-8'), ensure_ascii=False)
        json.dump(raw_response, open(raw_path, 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(raw_response, open(raw_path, 'w', encoding='utf-8'), ensure_ascii=False)


@torch.no_grad()
def AIJ_build_runner():
    config['beam_size'] = 5
    config['top_k'] = 64
    config['max_len'] = 100
    config['state_dict'] = "./cikm_save/Reorganize_model/G1-epoch11-B-0.11400.pt"
    model = BERTGPTEntity(config)
    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    train_data = pickle.load(open(config['train_data_path'], 'rb'))

    for start in range(config['start'], config['end'], config['distance']):
        AIJ_build_G2_train_dataset(start=start, end=start + config['distance'], model=model, train_data=train_data)
        torch.cuda.empty_cache()


def train_Reorganize_G2():
    config['model_name'] = 'Reorganize'
    # config['warm_up'] = 1
    # config['epoch'] = 50
    # config['batch_size'] = 2
    # config['batch_expand_times'] = 1
    # config['lr'] = 5e-5
    config["use_references_or_history_crossattention"] = True

    # train_data = pickle.load(open(config['train_data_path'], 'rb'))[:5]
    # ref_data = pickle.load(open(config['train_refs_path'], 'rb'))[:5]

    # ref_data = pickle.load(open("./data/cikm/references_for_G2_train.pkl", 'rb'))

    train_dataset = ReorganizeDataset(
        vocab_path=config['vocab_path'],
        data_type="train",
        config=config,
        data_path=config['train_data_path'],
        # data=train_data,
        # ref_data=ref_data,
        ref_data_path=config['train_refs_path']
    )

    dev_dataset = ReorganizeDataset(
        vocab_path=config['vocab_path'],
        data_type="dev",
        config=config,
        data_path=config['dev_data_path'],
        # data=train_data,
        # ref_data=ref_data,
        ref_data_path=config['dev_refs_path']
    )
    if dev_dataset.data == train_dataset.data:
        print("Debug...")

    if config['debug']:
        print("Debug...")
        train_dataset.data = train_dataset.data[:10]
        dev_dataset.data = dev_dataset.data[:10]

    print(len(train_dataset.data))
    print(len(dev_dataset.data))

    model = Reorganize(config)
    print("Train reorganize G2")
    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'), strict=False)

    trainer = ReorganizeTrainer(
        train_dataset,
        model,
        dev_dataset=dev_dataset,
        config=config,
        save_root="./cikm_save/Reorganize_model"
    )

    trainer.train()
    pass


def calculate_BLEU(reference, predict):
    predict_sentence = [list(i) for i in predict]
    reference_sentence_list = [[list(i)] for i in reference]
    bleu_1 = corpus_bleu(reference_sentence_list, predict_sentence, weights=[1, 0, 0, 0],
                         smoothing_function=SmoothingFunction().method7)
    bleu_2 = corpus_bleu(reference_sentence_list, predict_sentence, weights=[0.5, 0.5, 0, 0],
                         smoothing_function=SmoothingFunction().method7)
    bleu_3 = corpus_bleu(reference_sentence_list, predict_sentence, weights=[1 / 3, 1 / 3, 1 / 3, 0],
                         smoothing_function=SmoothingFunction().method7)
    bleu_4 = corpus_bleu(reference_sentence_list, predict_sentence, weights=[0.25, 0.25, 0.25, 0.25],
                         smoothing_function=SmoothingFunction().method7)
    return {
        "BLEU-1": bleu_1,
        "BLEU-2": bleu_2,
        "BLEU-3": bleu_3,
        "BLEU-4": bleu_4
    }


@torch.no_grad()
def build_G2_dataset_runner():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-s", "--start", type=int, required=True)
    # parser.add_argument("-e", "--end", type=int, required=True)
    # parser.add_argument("-d", "--distance", type=int, required=False, default=3500)
    # args = vars(parser.parse_args())
    config['state_dict'] = "./cikm_save/Reorganize_model/G1-epoch11-B-0.11400.pt"

    model = BERTGPTEntity(config)
    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    train_data = pickle.load(open(config['train_data_path'], 'rb'))

    for start in range(config['start'], config['end'], config['distance']):
        build_G2_train_dataset(start=start, end=start + config['distance'], model=model, train_data=train_data)
        torch.cuda.empty_cache()


def combine_references_data():
    root_path = "./reference_data"
    start = list(range(0, 140000, 3500))
    file_list = ["{}-{}.pkl".format(i, i + 3500) for i in start]
    file_list.append("140000-142828.pkl")
    file_list = [os.path.join(root_path, i) for i in file_list]

    combine_result = []
    for i in file_list:
        combine_result.extend(pickle.load(open(i, 'rb')))
    cnt = Counter([len(i) for i in combine_result])
    print("")
    pickle.dump(combine_result, open("./reference_data/5-5-refs.pkl", 'wb'))


@torch.no_grad()
def predict_Reorganize_G2():
    config['model_name'] = "Reorganize"
    if config.get("state_dict") is None:
        config['state_dict'] = "./cikm_save/Reorganize_model/epoch10-B[0.20066]-B1[0.27013]-B4[0.13119].pt"
    print(config['state_dict'])
    config["use_references_or_history_crossattention"] = True

    model = Reorganize(config)
    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))
    model.eval()

    # example id :116449
    # exp_id = 116449
    # train_dataset = pickle.load(open(config['train_data_path'], 'rb'))
    # train_refs = pickle.load(open(config['train_refs_path'], 'rb'))
    # test_data = [train_dataset[exp_id]]
    # test_refs = [train_refs[exp_id]]
    # for idx, i in enumerate(train_dataset):
    #     find_target = "早晚胃疼，吃了饭就不疼了。大便正常，也不恶心"
    #     if find_target in i['text'][0][0]['Sentence']:
    #         print("-----")
    #         print(idx)
    #         a = [t['Sentence'] for t in i['text'][0]]
    #         print(" ".join(a))
    #         print("-----")
    # test_dataset = ReorganizeDataset(
    #     vocab_path=config['vocab_path'],
    #     data_type="test",
    #     config=config,
    #     data=test_data,
    #     ref_data=test_refs
    # )
    test_dataset = ReorganizeDataset(
        vocab_path=config['vocab_path'],
        data_path=config['test_data_path'],
        ref_data_path=config['test_refs_path'],
        data_type="test",
        config=config,
    )
    # test_dataset.data = test_dataset.data[:10]
    generator = BeamSample(config)
    iterator = test_dataset.get_dataloader(batch_size=1, shuffle=False)
    input_process_func = prepare_input_utils.prepare_input_for_encode_step_Reorganize
    predict_result = generator.generate(
        early_stopping=False,
        prefix_allowed_tokens_fn=None,
        model=model,
        data_iterator=iterator,
        prepare_input_for_encode_step=input_process_func
    )
    scores = calculate_BLEU(predict_result['reference'], predict_result['predict'])

    save_root = "./cikm_predict_result"
    file_name = "{}-{}.json".format(
        config['model_name'],
        time.strftime("%m-%d")
    )
    with open(os.path.join(save_root, file_name), 'w', encoding="utf-8") as writer:
        dump_dict = {
            "score": scores,
            "generate_args": {
                "top_k": config['top_k'],
                "beam_size": config['beam_size'],
            },
            "predict": [i for i in predict_result['predict']]
        }
        json.dump(dump_dict, writer, ensure_ascii=False)
    return os.path.join(save_root, file_name)


def predict_and_cal_metrics():
    result_path = predict_Reorganize_G2()
    dump_path = "./cikm_predict_result/{}.json".format(time.strftime("%m-%d-%H-%M"))
    # annotate_predict_result_entity_from_k_fold_model(
    #     result_json_path=result_path,
    #     dump_path=dump_path
    # )
    kdm = KD_Metric()
    predict_entities = []
    res = json.load(open(result_path, 'r', encoding='utf-8'))
    for i in res['predict']:
        entities = kdm.convert_sen_to_entity_set(i)
        new_item = {
            "Symptom": [],
            "Medicine": [],
            "Test": [],
            "Attribute": [],
            "Disease": []
        }
        for e in entities:
            new_item[get_entity_type(e)].append(e)
        predict_entities.append(new_item)
    golden_entities = []
    golden_test = pickle.load(open(config['golden_test_data_path'], 'rb'))
    for i in golden_test:
        new_item = {
            "Symptom": i['text'][1]['Symptom'],
            "Medicine": i['text'][1]['Medicine'],
            "Test": i['text'][1]['Test'],
            "Attribute": i['text'][1]['Attribute'],
            "Disease": i['text'][1]['Disease']
        }
        golden_entities.append(new_item)
    f1_info = input_entities_cal_F1(golden_entities, predict_entities)
    # f1_info = input_entities_cal_F1(
    #     "./data/cikm/response_entity_annotation_5-1.pkl",
    #     dump_path
    # )
    # data = json.load(open(dump_path, 'r', encoding='utf-8'))
    res['f1_info'] = f1_info
    json.dump(res, open(dump_path, 'w', encoding='utf-8'), ensure_ascii=False)


def combine_raw_response():
    root_path = "./raw_response3"
    start = list(range(0, 140000, 3500))
    file_list = ["{}-{}.json".format(i, i + 3500) for i in start]
    # file_list.append("140000-142828.json")
    file_list.append("140000-143500.json")
    file_list = [os.path.join(root_path, i) for i in file_list]
    combine_dict = {str(idx): None for idx in range(0, 143500)}
    combine_result = []
    for i in file_list:
        cur = json.load(open(i, 'r', encoding='utf-8'))
        combine_dict.update(cur)
    res_dict = {idx: None for idx in range(0, 142828)}
    for idx in range(0, 142828):
        combine_result.append(combine_dict[str(idx)])
        res_dict[idx] = combine_dict[str(idx)]
    cnt = Counter([len(i) for i in combine_result])
    print(cnt)
    pickle.dump(combine_result, open("./raw_response3/0.6.pkl", 'wb'))
    json.dump(res_dict, open("./raw_response3/0.6.json", 'w', encoding='utf-8'), ensure_ascii=False)


# @torch.no_grad()
def draw_hot_map():
    config['model_name'] = "Reorganize"
    if config.get("state_dict") is None:
        config['state_dict'] = "./cikm_save/Reorganize_model/epoch10-B[0.20066]-B1[0.27013]-B4[0.13119].pt"
    print(config['state_dict'])
    config["use_references_or_history_crossattention"] = True

    model = Reorganize(config)
    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))
    model.eval()

    # example id :116449

    config['batch_size'] = 1

    train_dataset = ReorganizeDataset(
        vocab_path=config['vocab_path'],
        data_type="train",
        config=config,
        create_one=True,
    )

    history = [
        {"id": 'Patients', "Sentence": "最近总是拉红色糊状大便，有里急后重感。请问是什么原因（男，24岁）"},
        {"id": 'Doctor', "Sentence": "您好！这种情况多长时间了？"},
        {"id": 'Patients', "Sentence": "有一个多月了，隔几天就有这种情况。"},
        {"id": 'Patients', "Sentence": "手纸擦了，纸上有血。"},
        {"id": 'Doctor', "Sentence": "这种情况考虑是肠炎的可能，最好是到医院消化内科就诊。"},
        {"id": 'Doctor', "Sentence": "化验一下便常规，必要时做肠镜检查。"},
        {"id": 'Patients', "Sentence": "8月份做过肠镜，说是有盲肠息肉，已经去掉了。之前检查是因为拉完，滴血。现在是一直拉红色糊状物。"},
        {"id": 'Doctor', "Sentence": "红色糊状物考虑病情已经加重，最好是到医院消化内科复查肠镜。"},
        {"id": 'Doctor', "Sentence": "还要化验血常规，长期大便出血会引起贫血。"},
        {"id": 'Doctor', "Sentence": "因为伴有里急后重感，一定要注意查找病因。"},
        {"id": 'Patients', "Sentence": "要复查肠镜吗？之前也是有里急后重感。检查说没啥问题。现在这样一定要查肠镜吗？可能是什么病因。"},
        {"id": 'Doctor', "Sentence": "你可以先化验一下便常规看有没有脓细胞，白细胞。"},
        {"id": 'Doctor', "Sentence": "可应用抗菌消炎药物治疗，并注意饮食调理。"},
        {"id": 'Patients', "Sentence": "这个检查完了，能知道是什么病因吗？现在特别担心是不是病很严重。"},
    ]

    response = "肠炎的可能性较大，但是需要肠镜检查排除其他肠道疾病。"
    entities = {
        "Symptom": [],
        "Medicine": [],
        "Test": ["肠镜"],
        "Attribute": [],
        "Disease": ["肠炎"]
    }
    raw_response = [
        "如果肠镜没有问题，考虑是肠功能紊乱。",
        "不要担心，考虑肠炎的可能性比较大。"
    ]
    train_dataset.create_one_sample(
        history=history, response=response, entities=entities, raw_response=raw_response
    )

    print(len(train_dataset.data))

    model = Reorganize(config)
    print("Train reorganize G2")
    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'), strict=False)

    trainer = ReorganizeTrainer(
        train_dataset,
        model,
        # dev_dataset=dev_dataset,
        config=config,
        save_root="./cikm_save/tmp"
    )
    trainer.train()


def create_new_test():
    new_test = BaseDataset(
        vocab_path=config['vocab_path'],
        data_type="train",
        config=config,
        original_data_path="./test.pk",
        data_path="./process_test.pkl",
        preprocess=True,
    )
    print("")


if __name__ == '__main__':
    # train_Reorganize_G1()
    # build_G2_train_dataset(start=122500, end=140000)

    # train_Reorganize_G2()

    # build_G2_dataset_runner()
    # predict_path = predict_Reorganize_G2()

    predict_and_cal_metrics()

    predict_and_cal_metrics()

    predict_and_cal_metrics()

    # dev_data = pickle.load(open(config['dev_data_path'], 'rb'))
    # AIJ_build_G2_test_dev_dataset(train_data=dev_data, dump_path="./raw_response3/dev_raw.pkl")

    # test_data = pickle.load(open(config['test_data_path'], 'rb'))
    # AIJ_build_G2_test_dev_dataset(train_data=test_data, dump_path="./raw_response3/new_test_raw.pkl")

    # combine_references_data()

    # AIJ_build_runner()
    # combine_raw_response()
    # draw_hot_map()


    # annotate_predict_result_entity_from_k_fold_model(
    #     result_json_path=result_path,
    #     dump_path=dump_path
    # )
