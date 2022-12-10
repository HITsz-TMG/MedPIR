from cikm_config import config
from cikm_trainer.trainer import CIKMTrainer
from cikm_trainer.hred_trainer import HREDTrainer
from cikm_trainer.gpt_trainer import GPTTrainer
from cikm_dataset.gpt_dataset import GPTDataset
from cikm_dataset.dataset import BaseDataset
from cikm_dataset.hred_dataset import HREDDataset
from cikm_dataset.seq2seq_dataset import Seq2SeqDataset
from cikm_trainer.seq2seq_trainer import Seq2SeqTrainer
from src.model import BERTGPTEntity
from cikm_baseline.model import HRED, DialogGPT, Seq2Seq
import torch
import random
import pickle
import importlib
from tqdm import tqdm
from cikm_generate_utils import prepare_input_utils
from cikm_generate_utils.generator import BeamSample, Greedy
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import os
import json
import time
from ccks_evaluate import KD_Metric
from cikm_entity_annotation import input_entities_cal_F1, get_entity_type
from src.eval.eval_utils import input_entities_cal_F1, get_entity_type, sentence_BLEU_avg, calculate_BLEU, \
    distinct_n_corpus_level

DialogModel = getattr(importlib.import_module("src.model"), config['model_name'])
config['model_class'] = DialogModel
config['recall'] = False
config.update({
    "train_data_path": "./MedDialog/filtered_MedDialog/train_pairs.pkl",
    "dev_data_path": "./MedDialog/filtered_MedDialog/dev_pairs.pkl",
    "test_data_path": "./MedDialog/filtered_MedDialog/test_pairs.pkl",
    "train_summary_path": "./MedDialog/filtered_MedDialog/train-last6_summary.pkl",
    "dev_summary_path": "./MedDialog/filtered_MedDialog/dev-last6_summary.pkl",
    "test_summary_path": "./MedDialog/filtered_MedDialog/test-last6_summary.pkl",

    "rsep_as_associate": False,
})


def train_BertGPT():
    config['expand_token_type_embed'] = False
    config['use_entity_appendix'] = False
    model = BERTGPTEntity(config)

    model.init_bert_gpt_by_pcl_bert()

    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))

    train_dataset = BaseDataset(
        vocab_path=config['vocab_path'],
        data_path=config['train_data_path'],
        data_type="train",
        config=config,
    )

    dev_dataset = BaseDataset(
        vocab_path=config['vocab_path'],
        data_path=config['dev_data_path'],
        data_type="dev",
        config=config
    )

    select_dev_index = pickle.load(open("./MedDialog/filtered_MedDialog/select_dev_index.pkl", 'rb'))
    dev_dataset.data = [i for idx, i in enumerate(dev_dataset.data) if idx in select_dev_index]
    dev_dataset.data = dev_dataset.data[:3000]
    print(f"select {len(dev_dataset.data)} dev data")

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
        save_root="./cikm_save/BertGPT-baseline"
    )

    trainer.train()


def train_HRED():
    config['model_name'] = "HRED"

    config['lr'] = 5e-4
    config['warm_up'] = 1000
    config["batch_size"] = 32
    config["batch_expand_times"] = 1
    config["epoch"] = 20
    # config["state_dict"] = "./cikm_save/HRED_save/epoch90-0427-1651.pt"
    # config['debug'] = True
    # config['use_entity_appendix'] = True
    train_dataset = HREDDataset(
        vocab_path=config['vocab_path'],
        data_path=config['train_data_path'],
        # data_path="./data/cikm/debug_train.pkl",
        data_type="train",
        config=config,
    )

    dev_dataset = HREDDataset(
        vocab_path=config['vocab_path'],
        data_path=config['dev_data_path'],
        # data_path="./data/cikm/debug_dev.pkl",
        data_type="dev",
        config=config
    )

    config['vocab_size'] = len(train_dataset.idx2token)

    model = HRED(config)
    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))

    if config['debug']:
        print("Debug...")
        train_dataset.data = train_dataset.data[:10]
        # dev_dataset.data = dev_dataset.data[:10]
        dev_dataset.data = train_dataset.data
        # pickle.dump(train_dataset.data, open("./data/cikm/debug_train.pkl", 'wb'))
        # pickle.dump(dev_dataset.data, open("./data/cikm/debug_dev.pkl", 'wb'))

    print(len(train_dataset.data))
    print(len(dev_dataset.data))

    trainer = HREDTrainer(
        train_dataset,
        model,
        dev_dataset=dev_dataset,
        config=config,
        save_root="./cikm_save/HRED_save"
    )

    trainer.train()


def train_Seq2Seq():
    config['model_name'] = "Seq2Seq"

    config['lr'] = 5e-4
    config['warm_up'] = 1000
    config["batch_size"] = 32
    config["batch_expand_times"] = 1
    config["epoch"] = 20
    # config['debug'] = True
    # config['use_entity_appendix'] = True
    train_dataset = Seq2SeqDataset(
        vocab_path=config['vocab_path'],
        data_path=config['train_data_path'],
        # data_path="./data/cikm/debug_train.pkl",
        data_type="train",
        config=config,
    )
    dev_dataset = Seq2SeqDataset(
        vocab_path=config['vocab_path'],
        data_path=config['dev_data_path'],
        # data_path="./data/cikm/debug_dev.pkl",
        data_type="dev",
        config=config
    )

    config['vocab_size'] = len(train_dataset.idx2token)

    model = Seq2Seq(config)
    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))

    if config['debug']:
        print("Debug...")
        train_dataset.data = train_dataset.data[:10]
        # dev_dataset.data = dev_dataset.data[:10]
        dev_dataset.data = train_dataset.data
        # pickle.dump(train_dataset.data, open("./data/cikm/debug_train.pkl", 'wb'))
        # pickle.dump(dev_dataset.data, open("./data/cikm/debug_dev.pkl", 'wb'))

    print(len(train_dataset.data))
    print(len(dev_dataset.data))

    trainer = Seq2SeqTrainer(
        train_dataset,
        model,
        dev_dataset=dev_dataset,
        config=config,
        save_root="./cikm_save/Seq2Seq_save"
    )

    trainer.train()


def predict_HRED():
    # if config['use_entity_appendix']:
    #     config['state_dict'] = "./cikm_save/HRED_save/HREDEntityepoch23-B[0.18795]-B1[0.26364]-B4[0.11226].pt"
    # else:
    #     config['state_dict'] = "./cikm_save/HRED_save/HREDepoch21-B[0.11952]-B1[0.17993]-B4[0.05910].pt"

    config['state_dict'] = "./cikm_save/HRED_save/epoch10-B[0.09795]-B1[0.16838]-B4[0.02753].pt"

    config['model_name'] = "HRED"
    test_dataset = HREDDataset(
        vocab_path=config['vocab_path'],
        data_path=config['test_data_path'],
        data_type="test",
        config=config,
    )
    config['vocab_size'] = len(test_dataset.idx2token)
    model = HRED(config)

    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))
    # test_dataset.data = test_dataset.data[:10]
    iterator = test_dataset.get_dataloader(batch_size=1, shuffle=False)

    # generator = BeamSample(config)
    # input_process_func = prepare_input_utils.prepare_input_for_encode_step_HRED

    generator = Greedy(config)
    input_process_func = prepare_input_utils.prepare_input_for_greedy_generate_HRED

    predict_result = generator.greedy_generate(
        prefix_allowed_tokens_fn=None,
        model=model,
        data_iterator=iterator,
        prepare_input_for_encode_step=input_process_func
    )
    predict_result['reference'] = [i['text'][1]['Sentence'] for i in test_dataset.data]

    if 'MedDialog' in config['train_data_path']:
        calculate_scores_and_save(predict_result, config['state_dict'], "HRED",
                                  "./MedDialog_results/baseline_results")
        return

    scores = calculate_BLEU(predict_result['reference'], predict_result['predict'])
    kdm = KD_Metric()
    predict_entities = []
    for i in predict_result['predict']:
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

    save_root = "./cikm_predict_result"
    file_name = "{}-{}-{}.json".format(
        config['model_name'],
        "Entity" if config['use_entity_appendix'] else "Plain",
        time.strftime("%m-%d-%H-%M")
    )
    with open(os.path.join(save_root, file_name), 'w', encoding="utf-8") as writer:
        dump_dict = {
            "score": scores,
            # "generate_args": {
            #     "top_k": config['top_k'],
            #     "beam_size": config['beam_size'],
            # },
            "f1_info": f1_info,
            "predict": [i for i in predict_result['predict']]
        }
        json.dump(dump_dict, writer, ensure_ascii=False)


def train_DialogGPT():
    config['model_name'] = "DialogGPT"
    config['warm_up'] = 5000
    config['lr'] = 4e-5
    config['epoch'] = 20
    # config['state_dict'] = "./cikm_save/GPT_save/epoch20-ACC[0.63316].pt"
    # epoch20-ACC[0.63253].pt
    # epoch20-ACC[0.63316].pt

    model = DialogGPT(config)
    # model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))

    # if config.get('state_dict', None) is not None:
    #     model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))

    train_dataset = GPTDataset(
        vocab_path=config['vocab_path'],
        data_path=config['train_data_path'],
        data_type="train",
        config=config,
    )

    dev_dataset = GPTDataset(
        vocab_path=config['vocab_path'],
        data_path=config['dev_data_path'],
        data_type="dev",
        config=config
    )

    if config['debug']:
        print("Debug...")
        train_dataset.data = train_dataset.data[:10]
        dev_dataset.data = train_dataset.data

    print(len(train_dataset.data))
    print(len(dev_dataset.data))

    trainer = GPTTrainer(
        train_dataset,
        model,
        dev_dataset=dev_dataset,
        config=config,
        save_root="./cikm_save/GPT_save"
    )

    trainer.train()


def predict_DialogGPT():
    # config['length_penalty'] = 1
    # config['min_len'] = 10
    # if config['use_entity_appendix']:
    #     config['state_dict'] = "./cikm_save/GPT_save/epoch20-ACC[0.69329].pt"
    # else:
    #     config['state_dict'] = "./cikm_save/GPT_save/epoch20-ACC[0.69498].pt"
    config['state_dict'] = "./cikm_save/GPT_save/epoch20-ACC[0.49102].pt"

    config['model_name'] = "DialogGPT"
    test_dataset = GPTDataset(
        vocab_path=config['vocab_path'],
        data_path=config['test_data_path'],
        data_type="test",
        config=config,
    )
    # test_dataset.data = test_dataset.data[:20]
    model = DialogGPT(config)
    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))
    # test_dataset.data = test_dataset.data[:10]
    iterator = test_dataset.get_dataloader(batch_size=1, shuffle=False)

    # config['beam_size'] = 3
    # config['min_len'] = 3

    # generator = BeamSample(config)
    # input_process_func = prepare_input_utils.prepare_input_for_encode_step_GPT
    generator = Greedy(config)
    input_process_func = prepare_input_utils.prepare_input_for_greedy_generate_GPT
    # generator = Greedy(config)
    # input_process_func = prepare_input_utils.prepare_input_for_greedy_generate_GPT

    # predict_result = generator.generate(
    predict_result = generator.greedy_generate(
        prefix_allowed_tokens_fn=None,
        model=model,
        data_iterator=iterator,
        prepare_input_for_encode_step=input_process_func
    )

    if 'MedDialog' in config['train_data_path']:
        calculate_scores_and_save(predict_result, config['state_dict'], "HRED",
                                  "./MedDialog_results/baseline_results")
        return

    scores = calculate_BLEU(predict_result['reference'], predict_result['predict'])
    kdm = KD_Metric()
    predict_entities = []
    for i in predict_result['predict']:
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
    # bleu_1 = corpus_bleu(reference_sentence_list, predict_sentence, weights=[1, 0, 0, 0])
    # bleu_4 = corpus_bleu(reference_sentence_list, predict_sentence, weights=[0.25, 0.25, 0.25, 0.25])
    save_root = "./cikm_predict_result"
    file_name = "{}-{}-{}.json".format(
        config['model_name'],
        "Entity" if config['use_entity_appendix'] else "Plain",
        time.strftime("%m-%d-%H-%M")
    )
    with open(os.path.join(save_root, file_name), 'w', encoding="utf-8") as writer:
        dump_dict = {
            "score": scores,
            "f1_info": f1_info,
            "generate_args": {
                "top_k": config['top_k'],
                "beam_size": config['beam_size'],
            },
            "predict": [i for i in predict_result['predict']]
        }
        json.dump(dump_dict, writer, ensure_ascii=False)


def predict_MedDialog_BertGPT():
    config['expand_token_type_embed'] = False
    config['use_entity_appendix'] = False
    config['state_dict'] = "./cikm_save/BertGPT-baseline/epoch8-B[0.27372]-B1[0.45904]-B4[0.08840].pt"

    model = BERTGPTEntity(config)
    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))
    test_dataset = BaseDataset(
        vocab_path=config['vocab_path'],
        data_path=config['test_data_path'],
        data_type="test",
        config=config,
    )

    generator = Greedy(config)
    input_process_func = prepare_input_utils.prepare_input_for_greedy_generate_BertGPT
    iterator = test_dataset.get_dataloader(batch_size=1, shuffle=False)

    predict_result = generator.greedy_generate(
        prefix_allowed_tokens_fn=None,
        model=model,
        data_iterator=iterator,
        prepare_input_for_encode_step=input_process_func
    )
    predict_result['reference'] = [i['text'][1]['Sentence'] for i in test_dataset.data]

    if 'MedDialog' in config['train_data_path']:
        calculate_scores_and_save(predict_result, config['state_dict'], "BertGPT",
                                  "./MedDialog_results/baseline_results")
        print("Distinct-2:")
        print(distinct_n_corpus_level(predict_result['predict'], 2))
        return


def predict_BertGPT():
    if config['use_entity_appendix']:
        config[
            'state_dict'
        ] = "./cikm_save/BertGPT-baseline/BertGPT-Entityepoch22-B[0.19668]-B1[0.26764]-B4[0.12573].pt"
    else:
        config['state_dict'] = "./cikm_save/BertGPT-baseline/BertGPTepoch14-B[0.11407]-B1[0.16539]-B4[0.06275].pt"

    config['beam_size'] = 5
    config['length_penalty'] = 1
    config['no_repeat_ngram_size'] = 4
    config['encoder_no_repeat_ngram_size'] = 7
    config['top_k'] = 40

    model = BERTGPTEntity(config)
    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))

    test_dataset = BaseDataset(
        vocab_path=config['vocab_path'],
        data_path=config['test_data_path'],
        data_type="test",
        config=config,
    )
    # test_dataset.data = test_dataset.data[:10]
    generator = BeamSample(config)
    iterator = test_dataset.get_dataloader(batch_size=1, shuffle=False)
    input_process_func = prepare_input_utils.prepare_input_for_encode_step_BertGPT
    predict_result = generator.generate(
        early_stopping=False,
        prefix_allowed_tokens_fn=None,
        model=model,
        data_iterator=iterator,
        prepare_input_for_encode_step=input_process_func
    )
    predict_result['reference'] = [i['text'][1]['Sentence'] for i in test_dataset.data]
    scores = calculate_BLEU(predict_result['reference'], predict_result['predict'])

    kdm = KD_Metric()
    predict_entities = []
    for i in predict_result['predict']:
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

    save_root = "./cikm_predict_result"
    file_name = "{}-{}-{}.json".format(
        config['model_name'],
        "Entity" if config['use_entity_appendix'] else "Plain",
        time.strftime("%m-%d-%H-%M")
    )
    with open(os.path.join(save_root, file_name), 'w', encoding="utf-8") as writer:
        dump_dict = {
            "score": scores,
            "f1_info": f1_info,
            "generate_args": {
                "top_k": config['top_k'],
                "beam_size": config['beam_size'],
            },
            "predict": [i for i in predict_result['predict']]
        }
        json.dump(dump_dict, writer, ensure_ascii=False)


def predict_Seq2Seq():
    # if config['use_entity_appendix']:
    #     config['state_dict'] = "./cikm_save/Seq2Seq_save/epoch26-B[0.18095]-B1[0.25187]-B4[0.11002].pt"
    # else:
    #     config['state_dict'] = "./cikm_save/Seq2Seq_save/epoch15-B[0.11769]-B1[0.17666]-B4[0.05871].pt"

    config['state_dict'] = "./cikm_save/Seq2Seq_save/epoch15-B[0.09564]-B1[0.16741]-B4[0.02386].pt"

    config['model_name'] = "Seq2Seq"
    test_dataset = Seq2SeqDataset(
        vocab_path=config['vocab_path'],
        data_path=config['test_data_path'],
        data_type="test",
        config=config,
    )
    config['vocab_size'] = len(test_dataset.idx2token)
    model = Seq2Seq(config)

    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))
    # test_dataset.data = test_dataset.data[:10]
    iterator = test_dataset.get_dataloader(batch_size=1, shuffle=False)

    # generator = BeamSample(config)
    # input_process_func = prepare_input_utils.prepare_input_for_encode_step_HRED

    generator = Greedy(config)
    input_process_func = prepare_input_utils.prepare_input_for_greedy_generate_Seq2Seq

    predict_result = generator.greedy_generate(
        prefix_allowed_tokens_fn=None,
        model=model,
        data_iterator=iterator,
        prepare_input_for_encode_step=input_process_func
    )
    predict_result['reference'] = [i['text'][1]['Sentence'] for i in test_dataset.data]

    if 'MedDialog' in config['train_data_path']:
        calculate_scores_and_save(predict_result, config['state_dict'], "Seq2Seq",
                                  "./MedDialog_results/baseline_results")
        return

    scores = calculate_BLEU(predict_result['reference'], predict_result['predict'])

    kdm = KD_Metric()
    predict_entities = []
    for i in predict_result['predict']:
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

    save_root = "./cikm_predict_result"
    file_name = "{}-{}-{}.json".format(
        config['model_name'],
        "Entity" if config['use_entity_appendix'] else "Plain",
        time.strftime("%m-%d-%H-%M")
    )
    with open(os.path.join(save_root, file_name), 'w', encoding="utf-8") as writer:
        dump_dict = {
            "score": scores,
            # "generate_args": {
            #     "top_k": config['top_k'],
            #     "beam_size": config['beam_size'],
            # },
            "f1_info": f1_info,
            "predict": [i for i in predict_result['predict']]
        }
        json.dump(dump_dict, writer, ensure_ascii=False)


def re_blue():
    predict = json.load(open("./cikm_predict_result/06-25-18-53.json", 'r', encoding='utf-8'))
    gold = pickle.load(open("./data/cikm/process_test.pkl", 'rb'))

    predict = predict['predict']
    gold = [i['text'][1]['Sentence'] for i in gold]

    res = calculate_BLEU(gold, predict)
    print(res)


def calculate_scores_and_save(predict_result, model_path, model_name, save_root):
    complete_time = time.strftime("%m-%d-%H-%M")
    file_name = f"{model_name}-{complete_time}.json"
    wo_smooth_scores = sentence_BLEU_avg(predict_result['reference'], predict_result['predict'], use_smooth7=False)
    corpus_scores = calculate_BLEU(predict_result['reference'], predict_result['predict'])
    smooth_scores = sentence_BLEU_avg(predict_result['reference'], predict_result['predict'], use_smooth7=True)
    predict_and_target_pairs = []
    for i, j in zip(predict_result['predict'], predict_result['reference']):
        predict_and_target_pairs.append({"predict": i, "target": j})
    with open(os.path.join(save_root, file_name), 'w', encoding="utf-8") as writer:
        dump_dict = {
            "smooth_scores": smooth_scores,
            "corpus_scores": corpus_scores,
            "wo_smooth_scores": wo_smooth_scores,
            "model_path": model_path,
            "note": ["is_greedy"],
            "results": predict_and_target_pairs
        }
        json.dump(dump_dict, writer, ensure_ascii=False)
    print("-" * 30)
    print("smooth scores")
    for k, v in smooth_scores.items():
        print(f"{k}: {v * 100:.3f}")
    print("-" * 30)
    print("corpus scores")
    for k, v in corpus_scores.items():
        print(f"{k}: {v * 100:.3f}")
    print("-" * 30)
    for k, v in wo_smooth_scores.items():
        print(f"{k}: {v * 100:.3f}")
    print("-" * 30)


if __name__ == '__main__':

    predict_MedDialog_BertGPT()
