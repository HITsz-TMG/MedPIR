import zipfile

from src.model import BERTGPTEntity
from main_config import config
import torch
from cikm_dataset.dialogue_summary_dataset import DialogueSummaryDataset
from cikm_dataset.summary_response_dataset import SummaryResponseDataset
from cikm_trainer.summary_response_trainer import SummaryResponseTrainer
from cikm_generate_utils.generator import BeamSample, Greedy
from cikm_generate_utils import prepare_input_utils
import pickle
from cikm_model.hir_bertgpt import HireFusionModel
from cikm_model.BERT_Summary_GPT import BertSummaryGpt
from cikm_model.supervised_summary import GenSummaryEntityResponse
from cikm_trainer.supervised_summary_trainer import SupervisedSummaryTrainer
from cikm_trainer.supervised_summary_trainer import predict_GenSummaryEntityResponse
from cikm_trainer.trainer import CIKMTrainer, predict_GenRecallBERTGPT
import time, os
from src.eval.eval_utils import input_entities_cal_F1, get_entity_type, sentence_BLEU_avg, calculate_BLEU, \
    distinct_n_corpus_level
import json
import re


def train_Summary():
    model = BERTGPTEntity(config)
    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))

    train_dataset = DialogueSummaryDataset(
        vocab_path=config['vocab_path'],
        data_path=config['summary_train_path'],
        data_type="train",
        config=config,
    )

    dev_dataset = DialogueSummaryDataset(
        vocab_path=config['vocab_path'],
        data_path=config['summary_dev_path'],
        data_type="dev",
        config=config
    )
    # config['warm_up'] = 1
    # config['lr'] = 1e-4
    # config['epoch'] = 50
    # train_dataset.data = train_dataset.data[:8]
    # dev_dataset.data = train_dataset.data[:8]
    # if config['debug']:
    #     print("Debug...")
    #     train_dataset.data = train_dataset.data[:10]
    #     dev_dataset.data = dev_dataset.data[:10]

    print(len(train_dataset.data))
    print(len(dev_dataset.data))

    trainer = SummaryResponseTrainer(
        train_dataset,
        model,
        dev_dataset=dev_dataset,
        config=config,
        save_root="./cikm_save/SummaryResponse"
    )

    trainer.train()


def generate_one_sample_summary(sample):
    pass


def prepare_summary():
    config['model_name'] = "Summary"

    model = BERTGPTEntity(config)
    model.load_state_dict(torch.load(config['summary_state_dict'], map_location='cpu'))

    dataset = DialogueSummaryDataset(
        vocab_path=config['vocab_path'],
        data_path=config['train_data_path'],
        data_type="test",
        config=config
    )
    # dataset.data = dataset.data[:70000]
    dataset.data = dataset.data[70000:]
    generator = BeamSample(config)
    iterator = dataset.get_dataloader(batch_size=1, shuffle=False)
    input_process_func = prepare_input_utils.prepare_input_for_summary_generation
    predict_result = generator.generate(
        early_stopping=False,
        prefix_allowed_tokens_fn=None,
        model=model,
        data_iterator=iterator,
        prepare_input_for_encode_step=input_process_func
    )
    pickle.dump(predict_result['predict'], open("./data/summary/summary_for_train2-2.pkl", 'wb'))


def train_SummaryResponse():
    hir_attention = False
    bert_summary_gpt = True

    with_entity = True
    with_summary = True
    if hir_attention or bert_summary_gpt:
        with_entity = False
        with_summary = False

    if with_entity:
        config['expand_token_type_embed'] = True

    dev_dataset = SummaryResponseDataset(
        vocab_path=config['vocab_path'],
        data_path=config['dev_data_path'],
        summary_data_path=config['dialogue_summary_dev_path'],
        data_type="dev",
        config=config,
        with_entity=with_entity,
        with_summary=with_summary
    )

    train_dataset = SummaryResponseDataset(
        vocab_path=config['vocab_path'],
        # data_path=config['train_data_path'],
        # summary_data_path=config['dialogue_summary_train_path'],
        data_path=config['dev_data_path'],
        summary_data_path=config['dialogue_summary_dev_path'],
        data_type="train",
        config=config,
        with_entity=with_entity,
        with_summary=with_summary
    )

    # cnt = 0
    # for i in train_dataset.get_dataloader(batch_size=2, shuffle=False, use_hir_attention=False):
    #     """
    #         "history_ids": history_ids,
    #         "history_mask": history_mask,
    #         "history_spk": history_spk,
    #         "target_ids": target_ids,
    #         "target_mask": target_mask,
    #         "summary_end_pos": summary_end_pos,
    #     """
    #     print("-" * 30)
    #     print("".join(train_dataset.convert_ids_to_tokens(i['history_ids'][0].tolist())))
    #     print("".join(train_dataset.convert_ids_to_tokens(i['target_ids'][0].tolist())))
    #     print("-" * 30)
    #     cnt += 1
    #     if cnt > 10:
    #         break

    # config['lr'] = 1e-4
    # config['warm_up'] = 1
    # config['epoch'] = 200
    # config['batch_size'] = 2
    # dev_dataset.data = dev_dataset.data[1000:1200]
    train_dataset.data = dev_dataset.data

    if hir_attention:
        config['model_name'] = "HireFusionModel"
        print("hire model")
        model = HireFusionModel(config)
    elif bert_summary_gpt:
        config['model_name'] = "BertSummaryGpt"
        # config["summary_model_type"] = "bertgpt"
        config["summary_model_type"] = "pointer_net"

        print("BertSummaryGpt model")
        model = BertSummaryGpt(config)
    else:
        model = BERTGPTEntity(config)

    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))

    trainer = SummaryResponseTrainer(
        train_dataset,
        model,
        dev_dataset=dev_dataset,
        config=config,
        save_root="./cikm_save/SummaryAndResponse",
        use_hir_attention=hir_attention,
        use_bert_summary_gpt=bert_summary_gpt
    )
    # trainer.eval(1)

    trainer.train()


def train_GenSummaryEntityResponse():
    DEBUG = True
    config['state_dict'] = "./cikm_save/SummaryAndResponse/epoch25-B[0.21024]-B1[0.32189]-B4[0.09859].pt"
    config['warm_up'] = 0.1
    config['lr'] = 5e-6
    config['epoch'] = 2

    config["sentence=bert[cls]+gat"] = True
    config['recall_gate_network'] = None
    with_entity = False
    with_summary = False

    config['model_name'] = "GenSummaryEntityResponse"
    config['rsep_as_associate'] = False
    config.update({
        'summary_gate_open': False,
        'entity_gate_open': False,
    })
    if config['recall_gate_network'] == "GAT":
        config['batch_size'] = 2
        config['batch_expand_times'] = 16

    # summary_strategy = "last_3_utterance"
    # summary_strategy = "text_rank"
    summary_strategy = "pcl_bert_sim"

    dev_dataset = SummaryResponseDataset(
        vocab_path=config['vocab_path'],
        data_path=config['dev_data_path'],
        summary_data_path=config['dialogue_summary_dev_path'],
        data_type="dev",
        config=config,
        with_entity=with_entity,
        with_summary=with_summary,
        # decoder_with_entity=decoder_with_entity,
        summary_strategy=summary_strategy,
        use_gat=True if config['recall_gate_network'] == "GAT" else False
    )

    test_dataset = None

    train_dataset = SummaryResponseDataset(
        vocab_path=config['vocab_path'],
        data_path=config['train_data_path'] if not DEBUG else config['dev_data_path'],
        summary_data_path=config['dialogue_summary_train_path'] if not DEBUG else config['dialogue_summary_dev_path'],
        # data_path=config['dev_data_path'],
        # summary_data_path=config['dialogue_summary_dev_path'],
        data_type="train" if not DEBUG else "dev",
        config=config,
        with_entity=with_entity,
        with_summary=with_summary,
        # decoder_with_entity=decoder_with_entity,
        summary_strategy=summary_strategy,
        use_gat=True if config['recall_gate_network'] == "GAT" else False
    )

    if DEBUG:
        train_dataset.data = train_dataset.data[:20]
        dev_dataset.data = dev_dataset.data[:100]
        if test_dataset is not None:
            test_dataset.data = test_dataset.data[:10]

        config['epoch'] = 3000
        config['batch_size'] = 2
        config['batch_expand_times'] = 1
        # config['lr'] = 1e-3
        config['warm_up'] = 1

    # cnt = 0
    # for i in train_dataset.get_dataloader(batch_size=2, shuffle=False, use_hir_attention=False):
    #     print("-" * 30)
    #     response_start_pos = i['response_start_pos'][0]
    #     print("".join(train_dataset.convert_ids_to_tokens(i['history_ids'][0].tolist())))
    #     print("".join(train_dataset.convert_ids_to_tokens(i['target_ids'][0].tolist())))
    #     print("".join(train_dataset.convert_ids_to_tokens(i['target_ids'][0][response_start_pos:].tolist())))
    #     print("".join(train_dataset.convert_ids_to_tokens(i['summary_ids'][0].tolist())))
    #     print("".join(train_dataset.convert_ids_to_tokens(i['entity_ids'][0].tolist())))
    #     print("-" * 30)
    #     cnt += 1
    #     if cnt > 10:
    #         break

    model = GenSummaryEntityResponse(config)

    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))

    trainer = SupervisedSummaryTrainer(
        train_dataset,
        model,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        config=config,
        save_root="./cikm_save/SummaryAndResponse",
    )
    # trainer.eval(1)

    trainer.train()


def train_RecallBERTGPT():
    DEBUG = False

    model_name = {
        "wo_entity": "/cikm_save/RecallBertGpt/epoch2-B[0.28553]-B1[0.47866]-B4[0.09241].pt",
        "entity": "/cikm_save/RecallBertGpt/epoch2-B[0.34212]-B1[0.57352]-B4[0.11073].pt",
        "wo_entity_2": "/cikm_save/RecallBertGpt/epoch10-B[0.29521]-B1[0.49484]-B4[0.09558].pt",

        "weight": "./cikm_save/RecallBertGpt/epoch1-B[0.28027]-B1[0.47000]-B4[0.09055].pt",

        "weight2": "/cikm_save/RecallBertGpt/epoch1-B[0.35253]-B1[0.59109]-B4[0.11398].pt"
    }

    config['state_dict'] = model_name["weight2"]
    with_entity = True  # History with entity

    # config["test_data_path"] = "./data/cikm/response_entity_predict_new-0.35.pkl"
    config['model_name'] = "BERTGPTEntity"
    config['entity_appendix'] = True
    config['expand_token_type_embed'] = False
    config['recall'] = True
    config['rsep_as_associate'] = False
    config['lr'] = 1e-5
    config['warm_up'] = 1
    config['batch_size'] = 8
    config['batch_expand_times'] = 4
    with_summary = True  # Decoder with summary
    summary_strategy = 'pcl_bert_sim'
    dev_dataset = SummaryResponseDataset(
        vocab_path=config['vocab_path'],
        data_path=config['dev_data_path'],
        summary_data_path=config['dialogue_summary_dev_path'],
        data_type="dev",
        config=config,
        with_entity=with_entity,
        with_summary=with_summary,
        # decoder_with_entity=decoder_with_entity,
        summary_strategy=summary_strategy,
        use_gat=False
    )

    dev_dataset.data = dev_dataset.data[:2000]

    train_dataset = SummaryResponseDataset(
        vocab_path=config['vocab_path'],
        data_path=config['train_data_path'] if not DEBUG else config['dev_data_path'],
        summary_data_path=config['dialogue_summary_train_path'] if not DEBUG else config['dialogue_summary_dev_path'],
        # data_path=config['dev_data_path'],
        # summary_data_path=config['dialogue_summary_dev_path'],
        data_type="train" if not DEBUG else "dev",
        config=config,
        with_entity=with_entity,
        with_summary=with_summary,
        # decoder_with_entity=decoder_with_entity,
        summary_strategy=summary_strategy,
        use_gat=False
    )
    test_dataset = None
    if DEBUG:
        train_dataset.data = train_dataset.data[:1]
        dev_dataset.data = train_dataset.data
        if test_dataset is not None:
            test_dataset.data = test_dataset.data[:1]

        config['epoch'] = 3000
        config['batch_size'] = 1
        config['batch_expand_times'] = 1
        config['lr'] = 1e-3
        config['warm_up'] = 1

    # cnt = 0
    # for i in train_dataset.get_dataloader(batch_size=2, shuffle=False, use_hir_attention=False):
    #     print("-" * 30)
    #     response_start_pos = i['response_start_pos'][0]
    #     print("".join(train_dataset.convert_ids_to_tokens(i['history_ids'][0].tolist())))
    #     print("".join(train_dataset.convert_ids_to_tokens(i['target_ids'][0].tolist())))
    #     print("".join(train_dataset.convert_ids_to_tokens(i['target_ids'][0][response_start_pos:].tolist())))
    #     print("".join(train_dataset.convert_ids_to_tokens(i['summary_ids'][0].tolist())))
    #     print("".join(train_dataset.convert_ids_to_tokens(i['entity_ids'][0].tolist())))
    #     print("-" * 30)
    #     cnt += 1
    #     if cnt > 10:
    #         break

    # config['lr'] = 1e-4
    # config['warm_up'] = 1
    # config['epoch'] = 200
    # dev_dataset.data = dev_dataset.data[10:30]
    # train_dataset.data = dev_dataset.data

    model = BERTGPTEntity(config)

    if config.get('state_dict', None) is not None:
        print("load {} ".format(config['state_dict']))
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))

    trainer = CIKMTrainer(
        train_dataset,
        model,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        config=config,
        save_root="./cikm_save/RecallBertGpt",
    )
    # trainer.eval(1)

    trainer.train()


def train_MedDialog_SummaryResponse():
    # config['batch_size'] = 2

    train_data_path = "./MedDialog/filtered_MedDialog/train_pairs.pkl"
    dev_data_path = "./MedDialog/filtered_MedDialog/dev_pairs.pkl"
    test_data_path = "./MedDialog/filtered_MedDialog/test_pairs.pkl"

    train_summary_path = "./MedDialog/filtered_MedDialog/train-last6_summary.pkl"
    dev_summary_path = "./MedDialog/filtered_MedDialog/dev-last6_summary.pkl"
    test_summary_path = "./MedDialog/filtered_MedDialog/test-last6_summary.pkl"

    # train_data_path = dev_data_path
    # train_summary_path = dev_summary_path
    config['model_recall_strategy'] = "jointly"

    ########################*****########################*****########################
    # config['state_dict'] = "./cikm_save/MedDialog_save/12-09/epoch2_1-B[0.26567]-B1[0.43861]-B4[0.09273].pt"
    config['recall_gate_network'] = "GAT"
    # config['rec_weight'] = 0.2

    DEBUG = False
    if DEBUG:
        # config['batch_size'] = 2
        config['state_dict'] = None
        config['warm_up'] = 0.1
        config['batch_size'] = 6
        # config['lr'] = 1.5e-5
        # config['epoch'] = 5
        train_data_path = "./MedDialog/debug_pairs.pkl"
        train_summary_path = "./debug-MedDialog-summary.pkl"
        dev_data_path = "./MedDialog/debug_pairs.pkl"
        dev_summary_path = "./debug-MedDialog-summary.pkl"

    # config['train_summary_path'] = train_summary_path
    # config['dev_summary_path'] = dev_summary_path
    # config['recall_gate_network'] = None
    with_entity = False  # History with entity
    with_summary = True  # Decoder with summary
    # decoder_with_entity = True
    config['model_name'] = "GenSummaryEntityResponse"
    config['rsep_as_associate'] = False
    config.update({
        'summary_gate_open': True,
        'entity_gate_open': False,
    })

    print(f"\nwith_summary: {with_summary}\nsummary_gate_open: {config['summary_gate_open']}\n")

    summary_strategy = "pcl_bert_sim"

    if with_summary:
        # save_root = "./cikm_save/MedDialog_save"
        save_root = "./cikm_save/MedDialog_save/12-13"
    else:
        save_root = "./cikm_save/MedDialog_wo_recall_save"

    dev_dataset = SummaryResponseDataset(
        vocab_path=config['vocab_path'],
        data_path=dev_data_path,
        data=None,
        summary_data_path=dev_summary_path,
        data_type="dev",
        config=config,
        with_entity=with_entity,
        with_summary=with_summary,
        summary_strategy=summary_strategy,
        use_gat=True if config['recall_gate_network'] == "GAT" else False,
        dataset_name="MedDialog"
    )
    select_dev_index = pickle.load(open("./MedDialog/filtered_MedDialog/select_dev_index.pkl", 'rb'))
    dev_dataset.data = [i for idx, i in enumerate(dev_dataset.data) if idx in select_dev_index]
    # dev_dataset.data = dev_dataset.data[:10]
    print(f"select {len(dev_dataset.data)} dev data")
    decode_len = []
    for i in dev_dataset.data:
        target_ids, summary_end_pos, response_start_pos, prefix = dev_dataset.decoder_inputs(
            [i], with_summary=with_summary, decoder_with_entity=with_entity)
        decode_len.append(len(target_ids[0]))
    data_with_len = [(i, j) for i, j in zip(dev_dataset.data, decode_len)]
    sorted(data_with_len, key=lambda x: x[1], reverse=True)
    dev_dataset.data = [i for i, _ in data_with_len]

    # if not DEBUG:
    #     filtered_data = []
    #     for i in eval_dev_index:
    #         filtered_data.append(dev_dataset.data[i])
    #     dev_dataset.data = filtered_data
    # print(len(dev_dataset))
    test_dataset = None

    if not DEBUG:
        train_dataset = SummaryResponseDataset(
            vocab_path=config['vocab_path'],
            data_path=train_data_path if not DEBUG else dev_data_path,
            summary_data_path=train_summary_path if not DEBUG else dev_summary_path,
            # data_path=dev_data_path,
            # summary_data_path=dev_summary_path,
            data_type="train" if not DEBUG else "dev",
            config=config,
            with_entity=with_entity,
            with_summary=with_summary,
            summary_strategy=summary_strategy,
            use_gat=True if config['recall_gate_network'] == "GAT" else False,
            dataset_name="MedDialog"
        )
    else:
        train_dataset = dev_dataset

    if DEBUG:
        train_dataset.data = train_dataset.data[:100]
        dev_dataset.data = dev_dataset.data[:100]
        if test_dataset is not None:
            test_dataset.data = test_dataset.data[:10]

    model = GenSummaryEntityResponse(config)

    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))

    trainer = SupervisedSummaryTrainer(
        train_dataset,
        model,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        config=config,
        save_root=save_root,
    )

    trainer.train()


@torch.no_grad()
def predict_MedDialog():
    test_data_path = "./MedDialog/filtered_MedDialog/history_response_pairs_data/test_pairs.pkl"
    test_summary_path = "./MedDialog/filtered_MedDialog/test-last6_summary.pkl"

    config['state_dict'] = "./cikm_save/MedDialog_save/meddialog.pt"

    config.update({
        "model_recall_strategy": "jointly",
        "use_token_type_ids": True,
    })

    config['recall_gate_network'] = "GAT"
    with_entity = False
    with_summary = True
    config['model_name'] = "GenSummaryEntityResponse"
    config['rsep_as_associate'] = False
    config.update({
        'summary_gate_open': True,
        'entity_gate_open': True,
    })
    summary_strategy = "pcl_bert_sim"

    set_prefix = False

    two_processor = True
    print(f"set_prefix: {set_prefix}")
    print(f"use_beam_sample: {config['beam_sample']}")

    note = {"set_prefix": set_prefix, "strategy": "beam_sample" if config['beam_sample'] else "greedy",
            "two_processor": two_processor}

    model = GenSummaryEntityResponse(config)
    model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))
    model.with_target_recall = False
    model.eval()

    test_dataset = SummaryResponseDataset(
        vocab_path=config['vocab_path'],
        data_path=test_data_path,
        summary_data_path=test_summary_path,
        data_type="test",
        config=config,
        with_entity=with_entity,
        with_summary=with_summary,
        summary_strategy=summary_strategy,
        use_gat=True if config['recall_gate_network'] == "GAT" else False,
        dataset_name="MedDialog"
    )
    decode_len = []
    for i in test_dataset.data:
        target_ids, summary_end_pos, response_start_pos, prefix = test_dataset.decoder_inputs(
            [i], with_summary=with_summary, decoder_with_entity=with_entity)
        decode_len.append(len(target_ids[0]))
    data_with_len = [(i, j) for i, j in zip(test_dataset.data, decode_len)]
    sorted(data_with_len, key=lambda x: x[1], reverse=True)
    test_dataset.data = [i for i, _ in data_with_len]

    if config['beam_sample']:
        if config['start'] is not None and config['end'] is not None:
            start, end = config['start'], config['end']
        else:
            start, end = 0, len(test_dataset)
        test_dataset.data = test_dataset.data[start:end]
        print(f"Total {len(test_dataset)}, beam sample from {start} to {end}")
        note['start'], note['end'] = start, end
        note['beam_size'] = config['beam_size']
        note['top_k'] = config['top_k']

    if not set_prefix and not config['beam_sample']:
        iterator = test_dataset.get_dataloader(batch_size=100, shuffle=False)
    else:
        iterator = test_dataset.get_dataloader(batch_size=1, shuffle=False)

    trainer = SupervisedSummaryTrainer(test_dataset, model, dev_dataset=test_dataset, test_dataset=test_dataset,
                                       config=config, save_root="./cikm_save/MedDialog_save")

    if config['beam_sample']:
        print("Start BeamSample Generating")
        gt = BeamSample(config)
        gen_func = gt.with_prefix_generate
        input_process_func = prepare_input_utils.prepare_input_for_GenSummaryEntityResponse
        predict_result = gen_func(
            early_stopping=False,
            prefix_allowed_tokens_fn=None,
            model=model,
            data_iterator=iterator,
            prepare_input_for_encode_step=input_process_func,
            gen_prefix=True
        )
    else:
        print("Start Greedy Generate")
        gt = Greedy(config)
        predict_result = gt.greedy_generate(
            prefix_allowed_tokens_fn=None,
            model=model,
            data_iterator=iterator,
            prepare_input_for_encode_step=trainer.prepare_input_for_greedy_generate,
            with_prefix=set_prefix,
            two_processor=two_processor
        )

    predict = predict_result['predict']
    json.dump(predict_result, open("./MedDialog_results/tmp.json", 'w', encoding='utf-8'), ensure_ascii=False)
    predict_result['predict'] = trainer.select_response_from_result(predict)
    predict_result['reference'] = [i['text'][1]['Sentence'] for i, _ in test_dataset.data]

    calculate_scores_and_save(predict_result, config['state_dict'], config['model_name'],
                              "./MedDialog_results", note=note)
    print("Distinct-2:")
    print(distinct_n_corpus_level(predict_result['predict'], 2))


def calculate_scores_and_save(predict_result, model_path, model_name, save_root, note=None):
    complete_time = time.strftime("%m-%d-%H-%M")

    if isinstance(note, dict) and "start" in note.keys():
        file_name = f"{model_name}-{complete_time}-[{note['start']}:{note['end']}].json"
    else:
        file_name = f"{model_name}-{complete_time}.json"

    wo_smooth_scores = sentence_BLEU_avg(predict_result['reference'], predict_result['predict'], use_smooth7=False)
    corpus_scores = calculate_BLEU(predict_result['reference'], predict_result['predict'])
    smooth_scores = sentence_BLEU_avg(predict_result['reference'], predict_result['predict'], use_smooth7=True)
    f1, recall, precision = calculate_MedDialog_F1(predict_result)
    predict_and_target_pairs = []
    for i, j in zip(predict_result['predict'], predict_result['reference']):
        predict_and_target_pairs.append({"predict": i, "target": j})
    with open(os.path.join(save_root, file_name), 'w', encoding="utf-8") as writer:
        dump_dict = {
            "smooth_scores": smooth_scores,
            "corpus_scores": corpus_scores,
            "wo_smooth_scores": wo_smooth_scores,
            "F1": f1,
            "R": recall,
            "P": precision,
            "model_path": model_path,
            "note": note,
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
    print(f"F1:{f1 * 100:.5f}, Recall:{recall * 100:.5f}, Precision:{precision * 100:.5f}")
    print("-" * 30)


def combine_test(paths):
    files = [json.load(open(i, 'r', encoding='utf-8')) for i in paths]
    predict_result = {"reference": [], "predict": []}
    for file in files:
        for result in file['results']:
            predict_result["reference"].append(result["target"])
            predict_result["predict"].append(result["predict"])
    calculate_scores_and_save(predict_result, files[0]["model_path"], "Combine", "./MedDialog_results")
    # (predict_result, model_path, model_name, save_root, note=None)


def load_result_evaluate():
    paths = ["./MedDialog_results/meddialog_result.json"]
    for path in paths:
        print("-" * 50)
        if os.path.exists(path + ".update"):
            file = json.load(open(path + ".update", 'r', encoding='utf-8'))
        else:
            file = json.load(open(path, 'r', encoding='utf-8'))
        print(f"Result is produced by: {file['model_path']}")

        predict_result = {"reference": [], "predict": []}
        for result in file['results']:
            predict_result["reference"].append(result["target"])
            predict_result["predict"].append(result["predict"])

        print(f"B@1: {file['smooth_scores']['BLEU-1']:.5f}, "
              f"B@2: {file['smooth_scores']['BLEU-2']:.5f}, "
              f"B@4: {file['smooth_scores']['BLEU-4']:.5f}")

        if "F1" not in file.keys():
            f1, recall, precision = calculate_MedDialog_F1(predict_result)
            file["F1"] = f1
            file["R"] = recall
            file["P"] = precision
            json.dump(file, open(path + ".update", 'w', encoding='utf-8'), ensure_ascii=False)
        else:
            f1, recall, precision = file["F1"], file["R"], file["P"]
        print(f"F1:{f1 * 100:.5f}, Recall:{recall * 100:.5f}, Precision:{precision * 100:.5f}")

        if "D2" not in file.keys():
            d2 = distinct_n_corpus_level(predict_result['predict'], 2)
            json.dump(file, open(path + ".update", 'w', encoding='utf-8'), ensure_ascii=False)
        else:
            d2 = file['D2']

        print(f"Distinct-2: {d2}")
        print("-" * 50)


def calculate_MedDialog_F1(predict_result):
    all_entity_triples_root = "./cmekg/all_entity_triples"
    crawled_entity_path = list(os.walk(all_entity_triples_root))[0][2]
    crawled_entity = [_[:-5] for _ in crawled_entity_path]
    crawled_entity = list(set(crawled_entity))

    pred_pos_num, real_pos_num, pred_pos_correct_num = 0, 0, 0

    for i, j in zip(predict_result['reference'], predict_result['predict']):
        ref_entities = [e for e in crawled_entity if e in i]
        prd_entities = [e for e in crawled_entity if e in j]
        correct_entities = [e for e in prd_entities if e in ref_entities]

        pred_pos_num += len(prd_entities)
        real_pos_num += len(ref_entities)
        pred_pos_correct_num += len(correct_entities)

    precision = pred_pos_correct_num / pred_pos_num if pred_pos_num != 0 else 0
    recall = pred_pos_correct_num / real_pos_num if real_pos_num != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return f1, recall, precision


def calculate_vrbot_result():
    path = "./vbot/VRBot-master/data/cache/VRBOT/b0b2f31f/10-100000-test-(0.1353-0.0551-0.0311-0.0219).txt"

    ref_path = "./MedDialog/filtered_MedDialog/raw/meddialog_test.zip"
    test_root = "./MedDialog/filtered_MedDialog/raw"
    fi = json.load(open(path, 'r', encoding='utf-8'))

    zip_file = zipfile.ZipFile(ref_path)
    zip_namelist = list(filter(lambda x: x.endswith("json"), list(zip_file.namelist())))
    a = list(zip_namelist)
    b = [int(i.split("/")[1].split('.')[0]) for i in a]
    tt = [(i, j) for i, j in zip(a, b)]
    tt = sorted(tt, key=lambda x: x[1])
    zip_namelist = [i[0] for i in tt]

    session_dict = dict()
    for session in fi:
        session_name = session['session'][0]['session_name'][0]
        session_dict[session_name] = session['session']

    reference_dict = dict()
    for zip_n in zip_namelist:
        session_name = zip_n
        ref_path = os.path.join(test_root, session_name)
        session = json.load(open(ref_path, 'r', encoding='utf-8'))
        reference_dict[session_name] = session

    assert len(reference_dict) == len(session_dict)

    res = {"predict": [], "reference": []}
    for session_name in reference_dict.keys():
        pred_n = 0
        for reference_turn in reference_dict[session_name]['dialogues']:
            if reference_turn['role'] == 'doctor':
                predict_turn = session_dict[session_name][pred_n]
                pred_n += 1
                pred_sentence = predict_turn['hyp'].replace(" ", "")
                ref_sentence = reference_turn['sentence']

                res["predict"].append(pred_sentence)
                res["reference"].append(ref_sentence)
        assert pred_n == len(session_dict[session_name])

    calculate_scores_and_save(res, path, "VRBOT", "./MedDialog_results/baseline_results")
    return res


def after_processing_and_evaluate():
    path = "./MedDialog_results/BERTGPTEntity-01-11-21-31.json"

    data = json.load(open(path, 'r', encoding='utf-8'))
    predict_result = {"reference": [], "predict": []}
    for result in data['results']:
        max_len = 50
        predict_result["reference"].append(result["target"])
        pred = result["predict"][:max_len]
        predict_result["predict"].append(pred)
    smooth_scores = sentence_BLEU_avg(predict_result['reference'], predict_result['predict'], use_smooth7=True)
    print("-" * 30)
    print("smooth scores")
    for k, v in smooth_scores.items():
        print(f"{k}: {v * 100:.3f}")
    print("-" * 30)
    print(distinct_n_corpus_level(predict_result['predict'], 2))
    print("-" * 30)
    f1, recall, precision = calculate_MedDialog_F1(predict_result)

    print(f"F1:{f1 * 100:.5f}, Recall:{recall * 100:.5f}, Precision:{precision * 100:.5f}")
    print("-" * 30)
    print(distinct_n_corpus_level(predict_result['predict'], 2))
    print("-" * 30)


if __name__ == '__main__':
    # train_Summary()
    # prepare_summary()
    # train_SummaryResponse()

    # train_GenSummaryEntityResponse()
    # train_RecallBERTGPT()
    # predict_GenSummaryEntityResponse(config)
    # predict_GenRecallBERTGPT(config)

    # train_MedDialog_SummaryResponse()

    # res = json.load(open("./MedDialog_results/baseline_results/Seq2Seq-12-09-10-46.json", 'r', encoding='utf-8'))
    # predict_result = {"predict": [], "reference": []}
    # for i in res['results']:
    #     predict_result["predict"].append(i['predict'])
    #     predict_result["reference"].append(i['target'])
    # calculate_scores_and_save(predict_result, "None", "None", "./MedDialog_results/baseline_results")

    predict_MedDialog()
    # after_processing_and_evaluate()
