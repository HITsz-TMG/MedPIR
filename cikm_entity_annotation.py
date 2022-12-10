from cikm_config import config
import pickle
from src.entity_predict import EntityPredict
from cikm_dataset.entity_annotation_dataset import EntityAnnotationDataset
from src.entity_trainer import EntityPredTrainer
import os
import torch
from tqdm import tqdm
import json

config['model_class'] = EntityPredict


def create_k_fold_data(k_fold):
    all_data = pickle.load(open("./data/cikm/entity-annotation/sentence-and-entities.pkl", 'rb'))
    k_fold_data = []
    piece_num = len(all_data) // k_fold
    for k in range(k_fold):
        start = k * piece_num
        end = (k + 1) * piece_num
        if k == k_fold - 1:
            end = len(all_data)
        print(start, end)
        k_fold_data.append(all_data[start:end])
    return k_fold_data


def train_k_fold_model():
    k_fold = 8
    k_fold_data = create_k_fold_data(k_fold)
    config['warm_up'] = 1000
    config['epoch'] = 4
    config['lr'] = 2e-5
    config.update({
        "batch_size": 32,
        "batch_expand_times": 1,
    })
    for k in range(k_fold):

        model = EntityPredict(config=config, entity_type_num=config['entity_type_num'])
        tmp_train_data = []
        for kk in range(k_fold):
            if kk != k:
                tmp_train_data = tmp_train_data + list(k_fold_data[kk])
        dev_data = k_fold_data[k]
        train_dataset = EntityAnnotationDataset(
            vocab_path=config['vocab_path'],
            data_type="train",
            config=config,
            data=tmp_train_data
        )
        dev_dataset = EntityAnnotationDataset(
            vocab_path=config['vocab_path'],
            data_type="dev",
            config=config,
            data=dev_data
        )
        print("Entity Annotation Model Training FOLD-{}".format(k))
        print("train len: {}, dev len: {}".format(len(train_dataset), len(dev_dataset)))

        trainer = EntityPredTrainer(train_dataset, model, dev_dataset=dev_dataset, config=config,
                                    save_root="./cikm_save/Entity-annotation-model")
        trainer.train()


def get_entity_type(ent):
    ent = ent.lower()
    if ent in config['symptom']:
        return "Symptom"
    elif ent in config['medicine']:
        return "Medicine"
    elif ent in config["test"]:
        return "Test"
    elif ent in config['attribute']:
        return "Attribute"
    elif ent in config['disease']:
        return "Disease"
    else:
        raise RuntimeError("Entity type not found")


@torch.no_grad()
def annotate_test_from_k_fold_model(dump_path):
    test_data = pickle.load(open(config['test_data_path'], 'rb'))
    dataset_class = EntityAnnotationDataset(
        vocab_path=config['vocab_path'],
        data_type="train",
        config=config,
    )
    root_path = "./cikm_save/Entity-annotation-model"
    model_list = list(os.walk(root_path))[0][2]
    model_list = [os.path.join(root_path, _) for _ in model_list]
    vote_list = [[] for _ in range(len(model_list))]
    prob_order = ['Symptom', 'Medicine', 'Test', 'Attribute', 'Disease']
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_id, model_path in enumerate(model_list):
        model = EntityPredict(config=config, entity_type_num=config['entity_type_num'])
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = model.to(device)
        model.eval()

        for test_item in tqdm(test_data, desc="Model-{}".format(model_id)):
            sentence = test_item['text'][1]['Sentence']
            input_ids = dataset_class.convert_to_input_ids_for_annotation(sentence)
            input_ids = torch.tensor([input_ids]).to(device)
            topic_probs, five_topic_probs, loss = model(input_ids=input_ids)
            vote_list[model_id].append(topic_probs[0].tolist())

    vote2pos_model_num = int(len(model_list) / 2)
    # print(vote2pos_model_num)
    for test_id in range(len(test_data)):
        for et in config['entity_type']:
            assert test_data[test_id]['text'][1][et] is None
            test_data[test_id]['text'][1][et] = []
        cur_prob_list = [mv[test_id] for mv in vote_list]
        for eid in range(len(config['entity'])):
            pos_num = sum([1 if m[eid] > 0.5 else 0 for m in cur_prob_list])
            if pos_num >= vote2pos_model_num:
                entity = config['entity'][eid]
                entity_type = get_entity_type(entity)
                test_data[test_id]['text'][1][entity_type].append(entity)

    pickle.dump(test_data, open(dump_path, 'wb'))


@torch.no_grad()
def annotate_predict_result_entity_from_k_fold_model(result_json_path, dump_path):
    predict_result_json = json.load(open(result_json_path, 'r', encoding='utf-8'))
    predict_result = predict_result_json['predict']
    dataset_class = EntityAnnotationDataset(
        vocab_path=config['vocab_path'],
        data_type="train",
        config=config,
    )
    root_path = "./cikm_save/Entity-annotation-model"
    model_list = list(os.walk(root_path))[0][2]
    model_list = [os.path.join(root_path, _) for _ in model_list]
    vote_list = [[] for _ in range(len(model_list))]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_id, model_path in enumerate(model_list):
        model = EntityPredict(config=config, entity_type_num=config['entity_type_num'])
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = model.to(device)
        model.eval()

        for sentence in tqdm(predict_result, desc="Model-{}".format(model_id)):
            input_ids = dataset_class.convert_to_input_ids_for_annotation(sentence)
            input_ids = torch.tensor([input_ids]).to(device)
            topic_probs, five_topic_probs, loss = model(input_ids=input_ids)
            vote_list[model_id].append(topic_probs[0].tolist())

    vote2pos_model_num = int(len(model_list) / 2)
    predict_result_entity = []
    for sentence_id in range(len(predict_result)):
        predict_result_entity.append(dict([(k, []) for k in config['entity_type']]))
        cur_prob_list = [mv[sentence_id] for mv in vote_list]
        for eid in range(len(config['entity'])):
            pos_num = sum([1 if m[eid] > 0.5 else 0 for m in cur_prob_list])
            if pos_num >= vote2pos_model_num:
                entity = config['entity'][eid]
                entity_type = get_entity_type(entity)
                predict_result_entity[sentence_id][entity_type].append(entity)

    predict_result_json['entity_annotation'] = predict_result_entity

    json.dump(predict_result_json, open(dump_path, 'w', encoding='utf-8'), ensure_ascii=False)


def calculate_F1(annotated_test_path, predict_json_path):
    annotated_test_data = pickle.load(open(annotated_test_path, 'rb'))
    if predict_json_path.endswith("pkl"):
        predict_pkl = pickle.load(open(predict_json_path, 'rb'))
        predict_entity = []
        for i in predict_pkl:
            item = i['text'][1]
            predict_entity.append({
                'Symptom': item['Symptom'],
                'Medicine': item['Medicine'],
                'Attribute': item['Attribute'],
                'Disease': item['Disease'],
                'Test': item['Test']
            })
    elif predict_json_path.endswith("json"):
        predict_json = json.load(open(predict_json_path, 'r', encoding='utf-8'))
        predict_entity = predict_json['entity_annotation']

    """
    if ent in config['symptom']:
        return "Symptom"
    elif ent in config['medicine']:
        return "Medicine"
    elif ent in config["test"]:
        return "Test"
    elif ent in config['attribute']:
        return "Attribute"
    elif ent in config['disease']:
        return "Disease"
    """
    pred_pos_num = 0
    pred_pos_correct_num = 0
    # precision = pred_pos_correct_num/pred_pos_num
    real_pos_num = 0
    # recall = pred_pos_correct_num/real_pos_num
    cnt_dict = dict()
    for et in config['entity_type']:
        cnt_dict[et] = {
            "pred_pos_num": 0,
            "pred_pos_correct_num": 0,
            "real_pos_num": 0
        }

    # entities_num = {et: 0 for et in config['entity_type']}
    # predict_entities_num = {et: 0 for et in config['entity_type']}

    for test_item, predict_item in zip(annotated_test_data, predict_entity):
        for et in config['entity_type']:
            real_pos_entities = test_item['text'][1][et]
            pred_entities = predict_item[et]

            pred_pos_num += len(pred_entities)
            real_pos_num += len(real_pos_entities)
            correct_pred_num = len([1 for i in pred_entities if i in real_pos_entities])
            pred_pos_correct_num += correct_pred_num

            cnt_dict[et]['pred_pos_num'] += len(pred_entities)
            cnt_dict[et]['real_pos_num'] += len(real_pos_entities)
            cnt_dict[et]['pred_pos_correct_num'] += correct_pred_num

    category_F1 = {et: None for et in config['entity_type']}

    for et in config['entity_type']:
        precision = cnt_dict[et]['pred_pos_correct_num'] / cnt_dict[et]['pred_pos_num']
        recall = cnt_dict[et]['pred_pos_correct_num'] / cnt_dict[et]['real_pos_num']
        f1 = (2 * precision * recall) / (precision + recall)
        print("{}: P={:.5f} R={:.5f} F1={:.5f}".format(et, precision, recall, f1))

        category_F1[et] = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'pred_num': cnt_dict[et]['pred_pos_num'],
            'real_num': cnt_dict[et]['real_pos_num']
        }

    precision = pred_pos_correct_num / pred_pos_num
    recall = pred_pos_correct_num / real_pos_num
    f1 = (2 * precision * recall) / (precision + recall)
    print("P={:.5f} R={:.5f} F1={:.5f}".format(precision, recall, f1))

    res = {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'pred_num': pred_pos_num,
        'real_num': real_pos_num,
        'category_f1': category_F1,
    }
    print(res)
    return res


def input_entities_cal_F1(golden_entities, predict_entities):
    pred_pos_num = 0
    pred_pos_correct_num = 0
    # precision = pred_pos_correct_num/pred_pos_num
    real_pos_num = 0
    # recall = pred_pos_correct_num/real_pos_num
    cnt_dict = dict()
    for et in config['entity_type']:
        cnt_dict[et] = {
            "pred_pos_num": 0,
            "pred_pos_correct_num": 0,
            "real_pos_num": 0
        }

    for test_item, predict_item in zip(golden_entities, predict_entities):
        for et in config['entity_type']:
            real_pos_entities = test_item[et]
            pred_entities = predict_item[et]

            pred_pos_num += len(pred_entities)
            real_pos_num += len(real_pos_entities)
            correct_pred_num = len([1 for i in pred_entities if i in real_pos_entities])
            pred_pos_correct_num += correct_pred_num

            cnt_dict[et]['pred_pos_num'] += len(pred_entities)
            cnt_dict[et]['real_pos_num'] += len(real_pos_entities)
            cnt_dict[et]['pred_pos_correct_num'] += correct_pred_num

    category_F1 = {et: None for et in config['entity_type']}

    for et in config['entity_type']:
        precision = cnt_dict[et]['pred_pos_correct_num'] / cnt_dict[et]['pred_pos_num'] if cnt_dict[et][
                                                                                               'pred_pos_num'] != 0 else 0
        recall = cnt_dict[et]['pred_pos_correct_num'] / cnt_dict[et]['real_pos_num'] if cnt_dict[et][
                                                                                            'real_pos_num'] != 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        print("{}: P={:.5f} R={:.5f} F1={:.5f}".format(et, precision, recall, f1))

        category_F1[et] = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'pred_num': cnt_dict[et]['pred_pos_num'],
            'real_num': cnt_dict[et]['real_pos_num']
        }

    precision = pred_pos_correct_num / pred_pos_num
    recall = pred_pos_correct_num / real_pos_num
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    print("P={:.5f} R={:.5f} F1={:.5f}".format(precision, recall, f1))

    res = {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'pred_num': pred_pos_num,
        'real_num': real_pos_num,
        'category_f1': category_F1,
    }
    print(res)
    return res


if __name__ == '__main__':
    annotate_predict_result_entity_from_k_fold_model(
        result_json_path="./cikm_predict_result/Seq2Seq-Plain-05-20-19-37.json",
        dump_path="./cikm_predict_result/Seq2Seq-Plain-05-20-19-37.json"
    )

    calculate_F1(
        "./data/cikm/response_entity_annotation_5-1.pkl",
        "./cikm_predict_result/Seq2Seq-Plain-05-20-19-37.json"
    )
