from cikm_config import config
import pickle
import json
from collections import Counter


def calculate_entity_retrieve_F1(annotated_test_path):
    config["test_data_path"] = "./data/cikm/response_entity_predict_5-2.pkl"
    retrieved = pickle.load(open(config['test_data_path'], 'rb'))

    annotated_test_data = pickle.load(open(annotated_test_path, 'rb'))
    # predict_json = json.load(open(predict_json_path, 'r', encoding='utf-8'))
    # predict_entity = predict_json['entity_annotation']

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

    for test_item, predict_item in zip(annotated_test_data, retrieved):
        for et in config['entity_type']:
            real_pos_entities = test_item['text'][1][et]
            pred_entities = predict_item['text'][1][et]

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


print(len(pickle.load(open("data/original/new_train.pk", 'rb'))))
