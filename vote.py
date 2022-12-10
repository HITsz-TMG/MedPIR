# if input("160 entity?  [y/n]:") == 'y':
from vote_config import config
# else:
#     from entity_config import config
from src.dataset import MedDGDataset
from src.entity_trainer import EntityPredTrainer
from src.model import NextEntityPredict as Model
import torch
import importlib
import random
import pickle
from tqdm import tqdm

model_list = ["./save/EntityPred_save/8-{}.NextEntityPredict".format(i) for i in range(1, 9)]
test_data_path = "./data/cikm/test-4-25-input.pkl"


def get_prob_list():
    prob_list = []
    for m in model_list:
        config['model_class'] = Model
        model = Model(config=config)
        model.load_state_dict(torch.load(m, map_location='cpu'))
        test_dataset = MedDGDataset(config['original_train_data_path'], config['vocab_path'],
                                    data_path=config['test_data_path'], data_type='test',
                                    preprocess=config['preprocessing'], config=config)
        test_data_loader = test_dataset.get_dataloader(batch_size=1, shuffle=False)
        if torch.cuda.is_available():
            model = model.to('cuda')
        model.eval()
        prob_item = {}
        for idx, item in enumerate(tqdm(test_data_loader)):
            history_ids = item['history_ids'].to(model.device)
            history_mask = item['history_mask'].to(model.device)
            token_type_ids = item['history_speaker'].to(model.device)
            with torch.no_grad():
                topic_probs, five_topic_probs, entity_loss = model(
                    input_ids=history_ids,
                    attention_mask=history_mask,
                    token_type_ids=token_type_ids,
                )
            five_topic_probs = [p[0].tolist() for p in five_topic_probs]
            prob_item[idx] = five_topic_probs
        prob_list.append(prob_item)
    pickle.dump(prob_list, open("./data/cikm/8_prob_list-4-25-test.pkl", "wb"))
    return prob_list


def save_result(prob_list, threshold):
    print(threshold)
    prob_order = ['Symptom', 'Medicine', 'Test', 'Attribute', 'Disease']
    eid2ent = [
        config['symptom'],
        config['medicine'],
        config['test'],
        config['attribute'],
        config['disease']
    ]
    test_dataset = MedDGDataset(config['original_train_data_path'], config['vocab_path'],
                                data_path=config['test_data_path'], data_type='test',
                                preprocess=config['preprocessing'], config=config)
    test_data_loader = test_dataset.get_dataloader(batch_size=1, shuffle=False)
    test_data = test_dataset.data
    total_num = len(prob_list)
    vote_num = int((total_num + 1) / 2)
    pred_entity_num = 0
    cur_all_ent = []
    for idx, item in enumerate(tqdm(test_data_loader)):
        if test_data[idx]['text'][1] is None:
            test_data[idx]['text'][1] = dict()
        for et in config['entity_type']:
            test_data[idx]['text'][1][et] = []

        idx_prob = [p[idx] for p in prob_list]
        for cate_idx, cate in enumerate(prob_order):
            cur_cate_prob_list = [x[cate_idx] for x in idx_prob]
            for ent_id in range(len(cur_cate_prob_list[0])):
                right_entity_num = 0
                for ent in cur_cate_prob_list:
                    ent_prob = ent[ent_id]
                    if ent_prob > threshold:
                        right_entity_num += 1

                if right_entity_num >= vote_num:
                    test_data[idx]['text'][1][cate].append(
                        eid2ent[cate_idx][ent_id]
                    )
                    cur_all_ent.append(eid2ent[cate_idx][ent_id])
                    pred_entity_num += 1

    pickle.dump(test_data, open("./data/4-18-data/8_test2_add_spk_and_entity-{}-160.pkl".format(threshold), 'wb'))
    print("pred entity num: {}".format(pred_entity_num))


if __name__ == '__main__':
    import pickle

    prob_list = pickle.load(open("./data/4-18-data/prob_list.pkl", "rb"))
    save_result(prob_list, 0.3)
