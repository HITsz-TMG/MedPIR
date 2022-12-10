from cikm_config import config
import pickle
from cikm_model.next_entity_predict import CIKMNextEntityPredict
from cikm_dataset.next_entity_predict_dataset import CIKMNextEntityPredictDataset
from src.entity_trainer import EntityPredTrainer
import torch
import os
from tqdm import tqdm
from cikm_entity_annotation import get_entity_type, calculate_F1

config['model_class'] = CIKMNextEntityPredict


def create_k_fold_data(k_fold):
    data1 = pickle.load(open(config['train_data_path'], 'rb'))
    data2 = pickle.load(open(config['dev_data_path'], 'rb'))
    all_data = data1 + data2
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


def train_next_entity_predict_k_fold():
    k_fold = 10
    k_fold_data = create_k_fold_data(k_fold)
    config['warm_up'] = 1000
    config['epoch'] = 7
    config['lr'] = 2e-5
    config.update({
        "batch_size": 32,
        "batch_expand_times": 1,
    })
    for k in range(k_fold):

        model = CIKMNextEntityPredict(config=config)
        tmp_train_data = []
        for kk in range(k_fold):
            if kk != k:
                tmp_train_data = tmp_train_data + list(k_fold_data[kk])
        dev_data = k_fold_data[k]
        train_dataset = CIKMNextEntityPredictDataset(
            vocab_path=config['vocab_path'],
            data_type="train",
            config=config,
            data=tmp_train_data
        )
        dev_dataset = CIKMNextEntityPredictDataset(
            vocab_path=config['vocab_path'],
            data_type="dev",
            config=config,
            data=dev_data
        )
        print("Entity Annotation Model Training FOLD-{}".format(k))
        print("train len: {}, dev len: {}".format(len(train_dataset), len(dev_dataset)))

        trainer = EntityPredTrainer(train_dataset, model, dev_dataset=dev_dataset, config=config,
                                    save_root="./cikm_save/NextEntity-predict-model")
        trainer.train()


@torch.no_grad()
def predict_from_k_fold_model(dump_path, test_data=None):
    predefine_data = True
    # if test_data is None:
    #     predefine_data = False
    test_data = pickle.load(open(config['test_data_path'], 'rb'))
    test_dataset = CIKMNextEntityPredictDataset(
        vocab_path=config['vocab_path'],
        data_type="test",
        data=test_data,
        config=config,
    )
    root_path = "./cikm_save/NextEntity-predict-model"
    model_list = [
        "epoch5-F10.18414.pt",
        "epoch5-F10.18650.pt",
        "epoch7-F10.19121.pt",
        "epoch6-F10.19051.pt",
        "epoch6-F10.20010.pt",
        "epoch6-F10.19283.pt",
        "epoch7-F10.19487.pt",
        "epoch5-F10.19183.pt",
    ]

    model_list = [os.path.join(root_path, _) for _ in model_list]
    vote_list = [[] for _ in range(len(model_list))]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_id, model_path in enumerate(model_list):
        model = CIKMNextEntityPredict(config=config)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = model.to(device)
        model.eval()

        dataloader = test_dataset.get_dataloader(batch_size=1, shuffle=False)

        for batch in tqdm(dataloader):
            for k in batch:
                batch[k] = batch[k].to(device)
            topic_probs, five_topic_probs, loss = model(**batch)
            vote_list[model_id].append(topic_probs[0].tolist())

    vote2pos_model_num = int(len(model_list) / 2)
    # print(vote2pos_model_num)
    for test_id in range(len(test_data)):
        for et in config['entity_type']:
            if not predefine_data:
                assert test_data[test_id]['text'][1][et] is None
            test_data[test_id]['text'][1][et] = []
        cur_prob_list = [mv[test_id] for mv in vote_list]
        for eid in range(len(config['entity'])):
            pos_num = sum([1 if m[eid] > 0.35 else 0 for m in cur_prob_list])
            if pos_num >= vote2pos_model_num:
                entity = config['entity'][eid]
                entity_type = get_entity_type(entity)
                test_data[test_id]['text'][1][entity_type].append(entity)

    pickle.dump(test_data, open(dump_path, 'wb'))

    calculate_F1(config['golden_test_data_path'], dump_path)


if __name__ == '__main__':
    # predict_from_k_fold_model("./data/cikm/response_entity_predict_0.35_5-3.pkl")
    # data = pickle.load(open(config['dev_data_path'], 'rb'))
    # predict_from_k_fold_model("./data/cikm/dev_data_with_predict_next_entities_0.35-5-5.pkl",
    #                           test_data=data)
    predict_from_k_fold_model("./data/cikm/response_entity_predict_new-0.35.pkl")
    # calculate_F1(config['golden_test_data_path'], "./data/cikm/response_entity_predict_new-0.35.pkl")
