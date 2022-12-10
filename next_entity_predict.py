if input("160 entity?  [y/n]:") == 'y':
    from entity_config_160 import config
else:
    from entity_config import config
from src.dataset import MedDGDataset
from src.entity_trainer import EntityPredTrainer
from src.model import NextEntityPredict as Model
import torch
import importlib
import random
import pickle
from tqdm import tqdm

if __name__ == '__main__':
    config['model_class'] = Model
    model = Model(config=config)
    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))

    if config['task'] == 'train':
        k_fold = 6
        tmp_data_1 = pickle.load(open(config['train_data_path'], 'rb'))
        tmp_data_2 = pickle.load(open(config['dev_data_path'], 'rb'))
        all_data = tmp_data_2 + tmp_data_1
        k_fold_data = []
        piece_num = len(all_data) // k_fold
        for k in range(k_fold):
            start = k * piece_num
            end = (k + 1) * piece_num
            if k == k_fold - 1:
                end = len(all_data)
            print(start, end)
            k_fold_data.append(all_data[start:end])

        for k in range(k_fold):

            model = Model(config=config)

            tmp_train_data = []
            for kk in range(k_fold):
                if kk != k:
                    tmp_train_data = tmp_train_data + list(k_fold_data[kk])
            dev_data = k_fold_data[k]

            train_dataset = MedDGDataset(config['original_train_data_path'], config['vocab_path'],
                                         data_path=config['train_data_path'], data_type="train",
                                         preprocess=config['preprocessing'], config=config)
            train_data = []
            for i in tmp_train_data:
                t_l = 0
                for t in train_dataset.entity_type:
                    t_l += len(i['text'][1][t])
                if t_l != 0 or random.random() < 0.6:
                    train_data.append(i)

            train_dataset.data = train_data

            dev_dataset = MedDGDataset(config['original_train_data_path'], config['vocab_path'],
                                       data_path=config['dev_data_path'], data_type="dev",
                                       preprocess=config['preprocessing'], config=config)
            dev_dataset.data = dev_data

            print("Next entity predict FOLD-{}".format(k))
            print("train len: {}, dev len: {}".format(len(train_dataset), len(dev_dataset)))

            trainer = EntityPredTrainer(train_dataset, model, dev_dataset=dev_dataset, config=config)
            trainer.train()

    elif config['task'] == 'predict':

        test_dataset = MedDGDataset(config['original_train_data_path'], config['vocab_path'],
                                    data_path=config['test_data_path'], data_type='test',
                                    preprocess=config['preprocessing'], config=config)

        test_data_loader = test_dataset.get_dataloader(batch_size=1, shuffle=False)

        if torch.cuda.is_available():
            model = model.to('cuda')
        model.eval()

        prob_order = ['Symptom', 'Medicine', 'Test', 'Attribute', 'Disease']
        eid2ent = [
            config['symptom'],
            config['medicine'],
            config['test'],
            config['attribute'],
            config['disease']
        ]
        pickle.dump([], open("./data/4-18-data/test2_add_spk_and_entity-0.41-160.pkl", "wb"))
        test_data = test_dataset.data

        pred_entity_num = 0

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

            if test_data[idx]['text'][1] is None:
                test_data[idx]['text'][1] = dict()
            for et in config['entity_type']:
                test_data[idx]['text'][1][et] = []

            cur_all_ent = []
            five_topic_probs = [p[0].tolist() for p in five_topic_probs]
            for cate_idx, cate in enumerate(prob_order):
                cur_cate_prob = five_topic_probs[cate_idx]
                for ent_id, ent_prob in enumerate(cur_cate_prob):
                    if ent_prob > 0.41:
                        test_data[idx]['text'][1][cate].append(
                            eid2ent[cate_idx][ent_id]
                        )
                        cur_all_ent.append(eid2ent[cate_idx][ent_id])
                        pred_entity_num += 1

            # print("-" * 30)
            # for u in test_data[idx]['text'][0]:
            #     print(u['id'] + ":" + u['Sentence'])
            # print("、".join(cur_all_ent))
            # print("-" * 30)

        pickle.dump(test_data, open("./data/4-18-data/test2_add_spk_and_entity-0.41-160.pkl", 'wb'))
        print("pred entity num: {}".format(pred_entity_num))

# 0.41
# 0.4 18991
# 0.39 19915
# 0.35 23665
# 0.33 25767
# 0.32 26923
# 0.3 29355
