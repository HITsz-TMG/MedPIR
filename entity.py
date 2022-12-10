from entity_config import config
from src.dataset import EntityPredictDataset
from src.entity_trainer import EntityPredTrainer
import torch
import importlib
import random
import pickle
from tqdm import tqdm

if __name__ == '__main__':
    Model = getattr(importlib.import_module("src.entity_predict"), config['model_name'])
    config['model_class'] = Model

    if config['task'] == 'train':

        train_dataset = EntityPredictDataset(config['original_train_data_path'], config['vocab_path'],
                                             data_path=config['train_data_path'], data_type="train",
                                             preprocess=config['preprocessing'],
                                             # preprocess=True,
                                             config=config)

        print(len(train_dataset.data))
        if config['preprocessing']:
            random.shuffle(train_dataset.data)
            split_len = int(len(train_dataset.data) * 0.1)
            train_data = train_dataset.data[split_len:]
            dev_data = train_dataset.data[:split_len]
            pickle.dump(train_data, open(config['train_data_path'], 'wb'))
            pickle.dump(dev_data, open(config['dev_data_path'], 'wb'))
            train_dataset.data = train_data

        dev_dataset = EntityPredictDataset(config['original_train_data_path'], config['vocab_path'],
                                           data_path=config['dev_data_path'], data_type="dev",
                                           preprocess=False)

        print(len(train_dataset.data), len(dev_dataset.data))
        model = Model(config=config, entity_type_num=train_dataset.entity_type_num)
        if config.get('state_dict', None) is not None:
            model.load_state_dict(torch.load(config['state_dict']))

        trainer = EntityPredTrainer(train_dataset, model, dev_dataset=dev_dataset, config=config)
        trainer.train()


    elif config['task'] == 'predict':
        dev_dataset = EntityPredictDataset(config['original_train_data_path'], config['vocab_path'],
                                           data_path=config['dev_data_path'], data_type="test",
                                           preprocess=False)
        model = Model(config=config, entity_type_num=dev_dataset.entity_type_num)
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        model.eval()
        # eid2entity = dev_dataset.symptom + dev_dataset.medicine + dev_dataset.test + \
        #              dev_dataset.attribute + dev_dataset.disease

        test_data = pickle.load(open(config['test_data_path'], 'rb'))
        prob_order = ['Symptom', 'Medicine', 'Test', 'Attribute', 'Disease']
        eid2ent = [dev_dataset.symptom,
                   dev_dataset.medicine,
                   dev_dataset.test,
                   dev_dataset.attribute,
                   dev_dataset.disease]
        pickle.dump([], open("./data/change/test_add_spk_and_entity.pkl", 'wb'))

        for idx in tqdm(range(len(test_data))):
            for idx_t in range(len(test_data[idx]['text'][0])):
                sentence = dev_dataset._build_sentence(test_data[idx]['text'][0][idx_t]['Sentence'])[-510:]

                for cate in prob_order:
                    if cate not in test_data[idx]['text'][0][idx_t].keys():
                        test_data[idx]['text'][0][idx_t][cate] = []

                input_ids = torch.tensor([sentence]).to(device)
                with torch.no_grad():
                    topic_probs, five_topic_probs, loss = model(input_ids=input_ids)
                five_topic_probs = [p[0].tolist() for p in five_topic_probs]
                for cate_idx, cate in enumerate(prob_order):
                    cur_cate_prob = five_topic_probs[cate_idx]
                    for ent_id, ent_prob in enumerate(cur_cate_prob):
                        if ent_prob > 0.5:
                            test_data[idx]['text'][0][idx_t][cate].append(
                                eid2ent[cate_idx][ent_id]
                            )
                            # print("-"*30)
                            # print(eid2ent[cate_idx][ent_id])
                            # print(test_data[idx]['text'][0][idx_t]['Sentence'])
                            # print("-"*30)

        pickle.dump(test_data, open("./data/change/test_add_spk_and_entity.pkl", 'wb'))
