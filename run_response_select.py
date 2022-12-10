from response_select_config import config
from src.response_select_dataset import ResponseSelectDataset
from src.response_select_trainer import ResponseSelectTrainer
from src.response_select import ResponseSelector as Model
import torch
import importlib
import random
import pickle
from tqdm import tqdm

if __name__ == '__main__':

    if config['task'] == 'train':

        if config["preprocessing"] is False:
            t_data = pickle.load(open(config['done_data_path'], "rb"))
            train_data = t_data['train_data']
            dev_data = t_data["dev_data"]
            all_doctor_sentence = t_data['all_doctor_sentence']

            train_dataset = ResponseSelectDataset(config['vocab_path'],
                                                  data_type="train",
                                                  config=config,
                                                  all_doctor_sentence=all_doctor_sentence,
                                                  used_data=train_data,
                                                  neg_sample_num=config['neg_sample_num'])
            dev_dataset = ResponseSelectDataset(config['vocab_path'],
                                                data_type="dev",
                                                config=config,
                                                all_doctor_sentence=all_doctor_sentence,
                                                used_data=dev_data,
                                                neg_sample_num=config['neg_sample_num'])
        else:
            train_dataset = ResponseSelectDataset(config['vocab_path'],
                                                  data_path=config['train_data_path'],
                                                  data_type="train",
                                                  config=config,
                                                  neg_sample_num=config['neg_sample_num'])
            random.shuffle(train_dataset.data)
            split_len = 5000
            train_data = train_dataset.data[split_len:]
            dev_data = train_dataset.data[:split_len]
            train_dataset.data = train_data
            pickle.dump({
                "all_doctor_sentence": train_dataset.all_doctor_sentence,
                "train_data": train_data,
                "dev_data": dev_data
            }, open(config['done_data_path'], 'wb'))
            dev_dataset = ResponseSelectDataset(config['vocab_path'],
                                                data_type="dev",
                                                config=config,
                                                all_doctor_sentence=train_dataset.all_doctor_sentence,
                                                used_data=dev_data,
                                                neg_sample_num=config['neg_sample_num'])

        print(len(train_dataset.data), len(dev_dataset.data))
        model = Model(config=config)
        if config.get('state_dict', None) is not None:
            model.load_state_dict(torch.load(config['state_dict']))

        trainer = ResponseSelectTrainer(train_dataset, model, dev_dataset=dev_dataset, config=config)
        trainer.train()

    elif config['task'] == 'predict':
        raise NotImplementedError
