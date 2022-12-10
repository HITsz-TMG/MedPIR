from src.spk_predict import SpkPredict
from src.dataset import SpkPredictDataset
from src.spk_trainer import SpkPredTrainer
from spk_config import config
import torch
import pickle
from tqdm import tqdm

if __name__ == '__main__':
    model = SpkPredict(config=config)
    if config.get('state_dict', None) is not None:
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))

    train_dataset = SpkPredictDataset(config['original_train_data_path'], config['vocab_path'],
                                      data_path=config['train_data_path'], data_type="train",
                                      preprocess=config['preprocessing'], config=config)
    dev_dataset = SpkPredictDataset(config['original_train_data_path'], config['vocab_path'],
                                    data_path=config['dev_data_path'], data_type="dev",
                                    preprocess=config['preprocessing'])

    if config.get("task", None) != "test":
        trainer = SpkPredTrainer(train_dataset, model, dev_dataset=dev_dataset, config=config)
        trainer.train()
    else:
        test_data = pickle.load(open(config['original_test_data_path'], 'rb'))
        add_spk_data = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        # pickle.dump([], open("./data/4-18-data/input/test2_add_spk.pkl", 'wb'))

        pickle.dump([], open("./data/4-18-data/input/test2_add_spk.pkl", 'wb'))


        with torch.no_grad():
            for dialog_p in tqdm(test_data, desc="add speaker in test data"):
                dialog = dialog_p['history']
                cur_add_spk_data = [{"id": "Patients", "Sentence": str(dialog[0])}]
                history = train_dataset.build_sentence(dialog[0])
                for turn in dialog[1:]:
                    plain_sentence = str(turn)
                    sentence = train_dataset.build_sentence(turn)[-250:]
                    input_ids = [train_dataset.cls_idx] + sentence + [train_dataset.sep_idx] + \
                                history[-(512 - len(sentence) - 3):] + [train_dataset.sep_idx]
                    token_type_ids = [0] * (len(sentence) + 2)
                    token_type_ids = token_type_ids + [1] * (len(input_ids) - len(token_type_ids))
                    model_input = {
                        "input_ids": torch.tensor([input_ids]).to(device),
                        "token_type_ids": torch.tensor([token_type_ids]).to(device),
                    }
                    output = model(**model_input)
                    logits = output[0]
                    _, preds = logits.max(dim=-1)
                    spk = train_dataset.idx2spk[preds.item()]
                    cur_add_spk_data.append({"id": spk, "Sentence": plain_sentence})
                add_spk_data.append(list(cur_add_spk_data))

        pickle.dump(add_spk_data, open("./data/4-18-data/input/test2_add_spk.pkl", 'wb'))
