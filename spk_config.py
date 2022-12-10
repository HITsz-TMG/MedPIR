import os

config = {
    "model_name": "SpkPredict2",
    "device": "1",

    "epoch": 10,
    "batch_size": 32,
    "batch_expand_times": 1,
    "warm_up": 2000,
    "lr": 2e-5,
    "weight_decay": 0.0002,

    "preprocessing": False,

    "parallel": None,

    "vocab_path": "./data/vocab.txt",
    "bert_config_path": "./pretrained_model/config.json",
    "bertgpt_state_dict": "pretrained/bertGPT_pretrained_model.pth",
    "pretrained_state_dict_path": "./pretrained/PCL-MedBERT/pytorch_model.bin",
    "pretrained_encoder_config_path": "./pretrained/PCL-MedBERT/config.json",
    "pretrained_decoder_config_path": "./pretrained/PCL-MedBERT/config_for_decoder.json",
    "gpt2_config_path": "./pretrained/gpt2/config.json",

    "state_dict": "./save/SpkPred_save/epoch3-acc0.97441.SpkPredict2",
    "task": "test"
}
os.environ["CUDA_VISIBLE_DEVICES"] = config['device']

if config.get("task", None) == "debug":
    config.update({
        "train_data_path": "./data/spk_predict_train_debug.pkl",
        "dev_data_path": "./data/spk_predict_train_debug.pkl",
        "original_train_data_path": "./data/original/new_train.pk",
    })
else:
    config.update({
        "train_data_path": "./data/spk_predict_train_2.pkl",
        "dev_data_path": "./data/spk_predict_dev_2.pkl",
        "original_train_data_path": "./data/original/new_train.pk",
        # "original_test_data_path": "./data/change/test.pk",
        "original_test_data_path": "./data/4-18-data/input/B_test.pk"

    })

print(config)
