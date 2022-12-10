import os

config = {
    "device": "2",
    "model_name": "ResponseSelector",
    "epoch": 10,
    "batch_size": 4,
    "batch_expand_times": 8,
    "warm_up": 2000,
    "lr": 1e-5,
    "weight_decay": 0,
    "neg_sample_num": 5,
    "parallel": None,

    "metric": "acc",

    "vocab_path": "./data/vocab.txt",
    "bert_config_path": "./pretrained_model/config.json",
    "bertgpt_state_dict": "pretrained/bertGPT_pretrained_model.pth",
    "pretrained_state_dict_path": "./pretrained/PCL-MedBERT/pytorch_model.bin",
    "pretrained_encoder_config_path": "./pretrained/PCL-MedBERT/config.json",
    "pretrained_decoder_config_path": "./pretrained/PCL-MedBERT/config_for_decoder.json",
    "gpt2_config_path": "./pretrained/gpt2/config.json",

    # "entity_path": "./data/entity",

    "preprocessing": False,
    # "state_dict": "./save/ResponseSelect_save/epoch6-F10.81325.ResponseSelector",
    "state_dict": "./save/ResponseSelect_save/epoch5-F10.80969.ResponseSelector",

    # "task": "entity_predict"
    "task": "train"

}
os.environ["CUDA_VISIBLE_DEVICES"] = config['device']

config.update({
    "train_data_path": "./data/input_1500/plain_train_dev.pkl",
    "done_data_path": "./data/change/response_select.pkl",
})

print(config)
