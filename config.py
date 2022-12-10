import os

config = {
    "model_name": "BERTGPT",
    "device": "2",

    "epoch": 20,
    "batch_size": 8,
    "batch_expand_times": 4,
    "warm_up": 2000,
    "lr": 1.5e-5,
    "weight_decay": 0,
    "encoder_lr_factor": 0.8,

    "use_token_type_ids": False,
    "preprocessing": False,

    "parallel": None,
    "vocab_path": "./data/vocab.txt",
    "bert_config_path": "./pretrained_model/config.json",
    "bertgpt_state_dict": "pretrained/bertGPT_pretrained_model.pth",
    "pretrained_state_dict_path": "./pretrained/PCL-MedBERT/pytorch_model.bin",
    "pretrained_encoder_config_path": "./pretrained/PCL-MedBERT/config.json",
    "pretrained_decoder_config_path": "./pretrained/PCL-MedBERT/config_for_decoder.json",
    "gpt2_config_path": "./pretrained/gpt2/config.json",

    # "entity_attention": True,
    # "state_dict": "./save/bak/epoch1-step1999-0316-1555.BERTGPT",
    # ./save/BERTGPT_save/epoch10-acc0.36748.pt
    # ./save/epoch1-step9999-0316-1641.BERTGPT
    # epoch2-acc0.61244.BERTGPT
    # "state_dict": "./save/epoch4-B[0.07579]-B1[0.11295]-B4[0.03863].pt",
    "generate": "sample",
    "top_k": 1,
    "top_p": 0,
    "min_len": 3,
    "beam_size": 3,
    # "task": "debug"
}
os.environ["CUDA_VISIBLE_DEVICES"] = config['device']

if config.get("task", None) == "debug":
    config.update({
        "train_data_path": "./data/entity_attention.pkl",
        "dev_data_path": "./data/entity_attention.pkl",
        "original_train_data_path": "./data/original/train.pk",
        "original_test_data_path": "./data/original/test.pk"
    })
else:
    config.update({
        "train_data_path": "./data/input/final_new_train.pkl",
        "test_data_path": "./data/new_plain_test.pkl",
        "dev_data_path": "./data/input/final_new_dev.pkl",
        "original_train_data_path": "./data/original/new_train.pk",
        "original_test_data_path": "./data/original/new_test.pk",
        "original_dev_data_path": "./data/original/new_dev.pk"
    })

