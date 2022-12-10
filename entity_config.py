import os

config = {
    "model_name": "NextEntityPredict",
    "device": "0",

    "epoch": 20,
    "batch_size": 16,
    "batch_expand_times": 2,
    "warm_up": 2000,
    "lr": 2e-5,
    "weight_decay": 0,

    "preprocessing": False,

    "parallel": None,

    "vocab_path": "./data/vocab.txt",
    "bert_config_path": "./pretrained_model/config.json",
    "bertgpt_state_dict": "pretrained/bertGPT_pretrained_model.pth",
    "pretrained_state_dict_path": "./pretrained/PCL-MedBERT/pytorch_model.bin",
    "pretrained_encoder_config_path": "./pretrained/PCL-MedBERT/config.json",
    "pretrained_decoder_config_path": "./pretrained/PCL-MedBERT/config_for_decoder.json",
    "gpt2_config_path": "./pretrained/gpt2/config.json",

    "state_dict": "./save/EntityPred_save/epoch6-F10.22198.NextEntityPredict",

    # -------- Next Entity Predict Config
    "use_token_type_ids": True,
    "pcl_encoder_predict_entity": True,
    "clip_long_sentence": True,
    "entity_predict": True,

    # "task": "entity_predict"
    # "task": "train"
    "task": "predict"
}
os.environ["CUDA_VISIBLE_DEVICES"] = config['device']

if config['model_name'] == "EntityPredict":
    if config.get("task", None) == "debug":
        config.update({
            "train_data_path": "./data/entity_predict_train.pkl",
            "dev_data_path": "./data/entity_predict_dev.pkl",
            "original_train_data_path": "./data/original/new_train.pk",
        })
    else:
        config.update({
            "train_data_path": "./data/change/entity_predict_train.pkl",
            "dev_data_path": "./data/change/entity_predict_dev.pkl",
            "test_data_path": "./data/change/test_add_spk.pkl",
            "original_train_data_path": "./data/change/plain_train_dev.pkl",
            "original_dev_data_path": "./data/change/plain_train_dev.pkl",
            "original_test_data_path": "./data/change/plain_train_dev.pkl",

        })
elif config['model_name'] == "NextEntityPredict":
    config.update({
        "train_data_path": "./data/input_1500/plain_train.pkl",
        "dev_data_path": "./data/input_1500/plain_dev.pkl",
        # "train_data_path": "./data/change/plain_train.pkl",
        # "dev_data_path": "./data/change/plain_dev.pkl",
        "test_data_path": "./data/4-18-data/input/test2_add_spk_with_text.pkl",
        "original_train_data_path": "./data/original/new_train.pk",
        "original_dev_data_path": "./data/original/new_dev.pk"
    })
print(config)

entity_type = ['Symptom', 'Medicine', 'Test', 'Attribute', 'Disease']
symptom = ['乏力', '体重下降', '便血', '反流', '发热', '口苦', '吞咽困难', '呕吐', '呕血',
           '呼吸困难', '咳嗽', '咽部灼烧感', '咽部痛', '喷嚏', '嗜睡', '四肢麻木', '头晕',
           '头痛', '寒战', '尿急', '尿频', '心悸', '恶心', '打嗝', '排气', '月经紊乱', '有痰',
           '气促', '水肿', '消化不良', '淋巴结肿大', '烧心', '焦躁', '痉挛', '痔疮', '稀便',
           '粘便', '精神不振', '细菌感染', '肌肉酸痛', '肛周疼痛', '肠梗阻', '肠鸣', '胃痛',
           '胃肠不适', '胃肠功能紊乱', '背痛', '胸痛', '脱水', '腹泻', '腹痛', '腹胀', '菌群失调',
           '螺旋杆菌感染', '贫血', '过敏', '里急后重', '食欲不振', '饥饿感', '黄疸', '黑便', '鼻塞']
medicine = ['三九胃泰', '乳果糖', '乳酸菌素', '人参健脾丸', '健胃消食片', '健脾丸', '克拉霉素', '兰索拉唑',
            '吗丁啉', '嗜酸乳杆菌', '四磨汤口服液', '培菲康', '复方消化酶', '多潘立酮', '多酶片', '奥美',
            '山莨菪碱', '左氧氟沙星', '布洛芬', '康复新液', '开塞露', '得舒特', '思密达', '思连康', '抗生素',
            '整肠生', '斯达舒', '曲美布汀', '泌特', '泮托拉唑', '消炎利胆片', '瑞巴派特', '甲硝唑', '益生菌',
            '硫糖铝', '磷酸铝', '耐信', '肠溶胶囊', '肠胃康', '胃复安', '胃苏颗粒', '胶体果胶铋', '莫沙比利',
            '蒙脱石散', '藿香正气丸', '补脾益肠丸', '诺氟沙星胶囊', '谷氨酰胺肠溶胶囊', '达喜', '金双歧',
            '铝碳酸镁', '阿莫西林', '雷贝拉唑', '颠茄片', '香砂养胃丸', '马来酸曲美布丁']
test = ['b超', 'ct', '便常规', '呼气实验', '小肠镜', '尿常规', '尿检', '糖尿病', '结肠镜',
        '肛门镜', '肝胆胰脾超声', '肠镜', '胃蛋白酶', '胃镜', '胶囊内镜', '腹腔镜', '腹部彩超', '血常规',
        '转氨酶', '钡餐']
attribute = ['位置', '性质', '时长', '诱因']
disease = ['便秘', '感冒', '肝硬化', '肠易激综合征', '肠炎', '肺炎',
           '胃溃疡', '胃炎', '胆囊炎', '胰腺炎', '阑尾炎', '食管炎']

eid2entity = symptom + medicine + test + attribute + disease

entity_type_num = {
    'Symptom': len(symptom),
    'Medicine': len(medicine),
    'Test': len(test),
    'Attribute': len(attribute),
    'Disease': len(disease)
}
config['symptom'] = symptom
config['medicine'] = medicine
config['test'] = test
config['attribute'] = attribute
config['disease'] = disease

config['entity_type'] = entity_type
config['entity'] = eid2entity
config['entity_type_num'] = entity_type_num
