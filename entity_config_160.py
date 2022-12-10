import os
import pickle
import random

# all_data = pickle.load(open("./data/4-18-data/input/plain_train_dev.pkl", "rb"))
#
# print(len(all_data))
#
# random.shuffle(all_data)
# dev = all_data[:10000]
# train = all_data[10000:]
#
# pickle.dump(dev, open("./data/4-18-data/next_entity_predict/dev.pkl", 'wb'))
# pickle.dump(train, open("./data/4-18-data/next_entity_predict/train.pkl", 'wb'))


config = {
    "model_name": "NextEntityPredict",
    "device": "2",

    "epoch": 5,
    "batch_size": 34,
    "batch_expand_times": 1,
    "warm_up": 3000,
    "lr": 2.5e-5,
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

    # "entity_path": "./data/entity",
    # "state_dict": "./save/EntityPred_save/epoch6-F10.22198.NextEntityPredict",
    # "state_dict": "./save/EntityPred_save/epoch4-F10.16367.NextEntityPredict",

    # -------- Next Entity Predict Config
    "use_token_type_ids": True,
    "pcl_encoder_predict_entity": True,
    "clip_long_sentence": True,
    "entity_predict": True,
    # -------- Next Entity Predict Config

    # "task": "entity_predict"
    "task": "train"
    # "task": "predict"
}

os.environ["CUDA_VISIBLE_DEVICES"] = config['device']

if config['model_name'] == "EntityPredict":
    exit()
elif config['model_name'] == "NextEntityPredict":
    config.update({
        "train_data_path": "./data/4-18-data/next_entity_predict/train.pkl",
        "dev_data_path": "./data/4-18-data/next_entity_predict/dev.pkl",

        "test_data_path": "./data/4-18-data/input/test2_add_spk_with_text.pkl",
        "original_train_data_path": "./data/original/new_train.pk",
        "original_dev_data_path": "./data/original/new_dev.pk"
    })

entity_type = ['Symptom', 'Medicine', 'Test', 'Attribute', 'Disease']

symptom = ['烧心', '口苦', '肠鸣', '心悸', '精神不振', '水肿', '胸痛', '粘便', '咽部痛', '腹泻', '稀便', '四肢麻木', '呼吸困难', '黄疸', '气促', '焦躁',
           '痔疮', '咽部灼烧感', '寒战', '便血', '细菌感染', '肛周疼痛', '尿急', '胃肠不适', '体重下降', '呕吐', '吞咽困难', '发热', '黑便', '消化不良',
           '饥饿感', '呕血', '脱水', '腹痛', '菌群失调', '胃肠功能紊乱', '螺旋杆菌感染', '鼻塞', '里急后重', '过敏', '食欲不振', '乏力', '尿频', '恶心', '反流',
           '肌肉酸痛', '嗜睡', '胃痛', '咳嗽', '喷嚏', '肠梗阻', '有痰', '腹胀', '痉挛', '排气', '头晕', '头痛', '月经紊乱', '贫血', '背痛', '打嗝',
           '淋巴结肿大']
medicine = ['雷呗', '思连康', '肠溶胶囊', '藿香正气丸', '克拉霉素', '整肠生', '培菲康', '嗜酸乳杆菌', '磷酸铝', '金双歧', '莫沙必利', '复方消化酶', '肠胃康',
            '马来酸曲美布汀片', '瑞巴派特', '泌特', '铝碳酸镁', '消炎利胆片', '硫糖铝', '马来酸曲美布汀', '多潘立酮', '诺氟沙星胶囊', '谷氨酰胺肠溶胶囊', '胃复安',
            '胶体果胶铋', '泮托拉唑', '兰索拉唑', '思密达', '马来酸曲美布丁', '乳果糖', '曲美布汀', '四磨汤', '诺氟沙星', '健胃消食片', '香砂养胃丸', '健脾丸', '奥美',
            '达喜', '康复新液', '抗生素', '甲硝唑', '开塞露', '得舒特', '布洛芬', '阿莫西林', '果胶铋', '补脾益肠丸', '斯达舒', '胃苏', '多酶片', '颠茄片',
            '吗丁啉', '左氧氟沙星', '益生菌', '山莨菪碱', '蒙脱石散', '吗叮咛', '耐信', '人参健脾丸', '乳酸菌素', '莫沙比利', '三九胃泰']
test = ['b超', '便常规', '钡餐', '腹部彩超', '尿检', '胃蛋白酶', '呼气实验', '肝胆胰脾超声', '肠镜', '胶囊内镜', 'ct', '小肠镜', '转氨酶', '腹腔镜', '尿常规',
        '血常规', '糖尿病', '肛门镜', '结肠镜', '胃镜']
attribute = ['位置', '诱因', '时长', '性质']
disease = ['胃炎', '阑尾炎', '胆囊炎', '胰腺炎', '肺炎', '胃溃疡', '肝硬化', '肠易激综合征', '感冒', '食管炎', '肠炎', '便秘']

entity_type_num = {
    'Symptom': len(symptom),
    'Medicine': len(medicine),
    'Test': len(test),
    'Attribute': len(attribute),
    'Disease': len(disease)
}

eid2entity = [
    '烧心', '口苦', '肠鸣', '心悸', '精神不振', '水肿', '胸痛', '粘便', '咽部痛', '腹泻', '稀便', '四肢麻木', '呼吸困难', '黄疸',
    '气促', '焦躁', '痔疮', '咽部灼烧感', '寒战', '便血', '细菌感染', '肛周疼痛', '尿急', '胃肠不适', '体重下降', '呕吐', '吞咽困难',
    '发热', '黑便', '消化不良', '饥饿感', '呕血', '脱水', '腹痛', '菌群失调', '胃肠功能紊乱', '螺旋杆菌感染', '鼻塞', '里急后重', '过敏',
    '食欲不振', '乏力', '尿频', '恶心', '反流', '肌肉酸痛', '嗜睡', '胃痛', '咳嗽', '喷嚏', '肠梗阻', '有痰', '腹胀', '痉挛', '排气',
    '头晕', '头痛', '月经紊乱', '贫血', '背痛', '打嗝', '淋巴结肿大', '雷呗', '思连康', '肠溶胶囊', '藿香正气丸', '克拉霉素', '整肠生',
    '培菲康', '嗜酸乳杆菌', '磷酸铝', '金双歧', '莫沙必利', '复方消化酶', '肠胃康', '马来酸曲美布汀片', '瑞巴派特', '泌特', '铝碳酸镁',
    '消炎利胆片', '硫糖铝', '马来酸曲美布汀', '多潘立酮', '诺氟沙星胶囊', '谷氨酰胺肠溶胶囊', '胃复安', '胶体果胶铋', '泮托拉唑', '兰索拉唑',
    '思密达', '马来酸曲美布丁', '乳果糖', '曲美布汀', '四磨汤', '诺氟沙星', '健胃消食片', '香砂养胃丸', '健脾丸', '奥美', '达喜',
    '康复新液', '抗生素', '甲硝唑', '开塞露', '得舒特', '布洛芬', '阿莫西林', '果胶铋', '补脾益肠丸', '斯达舒', '胃苏', '多酶片',
    '颠茄片', '吗丁啉', '左氧氟沙星', '益生菌', '山莨菪碱', '蒙脱石散', '吗叮咛', '耐信', '人参健脾丸', '乳酸菌素', '莫沙比利',
    '三九胃泰', 'b超', '便常规', '钡餐', '腹部彩超', '尿检', '胃蛋白酶', '呼气实验', '肝胆胰脾超声', '肠镜', '胶囊内镜', 'ct', '小肠镜',
    '转氨酶', '腹腔镜', '尿常规', '血常规', '糖尿病', '肛门镜', '结肠镜', '胃镜', '位置', '诱因', '时长', '性质', '胃炎', '阑尾炎',
    '胆囊炎', '胰腺炎', '肺炎', '胃溃疡', '肝硬化', '肠易激综合征', '感冒', '食管炎', '肠炎', '便秘'
]

entity2eid = {e: idx for idx, e in enumerate(eid2entity)}

config['symptom'] = symptom
config['medicine'] = medicine
config['test'] = test
config['attribute'] = attribute
config['disease'] = disease

config['entity_type'] = entity_type
config['entity'] = eid2entity
config['entity_type_num'] = entity_type_num
