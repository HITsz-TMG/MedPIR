import os
import argparse

reorganize_state_path_dict = {
    "l=0.6": "",
    "l=0.7": "./cikm_save/Reorganize_model/epoch18-B[0.19721]-B1[0.26771]-B4[0.12670].pt",
    "!=": "",
}

config = {
    "model_name": "BERTGPTEntity",  # "BERT2BERTEntity" "BERTGPTEntity"

    "device": "4",

    "use_token_type_ids": False,
    "sentence_add_entity": False,

    "entity_predict": False,
    "use_entity_appendix": False,
    "pcl_encoder_predict_entity": False,

    "entity_attention": False,
    "entity_fused_way": "only_attention",  # entity_coarsely_select / entity_info_before_cls / only_attention
    "entity_query_model": "avg_pool_linear",
    "pretrained_select": "BertGPT",  # BertGPT / PCL

    'reorganize_method_id': 3,

    "epoch": 20,
    "batch_size": 8,
    "batch_expand_times": 4,
    "warm_up": 0.1,
    "lr": 2e-5,
    "weight_decay": 0,
    "encoder_lr_factor": 1,
    "entity_loss_factor": 1,

    "preprocessing": False,
    "parallel": None,

    "vocab_path": "./data/vocab.txt",
    "bert_config_path": "./pretrained_model/config.json",
    "bertgpt_state_dict": "pretrained/bertGPT_pretrained_model.pth",
    "pretrained_state_dict_path": "./pretrained/PCL-MedBERT/pytorch_model.bin",
    "pretrained_encoder_config_path": "./pretrained/PCL-MedBERT/config.json",
    "pretrained_decoder_config_path": "./pretrained/PCL-MedBERT/config_for_decoder.json",
    "gpt2_config_path": "./pretrained/gpt2/config.json",
    "dialog_gpt_path": "./GPT2-chitchat-master/dialogue_model",

    "train_data_path": "./data/cikm/train-4-25.pkl",
    "dev_data_path": "./data/cikm/dev-4-25.pkl",
    # "test_data_path": "./data/cikm/test-4-25-input.pkl",
    # "test_data_path": "./data/cikm/response_entity_predict_0.35_5-3.pkl",
    # "test_data_path": "./data/cikm/response_entity_predict_5-3.pkl",
    # "test_data_path": "./data/cikm/response_entity_predict_5-2.pkl",
    # "test_data_path": "./data/cikm/response_entity_predict_0.3_5-9.pkl",
    # "test_data_path": "./data/cikm/response_entity_predict_0.33_5-10.pkl",
    "test_data_path": "./data/cikm/response_entity_predict_new-0.35.pkl",
    # "test_data_path": "./data/cikm/test_new_with_retrieval.pkl",
    "golden_test_data_path": "./data/cikm/process_test.pkl",

    # "train_refs_path": "./reference_data/5-5-refs.pkl",
    # "dev_refs_path": "./reference_data/dev_dataset_references.pkl",
    # "test_refs_path": "./reference_data/test_dataset_references.pkl",
    # "test_refs_path": "./reference_data/test_dataset_references-5-11.pkl",

    # --- base --- #
    "train_refs_path": "./raw_response3/0.6.pkl",
    "dev_refs_path": "./raw_response3/dev_raw.pkl",
    "test_refs_path": "./raw_response3/new_test_raw.pkl",
    # "train_refs_path": "./raw_response2/6-8.pkl",
    # "dev_refs_path": "./raw_response2/dev_raw.pkl",
    # "test_refs_path": "./raw_response2/new_test_raw.pkl",
    # --- raw 1 --- #
    # "train_refs_path": "./comp_raw/train_raw4.pkl",
    # "dev_refs_path": "./comp_raw/dev_raw4.pkl",
    # "test_refs_path": "./comp_raw/test_raw4.pkl",

    "train_data_with_predict_next_entities": "./data/cikm/train_data_with_predict_next_entities-5-5.pkl",
    # "train_data_with_predict_next_entities": "./data/cikm/test_dataset_references-5-11.pkl",
    "top_k": 64,
    "top_p": 1,
    "min_len": 1,
    "max_len": 300,
    "beam_size": 4,
    "length_penalty": 1,
    "no_repeat_ngram_size": 5,
    "encoder_no_repeat_ngram_size": 8,
    "repetition_penalty": 1,

    "add_entity_noise": False,
    "entity_kl": False,
}

# config.update({
#     "dev_data_path": "./data/cikm/response_entity_predict_5-2.pkl",
#     "dev_refs_path": "./reference_data/test_dataset_references.pkl",
# })


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default=config['model_name'], required=False, type=str)
parser.add_argument("--debug", default=False, required=False, action="store_true")
parser.add_argument("--use_entity_appendix", default=config['use_entity_appendix'], required=False, type=bool)
parser.add_argument("--task", default=None, required=False, type=str)
parser.add_argument("--state_dict", default=config.get('state_dict'), required=False, type=str)
parser.add_argument("--device", default=config['device'], required=False, type=str)

parser.add_argument("--top_k", default=config['top_k'], required=False, type=int)
parser.add_argument("--beam_size", default=config['beam_size'], required=False, type=int)

parser.add_argument("--add_entity_noise", default=config['add_entity_noise'], required=False, type=bool)

# parser.add_argument("-s", "--start", type=int, required=True)
# parser.add_argument("-e", "--end", type=int, required=True)
# parser.add_argument("-d", "--distance", type=int, required=False, default=3500)

args = vars(parser.parse_args())
config.update(args)

print(args)

# if config['model_name'] == "DialogGPT":
#     config['vocab_path'] = "GPT2-chitchat-master/vocabulary/vocab_small.txt"

os.environ["CUDA_VISIBLE_DEVICES"] = config['device']

if config['use_entity_appendix']:
    # test_with_next_entities_path = "./data/cikm/response_entity_predict_5-2.pkl"
    # test_with_next_entities_path = "./data/cikm/response_entity_predict_5-3.pkl"
    # test_with_next_entities_path = "./data/cikm/response_entity_predict_0.35_5-3.pkl"
    # test_with_next_entities_path = "./data/cikm/response_entity_predict_0.3_5-9.pkl"
    # test_with_next_entities_path = "./data/cikm/response_entity_predict_0.33_5-10.pkl"
    # test_with_next_entities_path = "./data/cikm/response_entity_predict_new-0.35.pkl"
    test_with_next_entities_path = "./data/cikm/predict_and_retrieve.pkl"
    config['test_data_path'] = test_with_next_entities_path

print(config)
print("")

entity_type = ['Symptom', 'Medicine', 'Test', 'Attribute', 'Disease']
symptom = ['烧心', '口苦', '肠鸣', '心悸', '精神不振', '水肿', '胸痛', '粘便', '咽部痛', '腹泻', '稀便', '四肢麻木', '呼吸困难',
           '黄疸', '气促', '焦躁',
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
config['entity2eid'] = entity2eid

if config['use_entity_appendix']:
    print("USE ENTITY APPENDIX !!!")