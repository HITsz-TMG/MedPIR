import pyCMeKG
import os
import json

replace_dict = {
    "马来酸曲美布汀": "曲美布汀",
    "马来酸曲美布丁": "曲美布汀",
    "马来酸曲美布丁片": "曲美布汀",
    "马来酸曲美布汀片": "曲美布汀",
    "胃苏": "胃苏颗粒",
    "诺氟沙星": "诺氟沙星胶囊",
    "莫沙必利": "莫沙比利",
    "吗叮咛": "多潘立酮",
    "吗丁啉": "多潘立酮",
    "果胶铋": "胶体果胶铋",
    "雷呗": "雷贝拉唑",
    "四磨汤": "四磨汤口服液",
    "思连康": "双歧杆菌四联活菌片",
    "金双歧": "双歧杆菌嗜酸性乳杆菌粪链球菌",
    "培菲康": "双歧杆菌嗜酸性乳杆菌粪链球菌",
    "三九胃泰": "三九胃泰胶囊",
    "奥美": "奥美拉唑",
    "开塞露": "甘油氯化钠",
    "得舒特": "匹维溴铵",
    "思密达": "蒙脱石",
    "蒙脱石散": "蒙脱石",
    "整肠生": "地衣芽孢杆菌",
    "泌特": "复方阿嗪米特",
    "耐信": "埃索美拉唑",
    "谷氨酰胺肠溶胶囊": "复方谷氨酰胺肠溶胶囊",
    "肠溶胶囊": "复方谷氨酰胺肠溶胶囊",
    "肠胃康": "枫蓼肠胃康颗粒",
    "胃复安": "甲氧氯普胺",
    "达喜": "铝碳酸镁",
}


def get_reverse_dict():
    reverse_replace = dict()
    value = []
    for i in replace_dict:
        value.append(replace_dict[i])
    value = set(value)
    for i in value:
        reverse_replace[i] = []
        for j in replace_dict:
            if replace_dict[j] == i:
                reverse_replace[i].append(j)
    return reverse_replace


reverse_replace_dict = get_reverse_dict()

cmekg_dict = pyCMeKG.cmekg().model.db_dic

all_entity_triples_root = "./cmekg/all_entity_triples"
crawled_entity_path = list(os.walk(all_entity_triples_root))[0][2]
crawled_entity = [_[:-5] for _ in crawled_entity_path]
crawled_entity_path = [os.path.join(all_entity_triples_root, i) for i in crawled_entity_path]
entity_to_path = {k: v for k, v in zip(crawled_entity, crawled_entity_path)}
# entity_to_triple = dict()
entity_to_tails = dict()
for e, p in entity_to_path.items():
    # entity_to_triple[e] = json.load(open(p, 'r', encoding='utf-8'))['triples']
    triples = json.load(open(p, 'r', encoding='utf-8'))['triples']
    tmp = []
    for triple in triples:
        if triple[-1]:
            tmp.append(triple[-1].strip())
    entity_to_tails[e] = set(tmp)
for e in cmekg_dict:
    if e in entity_to_tails:
        continue
    tmp = []
    for key in cmekg_dict[e].keys():
        for tails in cmekg_dict[e][key]:
            tmp.append(tails.strip())
    entity_to_tails[e] = set(tmp)


def has_relation(ei, ej):
    ei_tails = entity_to_tails[ei]
    ej_tails = entity_to_tails[ej]
    if ej in ei_tails:
        return True
    elif ei in ej_tails:
        return True
    else:
        return False
