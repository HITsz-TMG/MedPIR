import pandas
import json
from collections import OrderedDict, Counter
import numpy as np
# fi1 = open("meddg_dev_vocab.csv", 'r', encoding='utf-8')
# fi2 = open("meddg_dev_vocab.csv", 'r', encoding='utf-8')
#
# f1 = fi1.readlines()[1:]
# f2 = fi2.readlines()[1:]
#
#
# def know_num(ff):
#     cnt = 0
#     for i in ff:
#         tt = i.strip().split(",")
#         if tt[1] == '1':
#             cnt += 1
#     return cnt
#
#
# def check_diff(ff1, ff2):
#     cnt = 0
#     for i, j in zip(ff1, ff2):
#         if i != j:
#             cnt += 1
#             print(i, j)
#     print(cnt)
#
#
# print(know_num(f1))
# print(know_num(f2))
# check_diff(f1, f2)

d = pandas.read_csv("meddg_vocab.csv")
d = d[d['Is_know'] > 0]
d = dict(zip(list(d['Word']), list(d["Is_know"])))
print()


def vbot_entity_score(vbot_vocab_file, result_file):
    disease2x = pandas.read_csv(vbot_vocab_file)
    disease2x = disease2x[disease2x["Is_know"] > 0]
    disease2x = dict(zip(list(disease2x["Word"]), list(disease2x["Is_know"])))
    with open(result_file) as f:
        result_fi = json.load(f)

    gths = []  # (session, episode) str
    hyps = []
    for session in sessions:
        tmp1, tmp2 = [], []
        for episode in session['session']:
            tmp1.append(episode['gth'])
            tmp2.append(episode['hyp'])
        gths.append(tmp1)
        hyps.append(tmp2)

    entity_gths = []  # (session, episode) str
    entity_hyps = []
    for y in gths:
        for x in y:
            tmp = []
            for i in x.split(" "):
                if i in disease2x:
                    tmp.append(i)
            entity_gths.append(" ".join(tmp))
    for y in hyps:
        for x in y:
            tmp = []
            for i in x.split(" "):
                if i in disease2x:
                    tmp.append(i)
            entity_hyps.append(" ".join(tmp))

    # def flat(lists):
    #     tmp = []
    #     for items in lists:
    #         tmp += items
    #     return tmp

    # gths = flat(gths)  # len: session*episode : "sentence"
    # hyps = flat(hyps)
    # entity_gths = flat(entity_gths)  # len: session*episode : "e1 e2 ... en"
    # entity_hyps = flat(entity_hyps)

    overlapped_entity = []  # each sentence overlapped entities list
    for x, y in zip(entity_hyps, entity_gths):
        cur_overlapped = []
        cur_gths_entity_list = y.split()
        for i in x.split():
            if i in cur_gths_entity_list:
                cur_overlapped.append(i)
        overlapped_entity.append(cur_overlapped)

    overlapped_entity = [list(set(x)) for x in overlapped_entity]  # remove duplicate
    hyp_entity = [set(y.split()) for y in entity_hyps]  # hyp_entity and entity_hyp !
    gth_entity = [set(y.split()) for y in entity_gths]

    entity2prf = OrderedDict()

    assert len(overlapped_entity) == len(hyp_entity) == len(gth_entity)
    for oe, he, ge in zip(overlapped_entity, hyp_entity, gth_entity):
        for e in oe:
            if e not in entity2prf:
                entity2prf = {"FN": 0, "FP": 0, "TP": 0}
            entity2prf[e]["TP"] += 1
            # for current entity "e" that is overlapped, so TP += 1

        for e in he:
            if e not in entity2prf:
                entity2prf = {"FN": 0, "FP": 0, "TP": 0}
            if e not in oe:
                entity2prf[e]["FP"] += 1
            # for current entity "e" that is in hyp but not in overlapped, so FP += 1

        for e in ge:
            if e not in entity2prf:
                entity2prf = {"FN": 0, "FP": 0, "TP": 0}
            if e not in oe:
                entity2prf["FN"] += 1
            # for current entity "e" that is should be in hyp but is not in, FN += 1

    eps = 1e-24
    def compute_f1(p, r):
        return 2 * p * r / (p + r + eps)

    counter = Counter()
    for gth in gth_entity:
        counter.update(gth)

    # Select the entities that appeared larger than 5 times to evaluate.
    need_entity_ind = [x[0] for x in counter.most_common() if x[1] > 5]

    ret_metrics = OrderedDict()
    ret_metric = OrderedDict()

    ret_metrics["ma-P"] = [entity2prf[e]["TP"] / (entity2prf[e]["TP"] + entity2prf[e]["FP"] + eps) for e in
                           need_entity_ind]
    ret_metrics["ma-R"] = [entity2prf[e]["TP"] / (entity2prf[e]["TP"] + entity2prf[e]["FN"] + eps) for e in
                           need_entity_ind]
    ret_metrics["ma-F1"] = [compute_f1(p, r) for (p, r) in zip(ret_metrics["ma-P"], ret_metrics["ma-R"])]
    ret_metric["ma-P"] = float(np.mean(ret_metrics["ma-P"]))
    ret_metric["ma-R"] = float(np.mean(ret_metrics["ma-R"]))
    ret_metric["ma-F1"] = compute_f1(ret_metric["ma-P"], ret_metric["ma-R"])
    mi_precision = [len(x) / (len(y) + 1e-14)
                    for x, y in zip(overlapped_entity, [set(y.split()) for y in entity_hyps])]
    mi_recall = [len(x) / (len(y) + 1e-14)
                 for x, y in zip(overlapped_entity, [set(y.split()) for y in entity_gths])]
    gth_n = [len(set(ws.split())) for ws in entity_gths]
    hyp_n = [len(set(ws.split())) for ws in entity_hyps]
    ret_metric["mi-P"] = np.sum([p * w for (p, w) in zip(mi_precision, hyp_n)]) / np.sum(hyp_n)
    ret_metric["mi-R"] = np.sum([r * w for (r, w) in zip(mi_recall, gth_n)]) / np.sum(gth_n)
    ret_metric["mi-F1"] = compute_f1(ret_metric["mi-P"], ret_metric["mi-R"])
    ret_metrics["mi-P"] = mi_precision
    ret_metrics["mi-R"] = mi_recall
    ret_metrics["mi-F1"] = [compute_f1(p, r) for (p, r) in zip(mi_precision, mi_recall)]
