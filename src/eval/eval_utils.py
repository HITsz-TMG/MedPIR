from itertools import chain

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def calculate_BLEU(reference, predict):
    predict_sentence = [list(i) for i in predict]
    reference_sentence_list = [[list(i)] for i in reference]
    b1_s, b2_s, b3_s, b4_s = 0, 0, 0, 0
    for i, j in zip(reference_sentence_list, predict_sentence):
        bleu_1 = sentence_bleu(i, j, weights=[1, 0, 0, 0],
                               smoothing_function=SmoothingFunction().method7)
        bleu_2 = sentence_bleu(i, j, weights=[0.5, 0.5, 0, 0],
                               smoothing_function=SmoothingFunction().method7)
        bleu_3 = sentence_bleu(i, j, weights=[1 / 3, 1 / 3, 1 / 3, 0],
                               smoothing_function=SmoothingFunction().method7)
        bleu_4 = sentence_bleu(i, j, weights=[0.25, 0.25, 0.25, 0.25],
                               smoothing_function=SmoothingFunction().method7)

        b1_s += bleu_1
        b2_s += bleu_2
        b3_s += bleu_3
        b4_s += bleu_4

    bleu_1, bleu_2, bleu_3, bleu_4 = b1_s / len(predict_sentence), b2_s / len(predict_sentence), \
                                     b3_s / len(predict_sentence), b4_s / len(predict_sentence)
    #
    # bleu_1 = corpus_bleu(reference_sentence_list, predict_sentence, weights=[1, 0, 0, 0],
    #                      # smoothing_function=SmoothingFunction().method7
    #                      )
    # bleu_2 = corpus_bleu(reference_sentence_list, predict_sentence, weights=[0.5, 0.5, 0, 0],
    #                      # smoothing_function=SmoothingFunction().method7
    #                      )
    # bleu_3 = corpus_bleu(reference_sentence_list, predict_sentence, weights=[1 / 3, 1 / 3, 1 / 3, 0],
    #                      # smoothing_function=SmoothingFunction().method7
    #                      )
    # bleu_4 = corpus_bleu(reference_sentence_list, predict_sentence, weights=[0.25, 0.25, 0.25, 0.25],
    #                      # smoothing_function=SmoothingFunction().method7
    #                      )

    return {
        "BLEU-1": bleu_1,
        "BLEU-2": bleu_2,
        "BLEU-3": bleu_3,
        "BLEU-4": bleu_4
    }


def input_entities_cal_F1(golden_entities, predict_entities, config):
    pred_pos_num = 0
    pred_pos_correct_num = 0
    # precision = pred_pos_correct_num/pred_pos_num
    real_pos_num = 0
    # recall = pred_pos_correct_num/real_pos_num
    cnt_dict = dict()
    for et in config['entity_type']:
        cnt_dict[et] = {
            "pred_pos_num": 0,
            "pred_pos_correct_num": 0,
            "real_pos_num": 0
        }

    for test_item, predict_item in zip(golden_entities, predict_entities):
        for et in config['entity_type']:
            real_pos_entities = test_item[et]
            pred_entities = predict_item[et]

            pred_pos_num += len(pred_entities)
            real_pos_num += len(real_pos_entities)
            correct_pred_num = len([1 for i in pred_entities if i in real_pos_entities])
            pred_pos_correct_num += correct_pred_num

            cnt_dict[et]['pred_pos_num'] += len(pred_entities)
            cnt_dict[et]['real_pos_num'] += len(real_pos_entities)
            cnt_dict[et]['pred_pos_correct_num'] += correct_pred_num

    category_F1 = {et: None for et in config['entity_type']}

    for et in config['entity_type']:
        precision = cnt_dict[et]['pred_pos_correct_num'] / cnt_dict[et]['pred_pos_num'] if cnt_dict[et][
                                                                                               'pred_pos_num'] != 0 else 0
        recall = cnt_dict[et]['pred_pos_correct_num'] / cnt_dict[et]['real_pos_num'] if cnt_dict[et][
                                                                                            'real_pos_num'] != 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        print("{}: P={:.5f} R={:.5f} F1={:.5f}".format(et, precision, recall, f1))

        category_F1[et] = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'pred_num': cnt_dict[et]['pred_pos_num'],
            'real_num': cnt_dict[et]['real_pos_num']
        }

    precision = pred_pos_correct_num / pred_pos_num
    recall = pred_pos_correct_num / real_pos_num
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    print("P={:.5f} R={:.5f} F1={:.5f}".format(precision, recall, f1))

    res = {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'pred_num': pred_pos_num,
        'real_num': real_pos_num,
        'category_f1': category_F1,
    }
    print(res)
    return res


def get_entity_type(ent, config):
    ent = ent.lower()
    if ent in config['symptom']:
        return "Symptom"
    elif ent in config['medicine']:
        return "Medicine"
    elif ent in config["test"]:
        return "Test"
    elif ent in config['attribute']:
        return "Attribute"
    elif ent in config['disease']:
        return "Disease"
    else:
        raise RuntimeError(" \"{}\"'s entity type not found".format(ent))


def sentence_BLEU_avg(reference, predict, use_smooth7=True):
    predict_sentence = [list(i) for i in predict]
    reference_sentence_list = [[list(i)] for i in reference]
    b1_s, b2_s, b3_s, b4_s = 0, 0, 0, 0
    for i, j in zip(reference_sentence_list, predict_sentence):
        if use_smooth7:
            try:
                bleu_1 = sentence_bleu(i, j, weights=[1, 0, 0, 0],
                                       smoothing_function=SmoothingFunction().method7)
            except ZeroDivisionError:
                bleu_1 = 0
            try:
                bleu_2 = sentence_bleu(i, j, weights=[0.5, 0.5, 0, 0],
                                       smoothing_function=SmoothingFunction().method7)
            except ZeroDivisionError:
                bleu_2 = 0
            try:
                bleu_3 = sentence_bleu(i, j, weights=[1 / 3, 1 / 3, 1 / 3, 0],
                                       smoothing_function=SmoothingFunction().method7)
            except ZeroDivisionError:
                bleu_3 = 0
            try:
                bleu_4 = sentence_bleu(i, j, weights=[0.25, 0.25, 0.25, 0.25],
                                       smoothing_function=SmoothingFunction().method7)
            except ZeroDivisionError:
                bleu_4 = 0
        else:
            try:
                bleu_1 = sentence_bleu(i, j, weights=[1, 0, 0, 0], )
            except ZeroDivisionError:
                bleu_1 = 0
            try:
                bleu_2 = sentence_bleu(i, j, weights=[0.5, 0.5, 0, 0], )
            except ZeroDivisionError:
                bleu_2 = 0
            try:
                bleu_3 = sentence_bleu(i, j, weights=[1 / 3, 1 / 3, 1 / 3, 0], )
            except ZeroDivisionError:
                bleu_3 = 0
            try:
                bleu_4 = sentence_bleu(i, j, weights=[0.25, 0.25, 0.25, 0.25], )
            except ZeroDivisionError:
                bleu_4 = 0
        b1_s += bleu_1
        b2_s += bleu_2
        b3_s += bleu_3
        b4_s += bleu_4

    bleu_1, bleu_2, bleu_3, bleu_4 = b1_s / len(predict_sentence), b2_s / len(predict_sentence), \
                                     b3_s / len(predict_sentence), b4_s / len(predict_sentence)

    return {
        "BLEU-1": bleu_1,
        "BLEU-2": bleu_2,
        "BLEU-3": bleu_3,
        "BLEU-4": bleu_4
    }


def line_fn_origin(line):
    return line


def get_line_fn_split(split_tag="<==>", index=1):
    def line_fn(line):
        return line.split(split_tag)[index]

    return line_fn


def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):
    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def distinct_n_sentence_level(sentence, n):
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    # return distinct_ngrams, len(sentence)
    return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level(sentences, n):
    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)
    # all_distinct = set()
    # all_len = 0
    # for sentence in sentences:
    #     cur_distinct_ngrams, cur_len = distinct_n_sentence_level(sentence, n)
    #     all_distinct.update(cur_distinct_ngrams)
    #     all_len += cur_len
    # return len(all_distinct) / all_len
