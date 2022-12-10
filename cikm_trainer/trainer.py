import torch
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import Adam
from torch import nn
import time
import os
import re
from torch.nn import functional as F
from nltk.translate.bleu_score import corpus_bleu
from cikm_generate_utils.generator import Greedy
from nltk.translate.bleu_score import SmoothingFunction

from src.model import BERTGPTEntity

from cikm_dataset.summary_response_dataset import SummaryResponseDataset
from cikm_generate_utils.generator import BeamSample
from cikm_generate_utils import prepare_input_utils
import json
from ccks_evaluate import KD_Metric
import pickle
from src.eval.eval_utils import calculate_BLEU, input_entities_cal_F1, get_entity_type


class CIKMTrainer:

    def __init__(self, train_dataset, model, dev_dataset=None, test_dataset=None,
                 config=None, save_root="./save/"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.model = model
        self.config = config
        self.model_name = config['model_name']
        self.epoch = config['epoch']
        self.batch_size = config['batch_size']
        self.batch_expand_times = config['batch_expand_times']
        self.lr = config['lr']

        self.save_root = save_root

        self.train_dataloader = train_dataset.get_dataloader(shuffle=True, batch_size=self.batch_size, num_workers=4)
        if dev_dataset:
            self.dev_dataloader = dev_dataset.get_dataloader(shuffle=False, batch_size=1)
        if test_dataset:
            self.test_dataloader = test_dataset.get_dataloader(shuffle=False, batch_size=self.batch_size)

        self.optimizer = Adam(model.parameters(), lr=self.lr, weight_decay=config.get("weight_decay", 0.005))
        total_steps = int(len(train_dataset) * self.epoch / (self.batch_size * self.batch_expand_times))

        warm_up_step = config['warm_up'] if config['warm_up'] >= 1 else int(total_steps * config['warm_up'])
        print("warm-up/total-step: {}/{}".format(warm_up_step, total_steps))
        self.schedule = get_linear_schedule_with_warmup(
            self.optimizer, num_training_steps=total_steps, num_warmup_steps=warm_up_step)

        self.config = config

        if config.get('parallel', None) == 'data_parallel':
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        self.save_name_list = []
        self.total_steps = total_steps
        self.max_acc = float('-inf')
        self.max_bleu = float('-inf')

        self.pad_idx = self.train_dataset.pad_idx
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_idx, reduction='sum')

    def train_input(self, batch):
        inputs = {
            "history_ids": batch['history_ids'].to(self.device),
            "history_mask": batch['history_mask'].to(self.device),
            "response_ids": batch['target_ids'].to(self.device) if batch.get("target_ids") else batch[
                'response_ids'].to(self.device),
            "response_mask": batch['target_mask'].to(self.device) if batch.get("target_mask") else batch[
                'response_mask'].to(self.device),
        }
        return inputs

    def train_epoch(self, epoch):
        self.model.train()
        iterator_bar = tqdm(self.train_dataloader)

        loss_sum, acc_sum = 0.0, 0.0
        step_num = 0
        self.optimizer.zero_grad()
        for step, batch in enumerate(iterator_bar):
            # for k in batch.keys():
            #     if k.startswith("group"):
            #         for idx in range(len(batch[k])):
            #             batch[k][idx] = batch[k][idx].to(self.device)
            #     elif type(batch[k]) == list:
            #         continue
            #     else:
            #         batch[k] = batch[k].to(self.device)
            # output = self.model(**batch)
            output = self.model(**self.train_input(batch))
            logits = output[0]

            if self.config['recall'] is False:
                loss, acc = self.calculate_loss_and_accuracy(logits, batch['response_ids'])
                iterator_bar.set_description("EPOCH[{}] LOSS[{:.5f}] ACC[{:.5f}]".format(
                    epoch, loss.item(), acc.item()))
            else:
                loss, acc, r_acc = self.calculate_loss_and_accuracy_with_recall(
                    logits, batch['target_ids'], response_start_pos=batch['response_start_pos']
                )
                iterator_bar.set_description("E[{}] L-[{:.5f}] A[{:.5f}] R-A[{:.5f}]".format(
                    epoch, loss.item(), acc.item(), r_acc.item()))

            loss_sum += loss.item()
            acc_sum += acc.item()
            step_num += 1
            loss.backward()
            if ((step + 1) % self.batch_expand_times) == 0:
                self.optimizer.step()
                self.schedule.step()
                self.optimizer.zero_grad()
        self.optimizer.step()
        self.optimizer.zero_grad()
        avg_loss = loss_sum / step_num
        avg_acc = acc_sum / step_num
        return avg_loss, avg_acc

    @staticmethod
    def select_response_from_result(list_sentence):
        summary_end = '\\[SummaryEnd\\]'
        entity_end = '\\[EntityEnd\\]'

        result = []
        for sent in list_sentence:
            end1, end2 = 0, 0
            re1 = list(re.finditer(summary_end, sent))
            if re1:
                end1 = re1[-1].end()
            re2 = list(re.finditer(entity_end, sent))
            if re2:
                end2 = re2[-1].end()
            start = max(end1, end2)
            result.append(sent[start:])

        return result

    @torch.no_grad()
    def eval(self, epoch):
        self.model.eval()
        iterator = self.dev_dataloader
        gt = Greedy(self.config)
        predict_result = gt.greedy_generate(
            prefix_allowed_tokens_fn=None,
            model=self.model,
            data_iterator=iterator,
            prepare_input_for_encode_step=self.prepare_input_for_greedy_generate,
            with_prefix=self.config.get("recall", False),
        )
        predict = predict_result['predict']
        reference = predict_result['reference']
        predict = self.select_response_from_result(predict)
        reference = self.select_response_from_result(reference)

        print(predict[0])
        print(reference[0])

        bleu_1 = corpus_bleu(reference, predict, weights=[1, 0, 0, 0],
                             smoothing_function=SmoothingFunction().method7)
        bleu_4 = corpus_bleu(reference, predict, weights=[0.25, 0.25, 0.25, 0.25],
                             smoothing_function=SmoothingFunction().method7)
        bleu = (bleu_1 + bleu_4) / 2
        print("bleu:{:.5f}  bleu1:{:.5f}  bleu4:{:.5f}".format(bleu, bleu_1, bleu_4))
        if bleu > self.max_bleu:
            self.max_bleu = bleu
            self.save_state_dict("epoch{}-B[{:.5f}]-B1[{:.5f}]-B4[{:.5f}].pt".format(epoch, bleu, bleu_1, bleu_4))

    def train(self):
        print("\nStart Training\n")
        for epoch in range(1, self.epoch + 1):
            avg_loss, avg_acc = self.train_epoch(epoch)
            print("# EPOCH[{}] AVG_LOSS[{:.5f}] AVG_ACC[{:.5f}]".format(epoch, avg_loss, avg_acc))
            if epoch >= 1:
                self.eval(epoch)
            else:
                a = time.strftime("%m%d-%H%M", time.localtime())
                self.save_state_dict(filename="epoch{}-{}.pt".format(epoch, a))

    def save_state_dict(self, filename="debug.pt", max_save_num=1):
        save_path = os.path.join(self.save_root, filename)
        if self.config.get('parallel', None) == 'data_parallel':
            torch.save(self.model.module.state_dict(), save_path)
        else:
            torch.save(self.model.state_dict(), save_path)
        check_list = []
        for i in self.save_name_list:
            if os.path.exists(i):
                check_list.append(i)
        if len(check_list) == max_save_num:
            del_file = check_list.pop(0)
            os.remove(del_file)
        self.save_name_list = check_list
        self.save_name_list.append(save_path)

    def calculate_loss_and_accuracy(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(self.device)
        loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        _, preds = shift_logits.max(dim=-1)
        not_ignore = shift_labels.ne(self.pad_idx)
        num_targets = not_ignore.long().sum().item()
        correct = (shift_labels == preds) & not_ignore
        correct = correct.float().sum()
        accuracy = correct / num_targets
        loss = loss / num_targets
        return loss, accuracy

    def calculate_loss_and_accuracy_with_recall(self, logits, labels, response_start_pos=None):
        assert response_start_pos is not None
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(self.device)

        _, preds = shift_logits.max(dim=-1)
        not_ignore = shift_labels.ne(self.pad_idx)
        num_targets = not_ignore.long().sum().item()
        correct = (shift_labels == preds) & not_ignore
        correct = correct.float().sum()
        accuracy = correct / num_targets

        batch_size, seq_len = shift_labels.shape
        range_matrix = torch.arange(0, seq_len).long().unsqueeze(0).expand(batch_size, seq_len)
        range_matrix = range_matrix.to(shift_labels.device)
        response_start_pos = response_start_pos.to(shift_labels.device) - 1
        response_start_pos_expand = response_start_pos.unsqueeze(1).expand_as(range_matrix)
        select_response_mask = response_start_pos_expand <= range_matrix
        select_response_mask = select_response_mask & not_ignore  # False处不是response
        r_num_targets = select_response_mask.sum().item()
        r_correct = (shift_labels == preds) & select_response_mask
        r_correct = r_correct.float().sum()
        r_accuracy = r_correct / r_num_targets

        response_weight = 1.0 * select_response_mask
        recall_weight = 0.0 * (~select_response_mask & not_ignore)
        weight = response_weight + recall_weight
        # loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), )
        # loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
        #                        weight=weight.view(-1),
        #                        reduction="sum")

        loss = self.ce_loss_pos_weight(shift_logits.view(-1, shift_logits.size(-1)),
                                       shift_labels.view(-1), weight.view(-1))
        normalize_factor = weight.sum().item()
        loss = loss / normalize_factor

        # loss = loss / num_targets

        return loss, accuracy, r_accuracy

    @staticmethod
    def ce_loss_pos_weight(logits, target, pos_weight):
        target = target.reshape(logits.shape[0], 1)
        log_pro = -1.0 * F.log_softmax(logits, dim=1)
        one_hot = torch.zeros(logits.shape[0], logits.shape[1]).cuda()
        one_hot = one_hot.scatter_(1, target, 1)
        loss = torch.mul(log_pro, one_hot).sum(dim=1)
        loss = loss * pos_weight
        loss = loss.sum()
        return loss

    def prepare_input_for_greedy_generate(self, batch, device, expand_batch_size=1):

        config = self.config
        history_ids = batch['history_ids'].to(device)
        history_mask = batch['history_mask'].to(device)
        history_speaker = batch['history_speaker'].to(device) if config['use_token_type_ids'] else None
        group_history_ids = None
        group_history_mask = None
        start_end_expand_as_batch_size = None
        each_sample_chunks_num = None

        kv_inputs = {
            "history_ids": history_ids,
            "token_type_ids": history_speaker,
            "history_mask": history_mask,
            "group_history_ids": group_history_ids,
            "group_history_mask": group_history_mask,
            "start_end_expand_as_batch_size": start_end_expand_as_batch_size,
            "each_sample_chunks_num": each_sample_chunks_num,
        }

        return kv_inputs


@torch.no_grad()
def predict_GenRecallBERTGPT(
        config=None,
):
    with_entity = True  # History with entity
    config['state_dict'] = "./cikm_save/RecallBertGpt/B[0.35621]-B1[0.59724]-B4[0.11518].pt"
    config['batch_size'] = 1
    config['model_name'] = "BERTGPTEntity"

    config['entity_appendix'] = True
    config['expand_token_type_embed'] = False
    config['recall'] = True
    config['rsep_as_associate'] = False

    config["test_data_path"] = "./data/cikm/response_entity_predict_new-0.35.pkl"

    gen_prefix = False
    summary_strategy = "pcl_bert_sim"


    model = BERTGPTEntity(config)
    model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))
    model.eval()

    test_dataset = SummaryResponseDataset(
        vocab_path=config['vocab_path'],
        data_path=config['test_data_path'],
        summary_data_path=config['dialogue_summary_test_path'],
        data_type="test",
        config=config,
        with_entity=with_entity,
        with_summary=True,
        # decoder_with_entity=decoder_with_entity,
        summary_strategy=summary_strategy,
        use_gat=False
    )

    generator = BeamSample(config)
    gen_func = generator.with_prefix_generate
    iterator = test_dataset.get_dataloader(batch_size=1, shuffle=False)
    input_process_func = prepare_input_utils.prepare_input_for_RecallBERTGPT
    predict_result = gen_func(
        early_stopping=False,
        prefix_allowed_tokens_fn=None,
        model=model,
        data_iterator=iterator,
        prepare_input_for_encode_step=input_process_func,
        gen_prefix=gen_prefix
    )
    predict_result['predict'] = CIKMTrainer.select_response_from_result(predict_result['predict'])

    predict_result['reference'] = [i['text'][1]['Sentence'] for i, _ in test_dataset.data]

    complete_time = time.strftime("%m-%d-%H-%M")
    save_root = "./cikm_predict_result"
    file_name = "{}-{}-{}.json".format(
        config['model_name'],
        summary_strategy,
        complete_time
    )
    with open(os.path.join(save_root, file_name), 'w', encoding="utf-8") as writer:
        dump_dict = {
            "generate_args": {
                "top_k": config['top_k'],
                "beam_size": config['beam_size'],
            },
            "predict": [i for i in predict_result['predict']]
        }
        json.dump(dump_dict, writer, ensure_ascii=False)

    scores = calculate_BLEU(predict_result['reference'], predict_result['predict'])

    kdm = KD_Metric()
    predict_entities = []
    for i in predict_result['predict']:
        entities = kdm.convert_sen_to_entity_set(i)
        new_item = {
            "Symptom": [],
            "Medicine": [],
            "Test": [],
            "Attribute": [],
            "Disease": []
        }
        for e in entities:
            new_item[get_entity_type(e, config)].append(e)
        predict_entities.append(new_item)

    golden_entities = []
    for i in predict_result['reference']:
        entities = kdm.convert_sen_to_entity_set(i)
        new_item = {
            "Symptom": [],
            "Medicine": [],
            "Test": [],
            "Attribute": [],
            "Disease": []
        }
        for e in entities:
            new_item[get_entity_type(e, config)].append(e)
        golden_entities.append(new_item)

    f1_info = input_entities_cal_F1(golden_entities, predict_entities, config)

    with open(os.path.join(save_root, file_name), 'w', encoding="utf-8") as writer:
        dump_dict = {
            "score": scores,
            "f1_info": f1_info,
            "generate_args": {
                "top_k": config['top_k'],
                "beam_size": config['beam_size'],
            },
            "predict": [i for i in predict_result['predict']]
        }
        json.dump(dump_dict, writer, ensure_ascii=False)
