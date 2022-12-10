import torch
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import Adam
from torch import nn
from src.generator2 import Generator
from nltk.translate.bleu_score import corpus_bleu
from cikm_generate_utils.generator import Greedy, BeamSample
from cikm_dataset.summary_response_dataset import SummaryResponseDataset
import re
from cikm_generate_utils import prepare_input_utils
import os
import json
import time
import math
from ccks_evaluate import KD_Metric
import pickle
from src.eval.eval_utils import calculate_BLEU, input_entities_cal_F1, get_entity_type, sentence_BLEU_avg
from cikm_model.supervised_summary import GenSummaryEntityResponse
from nltk.translate.bleu_score import SmoothingFunction
from torch.nn import functional as F


class SupervisedSummaryTrainer:
    def __init__(self, train_dataset, model, dev_dataset=None, test_dataset=None,
                 config=None, save_root="./cikm_save/Reorganize_model"):
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

        self.summary_gate_open = config['summary_gate_open']
        self.entity_gate_open = config['entity_gate_open']
        self.rsep_as_associate = config['rsep_as_associate']

        self.save_root = save_root
        self.train_dataloader = train_dataset.get_dataloader(shuffle=True, batch_size=self.batch_size, num_workers=8)

        if dev_dataset:
            self.dev_dataloader = dev_dataset.get_dataloader(shuffle=False, batch_size=50)
        if test_dataset:
            self.test_dataloader = test_dataset.get_dataloader(shuffle=False, batch_size=1)

        if self.config.get("pretrain_on_MedDialog"):
            self.train_dataloader = train_dataset.get_pretrain_dataloader(shuffle=True, batch_size=self.batch_size)
            if dev_dataset:
                self.dev_dataloader = dev_dataset.get_pretrain_dataloader(shuffle=False, batch_size=64)
        self.optimizer = Adam(model.parameters(), lr=self.lr, weight_decay=0)

        one_epoch_backward_step_num = math.ceil(len(train_dataset) / self.batch_size)
        total_steps = one_epoch_backward_step_num * self.epoch // self.batch_expand_times
        self.one_epoch_backward_step_num = one_epoch_backward_step_num

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
        # reduction = "mean" if config['entity_kl'] is False else "sum"
        reduction = "sum"
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_idx, reduction=reduction)
        self.gt = Generator(config=self.config)

        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')

    def print_text(self, batch):
        # ref_ent = batch['references_with_entities'][0].tolist()
        # history = batch['history_ids'][0].tolist()
        response = batch['target_ids'][0].tolist()
        print("")
        # print("".join(self.train_dataset.convert_ids_to_tokens(ref_ent)))
        # print("".join(self.train_dataset.convert_ids_to_tokens(history)))
        print("".join(self.train_dataset.convert_ids_to_tokens(response)))
        print("")

    def train_epoch(self, epoch):
        self.model.train()
        iterator_bar = tqdm(self.train_dataloader)

        loss_sum, acc_sum = 0.0, 0.0
        step_num = 0
        self.optimizer.zero_grad()
        epoch_eval_times = 0

        for step, batch in enumerate(iterator_bar):

            if (epoch >= self.config.get("start_eval_epoch", -1)) and (
                    (step + 1) %
                    (self.one_epoch_backward_step_num // self.config.get("eval_times_each_epoch", 3) + 1)
                    == 0
            ):
                epoch_eval_times += 1
                self.eval(f"{epoch}_{epoch_eval_times}", )
            # elif (step + 1) % (self.one_epoch_backward_step_num // 5) == 0:
            #     a = time.strftime("%m%d-%H%M", time.localtime())
            #     filename = "epoch{}-step{}-{}.pt".format(epoch, step, a)
            #     torch.save(self.model.state_dict(), filename)
            # self.save_state_dict(filename="epoch{}-step{}-{}.pt".format(epoch, step, a))

            model_inputs = self.prepare_for_input(batch)
            output = self.model(**model_inputs)
            logits = output[0]

            if self.config.get("model_recall_strategy") is not None:
                # recall logits在logits[1]
                loss, acc, r_acc = self.calculate_loss_and_accuracy_with_two_logits(
                    logits[1], logits[0], batch['padded_summary_target_ids'], batch['padded_response_target_ids']
                )
            else:
                loss, acc, r_acc = self.calculate_loss_and_accuracy(
                    logits, batch['target_ids'], response_start_pos=batch['response_start_pos'],
                    rec_weight=self.config.get("rec_weight", None)
                )

            recall_loss, recall_acc = output[1], output[2]
            if recall_loss is not None:
                iterator_bar.set_description("E[{}] L-[{:.5f}] A[{:.5f}] R-A[{:.5f}] rL[{:.3f}] rA[{:.3f}]".format(
                    epoch, loss.item(), acc.item(), r_acc.item(), recall_loss.item(), recall_acc.item()))
            else:
                iterator_bar.set_description(f"E[{epoch}] L-[{loss:.5f} A-[{acc:.5f}]")


            loss_sum += loss.item()
            acc_sum += acc.item()
            step_num += 1

            if recall_loss is not None:
                loss = loss + recall_loss

            loss /= self.config['batch_expand_times']
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
    def eval(self, epoch, iterator=None, update_score=True):
        self.model.eval()
        if iterator is None:
            iterator = self.dev_dataloader
        gt = Greedy(self.config)
        predict_result = gt.greedy_generate(
            prefix_allowed_tokens_fn=None,
            model=self.model if self.config['parallel'] is None else self.model.module,
            data_iterator=iterator,
            prepare_input_for_encode_step=self.prepare_input_for_greedy_generate,
            two_processor=True
        )
        predict = predict_result['predict']
        reference = predict_result['reference']
        predict = self.select_response_from_result(predict)
        reference = self.select_response_from_result(reference)

        select_to_print_idx = list(range(len(predict_result['predict'])))
        select_to_print_idx = list(set(select_to_print_idx[:3] + select_to_print_idx[-3:]))
        for i in select_to_print_idx:
            print("-" * 30)
            print("predict")
            print(predict_result['predict'][i])
            print(predict[i])
            print("target")
            print(predict_result['reference'][i])
            print(reference[i])
            print("-" * 30)

        predict = [list(_) for _ in predict]
        reference = [list(_) for _ in reference]
        wo_smooth_scores = sentence_BLEU_avg(reference, predict, use_smooth7=True)
        sbleu_1, sbleu_2, sbleu_3, sbleu_4 = wo_smooth_scores['BLEU-1'], wo_smooth_scores['BLEU-2'], \
                                             wo_smooth_scores['BLEU-3'], wo_smooth_scores['BLEU-4']
        print(f"BLEU-1: {sbleu_1}")
        print(f"BLEU-2: {sbleu_2}")
        print(f"BLEU-3: {sbleu_3}")
        print(f"BLEU-4: {sbleu_4}")

        if update_score:
            if sbleu_4 > self.max_bleu:
                self.max_bleu = sbleu_4
                self.save_state_dict("epoch{}-B2[{:.5f}]-B4[{:.5f}].pt".format(epoch, sbleu_2, sbleu_4))
        self.model.train()

    def train(self):
        print("\nStart Training\n")
        for epoch in range(1, self.epoch + 1):
            avg_loss, avg_acc = self.train_epoch(epoch)
            print("# EPOCH[{}] AVG_LOSS[{:.5f}] AVG_ACC[{:.5f}]".format(epoch, avg_loss, avg_acc))
            if self.dev_dataset is not None:
                # if epoch >= 1 and avg_acc > 0.5:
                if epoch >= 1:
                    self.eval(epoch)
                    # self.eval(epoch, update_score=False)
                    # self.eval(epoch, iterator=self.test_dataloader, update_score=True)
                    a = time.strftime("%m%d-%H%M", time.localtime())
                    self.save_state_dict(filename="epoch{}-{}.pt".format(epoch, a))
                else:
                    a = time.strftime("%m%d-%H%M", time.localtime())
                    self.save_state_dict(filename="epoch{}-{}.pt".format(epoch, a))
            else:
                a = time.strftime("%m%d-%H%M", time.localtime())
                self.save_state_dict(filename="epoch{}-{}.pt".format(epoch, a))

    def save_state_dict(self, filename="debug.pt", max_save_num=100):
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

    def calculate_loss_and_accuracy_with_two_logits(self, logits_rc, logits_rp, target_rc, target_rp):
        target_rc = target_rc.to(self.device)
        target_rp = target_rp.to(self.device)
        shift_logits_rc = logits_rc[..., :-1, :].contiguous()
        shift_logits_rp = logits_rp[..., :-1, :].contiguous()
        shift_labels_rc = target_rc[..., 1:].contiguous().to(self.device)
        shift_labels_rp = target_rp[..., 1:].contiguous().to(self.device)
        loss_rc = self.loss_fct(shift_logits_rc.view(-1, shift_logits_rc.size(-1)), shift_labels_rc.view(-1))
        loss_rp = self.loss_fct(shift_logits_rp.view(-1, shift_logits_rp.size(-1)), shift_labels_rp.view(-1))
        _, preds_rc = shift_logits_rc.max(dim=-1)
        _, preds_rp = shift_logits_rp.max(dim=-1)
        not_ignore_rc = shift_labels_rc.ne(self.pad_idx)
        not_ignore_rp = shift_labels_rp.ne(self.pad_idx)
        num_targets_rc = not_ignore_rc.long().sum().item()
        num_targets_rp = not_ignore_rp.long().sum().item()
        correct_rc = (shift_labels_rc == preds_rc) & not_ignore_rc
        correct_rp = (shift_labels_rp == preds_rp) & not_ignore_rp
        correct_rc = correct_rc.float().sum()
        correct_rp = correct_rp.float().sum()
        accuracy_rc = correct_rc / num_targets_rc
        accuracy_rp = correct_rp / num_targets_rp
        loss = (loss_rc + loss_rp) / (num_targets_rc + num_targets_rp)
        return loss, accuracy_rc, accuracy_rp

    def calculate_loss_and_accuracy(self, logits, labels, response_start_pos=None, rec_weight=None):
        if rec_weight is not None:
            return self.calculate_loss_and_accuracy_with_weight(
                logits, labels, response_start_pos=response_start_pos, rec_weight=rec_weight
            )
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(self.device)
        loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        _, preds = shift_logits.max(dim=-1)
        not_ignore = shift_labels.ne(self.pad_idx)
        num_targets = not_ignore.long().sum().item()
        correct = (shift_labels == preds) & not_ignore
        correct = correct.float().sum()
        accuracy = correct / num_targets

        if response_start_pos is not None:
            batch_size, seq_len = shift_labels.shape
            range_matrix = torch.arange(0, seq_len).long().unsqueeze(0).expand(batch_size, seq_len)
            range_matrix = range_matrix.to(shift_labels.device)
            response_start_pos = torch.tensor(response_start_pos).to(shift_labels.device) - 1
            response_start_pos_expand = response_start_pos.unsqueeze(1).expand_as(range_matrix)
            select_response_mask = response_start_pos_expand <= range_matrix
            select_response_mask = select_response_mask & not_ignore
            r_num_targets = select_response_mask.sum().item()
            r_correct = (shift_labels == preds) & select_response_mask
            r_correct = r_correct.float().sum()
            r_accuracy = r_correct / r_num_targets
        else:
            r_accuracy = None

        loss = loss / num_targets
        return loss, accuracy, r_accuracy

    def calculate_loss_and_accuracy_with_weight(self, logits, labels, response_start_pos=None, rec_weight=None):
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
        recall_weight = rec_weight * (~select_response_mask & not_ignore)
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
        # assert batch['history_ids'].shape[0] == 1
        kv_inputs = {
            "input_for_crossattention": batch['history_ids'].to(self.device),
            "crossattention_mask": batch['history_mask'].to(self.device),
            # "response_ids": batch['prefix'][0].unsqueeze(dim=0).to(self.device)
        }

        if self.config.get("use_token_type_ids", False):
            kv_inputs["token_type_ids"] = batch['history_spk'].to(self.device)

        # if self.config['summary_entity_encoder']:
        #     kv_inputs.update({
        #         "summary_ids": batch['summary_ids'].to(self.device),
        #         "summary_mask": batch['summary_mask'].to(self.device),
        #         "entity_ids": batch['entity_ids'].to(self.device),
        #         "entity_mask": batch['entity_mask'].to(self.device),
        #     })

        if self.summary_gate_open:
            kv_inputs.update({
                "summary_ids": batch['summary_ids'].to(self.device),
                "summary_mask": batch['summary_mask'].to(self.device),
            })
            # if self.rsep_as_associate:
            #     kv_inputs.update({
            #         "rsep_position": batch['response_start_pos'] - 1
            #     })

        if self.entity_gate_open:
            kv_inputs.update({
                "entity_ids": batch['entity_ids'].to(self.device),
                "entity_mask": batch['entity_mask'].to(self.device),
            })
        if self.config['recall_gate_network'] == "GAT":
            kv_inputs.update({
                "sentences_ids": batch['sentences_ids'].to(self.device),
                "sentences_mask": batch['sentences_mask'].to(self.device),
                "sentences_num": batch['sentences_num'],
                "adjacent_matrix": batch['adjacent_matrix'].to(self.device),
                "head_type": batch['head_type'].to(self.device),
                "edge_type": batch['edge_type'].to(self.device),
                "target_recall": batch['target_recall'].to(self.device),
            })

        return kv_inputs

    def prepare_for_input(self, batch):
        inputs = {
            "response_mask": batch['target_mask'].to(self.device),
            "response_ids": batch['target_ids'].to(self.device),
            "input_for_crossattention": batch['history_ids'].to(self.device),
            "crossattention_mask": batch['history_mask'].to(self.device)
        }

        if self.config.get("use_token_type_ids", False):
            inputs["token_type_ids"] = batch['history_spk'].to(self.device)

        if self.summary_gate_open:
            inputs.update({
                "summary_ids": batch['summary_ids'].to(self.device),
                "summary_mask": batch['summary_mask'].to(self.device),
            })
            if self.rsep_as_associate:
                inputs.update({
                    "rsep_position": batch['response_start_pos'] - 1
                })

        if self.entity_gate_open:
            inputs.update({
                "entity_ids": batch['entity_ids'].to(self.device),
                "entity_mask": batch['entity_mask'].to(self.device),
            })
        if self.config['recall_gate_network'] == "GAT":
            inputs.update({
                "sentences_ids": batch['sentences_ids'].to(self.device),
                "sentences_mask": batch['sentences_mask'].to(self.device),
                "sentences_num": batch['sentences_num'],
                "adjacent_matrix": batch['adjacent_matrix'].to(self.device),
                "head_type": batch['head_type'].to(self.device),
                "edge_type": batch['edge_type'].to(self.device),
                "target_recall": batch['target_recall'].to(self.device),
            })
        return inputs


@torch.no_grad()
def predict_GenSummaryEntityResponse(
        config=None,
):

    state_dict_name = "my_model_9_1"
    abl_study_state = {
        "my_model": "./cikm_save/SummaryAndResponse/epoch25-B[0.32128]-B1[0.45435]-B4[0.18822].pt",
        "1_no_recall": "./cikm_save/SummaryAndResponse/epoch25-B[0.29957]-B1[0.40998]-B4[0.18915].pt",
        "2_no_struct_enc_but_recall": "./cikm_save/SummaryAndResponse/epoch20-B[0.31543]-B1[0.44604]-B4[0.18482].pt",
        "3_no_entity": "./cikm_save/SummaryAndResponse/epoch13-B[0.26419]-B1[0.40400]-B4[0.12438].pt",
        "e_d": "./cikm_save/SummaryAndResponse/epoch25-B[0.21024]-B1[0.32189]-B4[0.09859].pt",

        "my_model_9_1": "./cikm_save/SummaryAndResponse/epoch7-B[0.33354]-B1[0.47002]-B4[0.19706].pt",

        "from_5_my_model": "./cikm_save/SummaryAndResponse/epoch20-B[0.28614]-B1[0.42722]-B4[0.14506].pt",
        "rsep_as_associate": "./cikm_save/SummaryAndResponse/epoch1-B[0.32994]-B1[0.46015]-B4[0.19972].pt",
        "from_5_indicator": "./cikm_save/SummaryAndResponse/epoch6-B[0.34478]-B1[0.48233]-B4[0.20722].pt",

        "stu_no_recall": "./cikm_save/SummaryAndResponse/epoch3-step44999-0908-1933.pt",

        "stu_no_recall_2": "./cikm_save/SummaryAndResponse/epoch1-0909-0646.pt"
    }
    config['state_dict'] = abl_study_state[state_dict_name]
    config['batch_size'] = 1
    config['model_name'] = "GenSummaryEntityResponse"
    config['rsep_as_associate'] = False
    with_entity = False  # History with entity
    gen_prefix = False
    summary_strategy = "pcl_bert_sim"
    # summary_strategy = "last_3_utterance"
    # summary_strategy = "text_rank"
    if state_dict_name in ['my_model', 'from_5_my_model', "my_model_9_1"]:
        config['recall_gate_network'] = "GAT"  # GAT None
        with_summary = True  # Decoder with summary
        config.update({
            'summary_gate_open': True,
            'entity_gate_open': True
        })
        gen_prefix = True
    elif state_dict_name == "rsep_as_associate":
        config['rsep_as_associate'] = True
        config['recall_gate_network'] = "GAT"
        with_summary = True
        config.update({
            'summary_gate_open': True,
            'entity_gate_open': True
        })
        gen_prefix = True
    elif state_dict_name == "1_no_recall":
        config['recall_gate_network'] = None
        with_summary = False
        config.update({
            'summary_gate_open': False,
            'entity_gate_open': True
        })
    elif state_dict_name == "2_no_struct_enc_but_recall":
        config['recall_gate_network'] = None
        with_summary = True
        config.update({
            'summary_gate_open': False,
            'entity_gate_open': True
        })
    elif state_dict_name == "3_no_entity":
        config['recall_gate_network'] = "GAT"
        with_summary = True
        config.update({
            'summary_gate_open': True,
            'entity_gate_open': False
        })
    elif state_dict_name == "e_d":
        config['recall_gate_network'] = None
        with_summary = False
        config.update({
            'summary_gate_open': False,
            'entity_gate_open': False
        })
    elif state_dict_name in ["stu_no_recall", "stu_no_recall_2"]:
        config['recall_gate_network'] = "GAT"
        with_summary = False
        config.update({
            'summary_gate_open': True,
            'entity_gate_open': True
        })
    else:
        raise ValueError

    model = GenSummaryEntityResponse(config)
    model.load_state_dict(torch.load(config['state_dict'], map_location='cpu'))
    model.eval()

    model.with_target_recall = False

    test_dataset = SummaryResponseDataset(
        vocab_path=config['vocab_path'],
        data_path=config['test_data_path'],
        summary_data_path=config['dialogue_summary_test_path'],
        data_type="test",
        config=config,
        with_entity=with_entity,
        with_summary=with_summary,
        # decoder_with_entity=decoder_with_entity,
        summary_strategy=summary_strategy,
        use_gat=True if config['recall_gate_network'] == "GAT" else False
    )

    generator = BeamSample(config)
    if "no_recall" in state_dict_name:
        gen_func = generator.generate
    else:
        gen_func = generator.with_prefix_generate
    iterator = test_dataset.get_dataloader(batch_size=1, shuffle=False)
    input_process_func = prepare_input_utils.prepare_input_for_GenSummaryEntityResponse
    predict_result = gen_func(
        early_stopping=False,
        prefix_allowed_tokens_fn=None,
        model=model,
        data_iterator=iterator,
        prepare_input_for_encode_step=input_process_func,
        gen_prefix=gen_prefix
    )
    predict_result['predict'] = SupervisedSummaryTrainer.select_response_from_result(predict_result['predict'])
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
