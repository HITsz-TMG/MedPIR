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
from ccks_evaluate import KD_Metric
from cikm_entity_annotation import input_entities_cal_F1, get_entity_type
import pickle
from cikm_main import calculate_BLEU
from cikm_model.supervised_summary import GenSummaryEntityResponse


class VMedTrainer:
    def __init__(self, train_dataset, model, dev_dataset=None, test_dataset=None,
                 config=None, save_root="./cikm_save/vmed_save"):
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

        self.train_dataloader = train_dataset.get_dataloader(shuffle=True, batch_size=self.batch_size, num_workers=1)
        if dev_dataset:
            self.dev_dataloader = dev_dataset.get_dataloader(shuffle=False, batch_size=1)
        if test_dataset:
            self.test_dataloader = test_dataset.get_dataloader(shuffle=False, batch_size=1)

        self.optimizer = Adam(model.parameters(), lr=self.lr, weight_decay=0)
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
        # reduction = "mean" if config['entity_kl'] is False else "sum"
        reduction = "sum"
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_idx, reduction=reduction)
        self.gt = Generator(config=self.config)

        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')

    def print_text(self, batch):
        ref_ent = batch['references_with_entities'][0].tolist()
        history = batch['history_ids'][0].tolist()
        response = batch['response_ids'][0].tolist()
        print("")
        print("".join(self.train_dataset.convert_ids_to_tokens(ref_ent)))
        print("".join(self.train_dataset.convert_ids_to_tokens(history)))
        print("".join(self.train_dataset.convert_ids_to_tokens(response)))
        print("")

    def train_epoch(self, epoch):
        self.model.train()
        iterator_bar = tqdm(self.train_dataloader)

        loss_sum, acc_sum = 0.0, 0.0
        step_num = 0
        self.optimizer.zero_grad()
        for step, batch in enumerate(iterator_bar):
            model_inputs = self.prepare_for_input(batch)
            # self.print_text(batch)

            output = self.model(**model_inputs)

            logits = output[0]
            loss, acc, r_acc = self.calculate_loss_and_accuracy(
                logits, batch['target_ids'], response_start_pos=batch['response_start_pos']
            )

            iterator_bar.set_description("EPOCH[{}] LOSS[{:.5f}] ACC[{:.5f}] R-ACC[{:.5f}]".format(
                epoch, loss.item(), acc.item(), r_acc.item()))

            loss_sum += loss.item()
            acc_sum += acc.item()
            step_num += 1

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

    def select_response_from_result(self, list_sentence):
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
            model=self.model if self.config['parallel'] is None else self.model.module,
            data_iterator=iterator,
            prepare_input_for_encode_step=self.prepare_input_for_greedy_generate
        )
        predict = predict_result['predict']
        reference = predict_result['reference']
        predict = self.select_response_from_result(predict)
        reference = self.select_response_from_result(reference)

        for i in range(5):
            print("-" * 30)
            print(predict_result['predict'][i])
            print(predict[i])
            print(predict_result['reference'][i])
            print(reference[i])
            print("-" * 30)

        predict = [list(_) for _ in predict]
        reference = [[list(_)] for _ in reference]
        bleu_1 = corpus_bleu(reference, predict, weights=[1, 0, 0, 0])
        bleu_4 = corpus_bleu(reference, predict, weights=[0.25, 0.25, 0.25, 0.25])
        bleu = (bleu_1 + bleu_4) / 2
        print("bleu:{:.5f}  bleu1:{:.5f}  bleu4:{:.5f}".format(bleu, bleu_1, bleu_4))
        if bleu > self.max_bleu:
            self.max_bleu = bleu
            self.save_state_dict("epoch{}-B[{:.5f}]-B1[{:.5f}]-B4[{:.5f}].pt".format(epoch, bleu, bleu_1, bleu_4))
        self.model.train()

    def train(self):
        print("\nStart Training\n")
        for epoch in range(1, self.epoch + 1):
            avg_loss, avg_acc = self.train_epoch(epoch)
            print("# EPOCH[{}] AVG_LOSS[{:.5f}] AVG_ACC[{:.5f}]".format(epoch, avg_loss, avg_acc))
            if self.dev_dataset is not None:
                if epoch >= 1:
                    self.eval(epoch)
                else:
                    a = time.strftime("%m%d-%H%M", time.localtime())
                    self.save_state_dict(filename="epoch{}-{}.pt".format(epoch, a))
            else:
                a = time.strftime("%m%d-%H%M", time.localtime())
                self.save_state_dict(filename="epoch{}-{}.pt".format(epoch, a))

    def save_state_dict(self, filename="debug.pt", max_save_num=3):
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

    def calculate_loss_and_accuracy(self, logits, labels, response_start_pos=None):
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

    def prepare_input_for_greedy_generate(self, batch, device, expand_batch_size=1):
        pass
        # assert batch['history_ids'].shape[0] == 1
        # kv_inputs = {
        #     "input_for_crossattention": batch['history_ids'].to(self.device),
        #     "crossattention_mask": batch['history_mask'].to(self.device),
        #     # "response_ids": batch['prefix'][0].unsqueeze(dim=0).to(self.device)
        # }
        # if self.config['summary_entity_encoder']:
        #     kv_inputs.update({
        #         "summary_ids": batch['summary_ids'].to(self.device),
        #         "summary_mask": batch['summary_mask'].to(self.device),
        #         "entity_ids": batch['entity_ids'].to(self.device),
        #         "entity_mask": batch['entity_mask'].to(self.device),
        #     })

        # return kv_inputs

    def prepare_for_input(self, batch):
        inputs = {
            "sentences_ids": batch['sentences_ids'].to(self.device),
            "original_sentence": batch['original_sentence'],
            "sentence_num": batch['sentence_num'],
            "sentence_adjacent_matrix": batch['sentence_adjacent_matrix'].to(self.device),
            "head_type": batch['head_type'].to(self.device),
            "edge_type": batch['edge_type'].to(self.device),
            "target_response": batch['target_response'].to(self.device),
            "original_target_response": batch['original_target_response'],
            "entity_for_decoder_ids": batch['entity_for_decoder_ids'],
            "input_for_crossattention": batch['input_for_crossattention'].to(self.device),
            "entity_ids": batch['entity_ids'].to(self.device),
            "entity_mask": batch['entity_mask'].to(self.device),
            "sentences_ids_mask": batch['sentences_ids_mask'].to(self.device),
            "target_response_mask": batch['target_response_mask'].to(self.device),
        }
        return inputs

