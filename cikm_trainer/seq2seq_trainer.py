import torch
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import Adam
from torch import nn
import time
import os
from src.generator2 import Generator
from nltk.translate.bleu_score import corpus_bleu
from cikm_generate_utils.generator import Greedy


class Seq2SeqTrainer:

    def __init__(self, train_dataset, model, dev_dataset=None, test_dataset=None,
                 config=None, save_root="./cikm_save/Seq2Seq_save"):
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
        if not os.path.exists(save_root):
            os.mkdir(save_root)
        self.save_root = save_root

        self.train_dataloader = train_dataset.get_dataloader(shuffle=True, batch_size=self.batch_size, num_workers=4)
        if dev_dataset:
            self.dev_dataloader = dev_dataset.get_dataloader(shuffle=False, batch_size=8)
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

    def train_epoch(self, epoch):
        self.model.train()
        iterator_bar = tqdm(self.train_dataloader)

        loss_sum, acc_sum = 0.0, 0.0
        step_num = 0
        self.optimizer.zero_grad()
        for step, batch in enumerate(iterator_bar):
            for k in ["history_ids", "history_mask", 'response_ids', 'response_mask']:
                batch[k] = batch[k].to(self.device)
            output = self.model(**batch)
            logits = output[0]
            loss, acc = self.calculate_loss_and_accuracy(logits, batch['response_ids'])
            iterator_bar.set_description("EPOCH[{}] LOSS[{:.5f}] ACC[{:.5f}]".format(
                epoch, loss.item(), acc.item()))
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

    def eval(self, epoch):
        iterator = self.dev_dataloader
        gt = Greedy(self.config)
        predict_result = gt.greedy_generate(
            prefix_allowed_tokens_fn=None,
            model=self.model,
            data_iterator=iterator,
            prepare_input_for_encode_step=self.prepare_input_for_greedy_generate
        )
        predict = predict_result['predict']
        reference = predict_result['reference']
        predict = [list(_) for _ in predict]
        reference = [[list(_)] for _ in reference]
        bleu_1 = corpus_bleu(reference, predict, weights=[1, 0, 0, 0])
        bleu_4 = corpus_bleu(reference, predict, weights=[0.25, 0.25, 0.25, 0.25])
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

    def prepare_input_for_greedy_generate(self, batch, device, expand_batch_size=1):
        history_ids = batch['history_ids'].to(device)
        history_mask = batch['history_mask'].to(device)

        kv_inputs = {
            "history_ids": history_ids,
            "history_mask": history_mask,
        }

        return kv_inputs
