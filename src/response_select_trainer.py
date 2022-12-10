import torch
from src.response_select import ResponseSelector
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import Adam
from torch import nn
import time
import os


class ResponseSelectTrainer:

    def __init__(self, train_dataset, model: ResponseSelector, dev_dataset=None, test_dataset=None,
                 config=None, save_root="./save/ResponseSelect_save"):
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

        self.train_dataloader = train_dataset.get_dataloader(shuffle=True, batch_size=self.batch_size)
        self.dev_dataloader = dev_dataset.get_dataloader(shuffle=False, batch_size=self.batch_size)
        param_optimizer = list(model.named_parameters())
        bak = list(param_optimizer)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        no_decay_group = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad]
        param_optimizer = list(filter(lambda x: x not in no_decay_group, param_optimizer))
        encoder_group = [(n, p) for n, p in param_optimizer if n.startswith('encoder.') and p.requires_grad]
        param_optimizer = list(filter(lambda x: x not in encoder_group, param_optimizer))
        other_group = param_optimizer
        assert len(bak) == len(other_group) + + len(encoder_group) + len(no_decay_group)
        optimizer_grouped_parameters = [
            {"params": [p for n, p in encoder_group],
             "weight_decay": config.get("weight_decay", 0.005),
             "lr": self.lr},
            {"params": [p for n, p in no_decay_group],
             "weight_decay": 0.0,
             "lr": self.lr},
            {"params": [p for n, p in other_group],
             "weight_decay": config.get("weight_decay", 0.005),
             "lr": self.lr * 2}
        ]
        self.optimizer = Adam(optimizer_grouped_parameters, lr=self.lr, weight_decay=config.get("weight_decay", 0.005))

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
        self.max_F1 = float('-inf')

        self.pad_idx = self.train_dataset.pad_idx

        # self.summary = SummaryWriter('./summary/result')

    def train_epoch(self, epoch):
        self.model.train()

        iterator_bar = tqdm(self.train_dataloader)
        loss_sum = 0.0
        step_num = 0
        self.optimizer.zero_grad()
        for step, batch in enumerate(iterator_bar):
            for k in batch.keys():
                batch[k] = batch[k].to(self.device)
            output = self.model(**batch)
            loss = output[1]
            iterator_bar.set_description("EPOCH[{}] LOSS[{:.5f}]".format(epoch, loss.item()))
            loss_sum += loss.item()
            step_num += 1
            loss.backward()
            if ((step + 1) % self.batch_expand_times) == 0:
                self.optimizer.step()
                self.schedule.step()
                self.optimizer.zero_grad()
            if (step + 1) % 4000 == 0:
                self.eval(epoch, step, len(iterator_bar))
        self.optimizer.step()
        self.optimizer.zero_grad()
        avg_loss = loss_sum / step_num
        return avg_loss

    def eval(self, epoch, step=None, total_step=None):
        self.model.eval()

        positive = 0
        correct_positive = 0
        pred_positive = 0
        pred_negative = 0
        with torch.no_grad():
            iterator_bar = self.dev_dataloader
            correct_num = 0
            total = 0
            for batch in iterator_bar:
                for k in batch.keys():
                    batch[k] = batch[k].to(self.device)
                output = self.model(**batch)
                scores = output[0]
                pred = (scores > 0.5).long()
                pred_positive += pred.sum().item()
                pred_negative += (pred <= 0.5).long().sum().item()

                correct = (batch["labels"] == pred).sum().item()
                correct_positive += (pred * batch["labels"]).sum().item()
                real_positive = batch["labels"].sum().item()
                positive += real_positive

                correct_num += correct
                total += len(pred.view(-1))

        acc = correct_num / total
        pos_acc = correct_positive / positive

        print("correct pos{}, pos{}".format(correct_positive, positive))
        print("pred pos{}, pred neg{}".format(pred_positive, pred_negative))
        P = correct_positive / pred_positive if pred_positive != 0 else 0
        R = correct_positive / positive if positive != 0 else 0
        F1 = (2 * P * R) / (P + R) if (P + R) != 0 else 0
        print("POS_P[{:.5f}] POS_R[{:.5f}] F1[{:.5f}]".format(P, R, F1))
        print("# EPOCH[{}] ACC[{:.5f}] POS_ACC[{:.5f}]".format(epoch, acc, pos_acc), end='')

        if self.config['metric'] == 'acc':
            if acc > self.max_acc:
                self.max_acc = acc
                self.save_state_dict("epoch{}-acc{:.5f}.{}".format(epoch, acc, self.model_name))
                print("\t saved")
            else:
                print("")
        elif self.config['metric'] == 'F1':
            if F1 > self.max_F1:
                self.max_F1 = F1
                self.save_state_dict("epoch{}-F1{:.5f}.{}".format(epoch, F1, self.model_name))
                print("\t saved")
            else:
                print("")

        self.model.train()
        return F1

    def train(self):
        print("\nStart Training\n")
        # self.eval(-1)

        for epoch in range(1, self.epoch + 1):
            avg_loss = self.train_epoch(epoch)
            acc = self.eval(epoch)
            print("# EPOCH[{}] AVG_LOSS[{:.5f}]".format(epoch, avg_loss))

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
