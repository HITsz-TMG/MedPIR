import torch
from src.spk_predict import SpkPredict
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import Adam
from torch import nn
import time
import os


class SpkPredTrainer:

    def __init__(self, train_dataset, model: SpkPredict, dev_dataset=None, test_dataset=None,
                 config=None, save_root="./save/SpkPred_save"):
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

        # param_optimizer = list(model.named_parameters())
        # bak = list(param_optimizer)
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # no_decay_group = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad]
        # param_optimizer = list(filter(lambda x: x not in no_decay_group, param_optimizer))
        # encoder_group = [(n, p) for n, p in param_optimizer if n.startswith('encoder.') and p.requires_grad]
        # param_optimizer = list(filter(lambda x: x not in encoder_group, param_optimizer))
        # decoder_group = [(n, p) for n, p in param_optimizer if n.startswith('decoder.') and p.requires_grad]
        # param_optimizer = list(filter(lambda x: x not in decoder_group, param_optimizer))
        # other_group = param_optimizer
        # assert len(bak) == len(other_group) + len(decoder_group) + len(encoder_group) + len(no_decay_group)
        # optimizer_grouped_parameters = [
        #     {"params": [p for n, p in decoder_group],
        #      "weight_decay": config.get("weight_decay", 0.005),
        #      "lr": self.lr},
        #     {"params": [p for n, p in encoder_group],
        #      "weight_decay": config.get("weight_decay", 0.005),
        #      "lr": self.lr * config["encoder_lr_factor"]},
        #     {"params": [p for n, p in no_decay_group],
        #      "weight_decay": 0.0,
        #      "lr": self.lr},
        #     {"params": [p for n, p in other_group],
        #      "weight_decay": config.get("weight_decay", 0.005),
        #      "lr": self.lr}
        # ]
        # self.optimizer = Adam(optimizer_grouped_parameters, lr=self.lr,
        # weight_decay=config.get("weight_decay", 0.005))

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

        self.pad_idx = self.train_dataset.pad_idx
        self.loss_fct = nn.CrossEntropyLoss(reduction='mean')

        # self.summary = SummaryWriter('./summary/result')

    def train_epoch(self, epoch):
        self.model.train()
        if epoch > 1:
            iterator_bar = self.train_dataloader
        else:
            iterator_bar = tqdm(self.train_dataloader)
        loss_sum = 0.0
        step_num = 0
        self.optimizer.zero_grad()
        for step, batch in enumerate(iterator_bar):
            for k in batch.keys():
                batch[k] = batch[k].to(self.device)
            output = self.model(**batch)
            logits = output[0]
            loss = self.loss_fct(logits, batch['label'])
            if epoch <= 1:
                iterator_bar.set_description("EPOCH[{}] LOSS[{:.5f}]".format(epoch, loss.item()))
            loss_sum += loss.item()
            step_num += 1
            loss.backward()
            if ((step + 1) % self.batch_expand_times) == 0:
                self.optimizer.step()
                self.schedule.step()
                self.optimizer.zero_grad()
            if (step + 1) % 1000 == 0:
                self.eval(epoch, step, len(iterator_bar))
        self.optimizer.step()
        self.optimizer.zero_grad()
        avg_loss = loss_sum / step_num
        return avg_loss

    def eval(self, epoch, step=None, total_step=None):
        self.model.eval()
        with torch.no_grad():
            # iterator_bar = tqdm(self.dev_dataloader)
            iterator_bar = self.dev_dataloader
            correct_num = 0
            total = 0
            for batch in iterator_bar:
                for k in batch.keys():
                    batch[k] = batch[k].to(self.device)
                output = self.model(**batch)
                logits = output[0]
                _, preds = logits.max(dim=-1)
                correct = (batch['label'] == preds).sum().item()
                correct_num += correct
                total += len(preds)
        acc = correct_num / total

        if step is None:
            print("# EPOCH[{}] ACC[{:.5f}]".format(epoch, acc), end='')
        else:
            print("# EPOCH[{}][{}/{}] ACC[{:.5f}]".format(epoch, step, total_step, acc), end='')

        if acc > self.max_acc:
            self.max_acc = acc
            self.save_state_dict("epoch{}-acc{:.5f}.{}".format(epoch, acc, self.model_name))
            print("\t saved")
        else:
            print("")
        self.model.train()
        return acc

    def train(self):
        print("\nStart Training\n")
        for epoch in range(1, self.epoch + 1):
            avg_loss = self.train_epoch(epoch)
            acc = self.eval(epoch)
            print("# EPOCH[{}] AVG_LOSS[{:.5f}] ACC[{:.5f}]".format(epoch, avg_loss, self.max_acc))

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
