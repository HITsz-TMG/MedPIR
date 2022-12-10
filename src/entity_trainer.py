import torch
from src.entity_predict import EntityPredict
from src.model import NextEntityPredict
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import Adam
from torch import nn
import time
import os


class EntityPredTrainer:

    def __init__(self, train_dataset, model, dev_dataset=None, test_dataset=None,
                 config=None, save_root="./save/EntityPred_save"):
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
        self.dev_dataloader = dev_dataset.get_dataloader(shuffle=False, batch_size=50)

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
        self.max_F1 = float('-inf')

        self.pad_idx = self.train_dataset.pad_idx
        self.loss_fct = nn.BCELoss(reduction='mean')
        self.entity_label_keys = ["label{}".format(e_idx) for e_idx in range(160)]

        # self.summary = SummaryWriter('./summary/result')

    def next_entity_predict_inputs(self, batch):
        # input_ids = None, attention_mask = None, token_type_ids = None, label = None

        history_ids = batch['history_ids'].to(self.device)
        history_mask = batch['history_mask'].to(self.device)
        token_type_ids = batch['history_speaker'].to(self.device)
        entity_label = batch['entity_label'].to(self.device)

        return {
            "input_ids": history_ids,
            "attention_mask": history_mask,
            "token_type_ids": token_type_ids,
            "label": entity_label
        }

    def train_epoch(self, epoch):
        self.model.train()
        iterator_bar = tqdm(self.train_dataloader)
        loss_sum = 0.0
        step_num = 0
        self.optimizer.zero_grad()
        for step, batch in enumerate(iterator_bar):

            if self.config['model_class'] == NextEntityPredict:
                output = self.model(**self.next_entity_predict_inputs(batch))
            else:
                for k in batch.keys():
                    batch[k] = batch[k].to(self.device)
                output = self.model(**batch)
            # topic_probs, five_topic_probs, loss,
            loss = output[2]
            probs = output[0]
            preds = (probs > 0.5).long()
            if "entity_label" in batch.keys():
                correct = (batch["entity_label"].to(self.device) == preds).sum().item()
            else:
                correct = (batch["label"].to(self.device) == preds).sum().item()

            acc = correct / len(preds.view(-1))
            iterator_bar.set_description(
                "EPOCH[{}] LOSS[{:.5f}] ACC[{:.5f}]".format(epoch, loss.item(), acc))
            loss_sum += loss.item()
            step_num += 1
            loss.backward()
            if ((step + 1) % self.batch_expand_times) == 0:
                self.optimizer.step()
                self.schedule.step()
                self.optimizer.zero_grad()
            if epoch >= 3:
                if (step + 1) % 1000 == 0:
                    self.eval(epoch, step, len(iterator_bar))
        self.optimizer.step()
        self.optimizer.zero_grad()
        avg_loss = loss_sum / step_num
        return avg_loss

    def eval(self, epoch, step=None, total_step=None):
        self.model.eval()
        with torch.no_grad():
            iterator_bar = tqdm(self.dev_dataloader)
            # iterator_bar = self.dev_dataloader
            correct_num = 0
            total = 0

            positive = 0
            correct_positive = 0
            pred_positive = 0
            pred_negative = 0

            threshold = 0.5
            # print("threshold: {}".format(threshold))

            for batch in iterator_bar:
                if self.config['model_class'] == NextEntityPredict:
                    output = self.model(**self.next_entity_predict_inputs(batch))
                    batch['label'] = batch['entity_label'].to(self.device)
                else:
                    for k in batch.keys():
                        batch[k] = batch[k].to(self.device)
                    output = self.model(**batch)
                # topic_probs, five_topic_probs, loss,

                topic_probs = output[0]
                pred = (topic_probs > threshold).long()
                pred_positive += pred.sum().item()
                pred_negative += (pred <= 0.5).long().sum().item()

                correct = (batch["label"] == pred).sum().item()
                correct_positive += (pred * batch["label"]).sum().item()
                real_positive = batch["label"].sum().item()
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

        if epoch > 0:
            if F1 > self.max_F1:
                self.max_F1 = F1
                self.save_state_dict("epoch{}-F1{:.5f}.pt".format(epoch, F1))
                print("\t saved")
            else:
                print("")
        self.model.train()
        return F1

    def train(self):
        # self.eval(-1, -1, -1)
        print("\nStart Training\n")
        for epoch in range(1, self.epoch + 1):
            avg_loss = self.train_epoch(epoch)
            acc = self.eval(epoch)
            print("# EPOCH[{}] AVG_LOSS[{:.5f}]".format(epoch, avg_loss))

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
