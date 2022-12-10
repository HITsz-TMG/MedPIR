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


class ReorganizeTrainer:
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

        self.save_root = save_root

        self.train_dataloader = train_dataset.get_dataloader(shuffle=True, batch_size=self.batch_size, num_workers=1)
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
        # reduction = "mean" if config['entity_kl'] is False else "sum"
        reduction = "mean"
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_idx, reduction=reduction)
        self.gt = Generator(config=self.config)

        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')

    def prepare_for_input(self, batch):
        """
            # 历史不一定是历史 历史有可能在input for crossattention
            model:
            history_ids=None,
            history_mask=None,
            history_speaker=None,
            response_mask=None,
            response_ids=None,
            # reorganize
            input_for_crossattention=None,
            crossattention_mask=None,

            batch:
            "history_ids": history_ids,
            "history_mask": history_mask,
            "references_with_entities": combined_references_with_entities,
            "references_mask": references_mask,
            "token_type_ids": token_type_ids,
        """
        inputs = {
            "history_ids": batch['references_with_entities'].to(self.device),
            "history_mask": batch['references_mask'].to(self.device),
            "history_speaker": batch['token_type_ids'].to(self.device),
            "response_mask": batch['response_mask'].to(self.device),
            "response_ids": batch['response_ids'].to(self.device),
            "input_for_crossattention": batch['history_ids'].to(self.device),
            "crossattention_mask": batch['history_mask'].to(self.device)
        }
        return inputs

    def prepare_for_kl_target_input(self, batch):
        """
            "kl_target_token_type_ids": kl_target_token_type_ids,
            "kl_target_combined_references_with_entities": kl_target_combined_references_with_entities,
            "kl_target_references_mask": kl_target_references_mask
        """
        inputs = {
            "history_ids": batch['kl_target_combined_references_with_entities'].to(self.device),
            "history_mask": batch['kl_target_references_mask'].to(self.device),
            "history_speaker": batch['kl_target_token_type_ids'].to(self.device),
            "response_mask": batch['response_mask'].to(self.device),
            "response_ids": batch['response_ids'].to(self.device),
            "input_for_crossattention": batch['history_ids'].to(self.device),
            "crossattention_mask": batch['history_mask'].to(self.device)
        }
        return inputs

    def get_kl_loss(self, norm_logits, kl_logits, response_mask):
        logits_mask = response_mask.unsqueeze(dim=-1)
        norm_logits = logits_mask * norm_logits
        kl_logits = logits_mask * kl_logits
        norm_log_prob_dis = torch.log_softmax(norm_logits, dim=-1)
        kl_prob_dis = torch.softmax(kl_logits, dim=-1)
        return self.kl_div_loss(norm_log_prob_dis, kl_prob_dis)

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
            loss, acc = self.calculate_loss_and_accuracy(logits, batch['response_ids'])

            if self.config['entity_kl']:
                kl_target_model_input = self.prepare_for_kl_target_input(batch)
                kl_output = self.model(**kl_target_model_input)
                kl_logits = kl_output[0]
                kl_loss = self.get_kl_loss(logits, kl_logits, batch['response_mask'].to(self.device))
            else:
                kl_loss = None

            if kl_loss is None:
                iterator_bar.set_description("EPOCH[{}] LOSS[{:.5f}] ACC[{:.5f}]".format(
                    epoch, loss.item(), acc.item()))
            else:
                iterator_bar.set_description("EPOCH[{}] LOSS[{:.5f}] KL[{:.5f}] ACC[{:.5f}]".format(
                    epoch, loss.item(), kl_loss.item(), acc.item()))

            loss_sum += loss.item()
            acc_sum += acc.item()
            step_num += 1

            if kl_loss is None:
                loss /= self.config['batch_expand_times']
                loss.backward()
            else:
                loss = loss + 0.001 * kl_loss
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
        kv_inputs = {
            "history_ids": batch['references_with_entities'].to(self.device),
            "history_mask": batch['references_mask'].to(self.device),
            "token_type_ids": batch['token_type_ids'].to(self.device),
            # "response_mask": batch['response_mask'].to(self.device),
            # "response_ids": batch['response_ids'].to(self.device),
            "input_for_crossattention": batch['history_ids'].to(self.device),
            "crossattention_mask": batch['history_mask'].to(self.device)
        }
        return kv_inputs
