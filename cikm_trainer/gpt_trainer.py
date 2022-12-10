from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from cikm_generate_utils.generator import Greedy
from cikm_trainer.hred_trainer import HREDTrainer
import torch


class GPTTrainer(HREDTrainer):
    def __init__(self, train_dataset, model, dev_dataset=None, test_dataset=None,
                 config=None, save_root="./cikm_save/GPT_save"):
        super(GPTTrainer, self).__init__(
            train_dataset, model, dev_dataset=dev_dataset, test_dataset=test_dataset,
            config=config, save_root=save_root
        )
        self.max_acc = -1.0

    def train_epoch(self, epoch):
        self.model.train()
        iterator_bar = tqdm(self.train_dataloader)
        loss_sum, acc_sum = 0.0, 0.0
        step_num = 0
        self.optimizer.zero_grad()
        for step, batch in enumerate(iterator_bar):
            for k in ["input_ids"]:
                batch[k] = batch[k].to(self.device)
            if batch["token_type_ids"] is not None:
                batch["token_type_ids"] = batch["token_type_ids"].to(self.device)

            # tmp = batch['input_ids'][0].tolist()
            # print("".join(self.train_dataset.convert_ids_to_tokens(tmp)))

            output = self.model(**batch)
            logits = output[0]
            loss, acc = self.calculate_loss_and_accuracy(logits, batch['input_ids'])
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
        acc_sum = 0.0
        step = 0
        for batch in iterator:
            for k in ["input_ids"]:
                batch[k] = batch[k].to(self.device)
            if batch["token_type_ids"] is not None:
                batch["token_type_ids"] = batch["token_type_ids"].to(self.device)

            output = self.model(**batch)
            logits = output[0]
            loss, acc = self.calculate_loss_and_accuracy(logits, batch['input_ids'])
            acc_sum += acc.item()
            step += 1
        acc = acc_sum / step
        print("acc {}   max_acc {}".format(acc, self.max_acc))
        if acc > self.max_acc:
            self.max_acc = acc
            self.save_state_dict("epoch{}-ACC[{:.5f}].pt".format(epoch, acc))
        self.model.train()

    def prepare_input_for_greedy_generate(self, batch, device, expand_batch_size=1):
        input_ids = batch['input_ids'].to(device)
        if batch['token_type_ids'] is not None:
            token_type_ids = batch['token_type_ids'].to(device)
        else:
            token_type_ids = None
        kv_inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
        }

        return kv_inputs
