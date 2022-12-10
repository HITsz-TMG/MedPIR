import json
import pickle
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy


class Generator:
    def __init__(self, config):
        self.config = dict(config)
        self.top_k = config.get("top_k", 0)
        self.top_p = config.get("top_p", 0)
        self.threshold = config.get("sampling_threshold", 0)
        self.beam_size = config.get("beam_size", 3)
        self.min_len = config.get("min_len", 3)
        self.max_len = config.get("max_len", 300)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.token2idx = dict()
        self.idx2token = dict()
        with open(config['vocab_path'], 'r', encoding='utf-8') as reader:
            for idx, token in enumerate(list(reader.readlines())):
                token = token.strip()
                self.token2idx[token] = idx
                self.idx2token[idx] = token
        self.unk_idx = self.token2idx['[UNK]']
        self.pad_idx = self.token2idx['[PAD]']
        self.sep_idx = self.token2idx['[SEP]']
        self.cls_idx = self.token2idx['[CLS]']

        self.ban_seq = ["好的", "可以", "嗯", "是", "好"]

    def check_ban_seq(self, ids):
        seq = "".join([self.idx2token[i] for i in ids])
        seq = seq.replace('，', '').replace('。', '')
        if seq in self.ban_seq:
            return False
        return True

    def top_filtering(self, logits, filter_value=-float('inf')):
        """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
                top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                    whose total probability mass is greater than or equal to the threshold top_p.
                    In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                    the threshold top_p.
                threshold: a minimal threshold to keep logits
                filter_value:
        """
        assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
        self.top_k = min(self.top_k, logits.size(-1))
        logits[self.unk_idx] = filter_value
        if self.top_k > 0:
            # Remove all tokens with a probability less than the last token in the top-k tokens
            indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if self.top_p > 0.0:
            # Compute cumulative probabilities of sorted tokens
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probabilities = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probabilities > self.top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Back to unsorted indices and set them to -infinity
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        indices_to_remove = logits < self.threshold
        logits[indices_to_remove] = filter_value

        return logits

    def generate_sentences_bertgpt_bak(self, model_class, model_path, dataset, idx2token, rank, mp_list, model=None):
        if model is None:
            model = model_class(self.config)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = model.to(self.device)
        model.eval()
        encoder, decoder = model.encoder, model.decoder
        encoder.eval()
        decoder.eval()

        if rank == 0:
            dataset = tqdm(dataset)
        cnt = 0
        for batch in dataset:  # batch size=1
            with torch.no_grad():
                history_ids = batch['history_ids']
                history_speaker = torch.tensor([batch['history_speaker']]).to(self.device) if self.config[
                    'use_token_type_ids'] else None
                mask = [1] * len(history_ids)
                _, past = encoder(torch.tensor([history_ids]).to(self.device), torch.tensor([mask]).to(self.device),
                                  token_type_ids=history_speaker)
                sentence = []
                prev_pred = self.cls_idx
                sentence.append(prev_pred)
                # decoding loop
                past_length = 0
                mask = torch.tensor([mask]).to(self.device)
                for i in range(self.max_len):
                    mask = F.pad(mask, [0, 1], "constant", 1.0)
                    logits, past = decoder(torch.tensor([[prev_pred]]).to(self.device), mask, past=past,
                                           past_length=past_length)
                    logits = logits[0][0]
                    if i < self.min_len:
                        logits[self.sep_idx] = float('-inf')
                    logits = self.top_filtering(logits)
                    probs = torch.softmax(logits, dim=-1)
                    prev_pred = torch.multinomial(probs, num_samples=1)
                    while prev_pred.item() in [self.unk_idx, self.pad_idx, self.cls_idx]:
                        prev_pred = torch.multinomial(probs, num_samples=1)
                    if i < self.min_len and prev_pred.item() == self.sep_idx:
                        while prev_pred.item() in [self.sep_idx, self.unk_idx, self.pad_idx, self.cls_idx]:
                            prev_pred = torch.multinomial(probs, num_samples=1)
                    prev_pred = prev_pred.item()
                    sentence.append(prev_pred)
                    if prev_pred == self.sep_idx:
                        break
                    past_length += 1
                predict = [idx2token[i] for i in sentence]
                if 'response_ids' in batch.keys():
                    target = batch['response_ids']
                    reference = [idx2token[i] for i in target]
                else:
                    reference = []

                history = [idx2token[i] for i in history_ids]
                mp_list.append(["".join(history), "".join(predict[1:-1]), "".join(reference)])
                cnt += 1
                print("------------------------------------------")
                print(cnt)
                print("".join(history))
                print("{}".format("".join(predict[1:-1])))
                print("------------------------------------------")
        return mp_list

    def generate_sentences_bertgpt(self, model_class, model_path, dataset, idx2token, rank, mp_list, model=None):
        if model is None:
            model = model_class(self.config)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = model.to(self.device)
        model.eval()
        encoder, decoder = model.encoder, model.decoder
        encoder.eval()
        decoder.eval()

        if rank == 0:
            dataset = tqdm(dataset)
        cnt = 0
        for batch in dataset:  # batch size=1
            with torch.no_grad():
                history_ids = batch['history_ids']
                history_speaker = torch.tensor([batch['history_speaker']]).to(self.device) if self.config[
                    'use_token_type_ids'] else None
                flag = True
                count = 0
                while flag:
                    count += 1
                    flag = False
                    mask = [1] * len(history_ids)
                    _, past = encoder(torch.tensor([history_ids]).to(self.device),
                                      torch.tensor([mask]).to(self.device),
                                      token_type_ids=history_speaker)
                    mask = torch.tensor([mask]).to(self.device)

                    sentence = []
                    prev_pred = self.cls_idx
                    sentence.append(prev_pred)
                    # decoding loop
                    past_length = 0
                    for i in range(self.max_len):
                        mask = F.pad(mask, [0, 1], "constant", 1.0)
                        logits, past = decoder(torch.tensor([[prev_pred]]).to(self.device), mask, past=past,
                                               past_length=past_length)
                        logits = logits[0][0]
                        if i < self.min_len:
                            logits[self.sep_idx] = float('-inf')
                        logits = self.top_filtering(logits)
                        probs = torch.softmax(logits, dim=-1)
                        prev_pred = torch.multinomial(probs, num_samples=1)
                        while prev_pred.item() in [self.unk_idx, self.pad_idx, self.cls_idx]:
                            prev_pred = torch.multinomial(probs, num_samples=1)
                        if i < self.min_len and prev_pred.item() == self.sep_idx:
                            while prev_pred.item() in [self.sep_idx, self.unk_idx, self.pad_idx, self.cls_idx]:
                                prev_pred = torch.multinomial(probs, num_samples=1)
                        prev_pred = prev_pred.item()
                        sentence.append(prev_pred)
                        if prev_pred == self.sep_idx:
                            break
                        past_length += 1
                    predict = [idx2token[i] for i in sentence]
                    pre_str = "".join(predict[1:-1])
                    if len(pre_str) < 8:
                        for ban in self.ban_seq:
                            if ban in pre_str:
                                flag = True
                                self.top_k += 1
                                break

                    if count > 10:
                        flag = False

                self.top_k = self.config.get("top_k", 0)
                if 'response_ids' in batch.keys():
                    target = batch['response_ids']
                    reference = [idx2token[i] for i in target]
                else:
                    reference = []

                history = [idx2token[i] for i in history_ids]
                mp_list.append(["".join(history), "".join(predict[1:-1]), "".join(reference)])
                cnt += 1
                if rank == 0:
                    print("------------------------------------------")
                    print(cnt)
                    print("".join(history))
                    print("{}".format("".join(predict[1:-1])))
                    print("------------------------------------------")
        return mp_list

    def generate_sentences(self, model_class, model_path, dataset, idx2token, rank, mp_list):
        model = model_class(self.config)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = model.to(self.device)
        encoder, decoder, lm_linear = model.encoder, model.decoder, model.lm_linear
        encoder.eval()
        decoder.eval()
        lm_linear.eval()

        # if rank == 0:
        #     dataset = tqdm(dataset)
        for batch in dataset:  # batch size=1
            with torch.no_grad():
                history_ids = batch['history_ids']
                history_speaker = batch['history_speaker']
                encoder_output = encoder(
                    input_ids=torch.tensor([history_ids]),
                    attention_mask=torch.ones((1, len(history_ids))),
                    token_type_ids=torch.tensor([history_speaker]) if self.config['use_token_type_ids'] else None
                )
                encoded_history = encoder_output[0]
                sentence = []
                prev_pred = 101
                sentence.append(prev_pred)
                # decoding loop
                for i in range(self.max_len):
                    decoder_output = decoder(
                        input_ids=torch.tensor([sentence]),
                        attention_mask=torch.ones((1, len(sentence))),
                        encoder_hidden_states=encoded_history,
                        encoder_attention_mask=torch.ones((1, len(history_ids)))
                    )
                    logits = lm_linear(decoder_output[0])[0][-1]
                    logits = self.top_filtering(logits)
                    probs = torch.softmax(logits, dim=-1)
                    prev_pred = torch.multinomial(probs, num_samples=1)
                    prev_pred = prev_pred.item()
                    sentence.append(prev_pred)
                    if prev_pred == 102:
                        break

                predict = [idx2token[i] for i in sentence]
                if 'response_ids' in batch.keys():
                    target = batch['response_ids']
                    reference = [idx2token[i] for i in target]
                else:
                    reference = []

                history = [idx2token[i] for i in history_ids]
                mp_list.append(["".join(history), "".join(predict[1:-1]), "".join(reference)])
                print("ref: {}".format(reference))
                print("prd: {}".format(predict))
        return mp_list

    def sample_generate(self, model_class=None, model_path="", process_num=1, result_save_path="result.json"):
        test_data = pickle.load(open(self.config['test_data_path'], 'rb'))
        # test_data = test_data[:10]
        mp_list = []

        if self.config['model_name'] == "BERTGPT":
            target_func = self.generate_sentences_bertgpt
        elif self.config['model_name'] == "BERTGPTEntity":
            target_func = self.generate_sentences_entity_attention
        elif self.config['model_name'] == "BERT2BERTEntity":
            target_func = self.generate_sentences_bert2bert_entity
        else:
            target_func = None

        mp_list = target_func(model_class, model_path, test_data, self.idx2token, 0, mp_list)

        Dialog_list = []
        with open(result_save_path, 'w', encoding='utf-8') as f:
            for s in mp_list:
                cases = dict()
                cases['input'] = s[0]
                cases['predict'] = s[1]
                cases['reference'] = s[2]
                Dialog_list.append(cases)
            json.dump(Dialog_list, f, ensure_ascii=False, indent=4)

        predict = [i[1] for i in mp_list]
        pickle.dump(predict, open("result.pk", 'wb'))
        return predict

    def generate_sentences_entity_attention(self, model_class, model_path, dataset, idx2token, rank, mp_list,
                                            model=None):
        if model is None:
            model = model_class(self.config)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = model.to(self.device)
        model.eval()
        encoder, decoder = model.encoder, model.decoder
        encoder.eval()
        decoder.eval()
        entity_attention = model.entity_attention
        entity_attention.eval()
        lm_linear = model.lm_linear
        lm_linear.eval()

        if rank == 0:
            dataset = tqdm(dataset)
        cnt = 0
        for batch in dataset:  # batch size=1
            with torch.no_grad():
                history_ids = batch['history_ids']
                history_speaker = torch.tensor([batch['history_speaker']]).to(self.device) if self.config[
                    'use_token_type_ids'] else None
                flag = True
                count = 0
                while flag:
                    count += 1
                    flag = False
                    mask = [1] * len(history_ids)
                    encoder_output = encoder(torch.tensor([history_ids]).to(self.device),
                                             torch.tensor([mask]).to(self.device),
                                             token_type_ids=history_speaker, use_cache=True)

                    encoded_history = encoder_output.last_hidden_state
                    past_key_values = encoder_output.past_key_values
                    history_cls = encoded_history[:, 0].unsqueeze(dim=1)
                    attention_score, entity_info = entity_attention(history_cls)
                    mask = torch.tensor([mask]).to(self.device)
                    sentence = []
                    prev_pred = self.cls_idx
                    sentence.append(prev_pred)
                    # decoding loop
                    for i in range(self.max_len):
                        mask = F.pad(mask, [0, 1], "constant", 1.0)
                        decoder_output = decoder(
                            torch.tensor([sentence]).to(self.device),
                            mask,
                            past_key_values=past_key_values,
                            past_key_values_len=0,
                            entity_info=entity_info,
                        )
                        hidden_state = decoder_output.last_hidden_state
                        logits = lm_linear(hidden_state)
                        logits = logits[0][-1]
                        if i < self.min_len:
                            logits[self.sep_idx] = float('-inf')
                        logits = self.top_filtering(logits)
                        probs = torch.softmax(logits, dim=-1)
                        prev_pred = torch.multinomial(probs, num_samples=1)
                        while prev_pred.item() in [self.unk_idx, self.pad_idx, self.cls_idx]:
                            prev_pred = torch.multinomial(probs, num_samples=1)
                        if i < self.min_len and prev_pred.item() == self.sep_idx:
                            while prev_pred.item() in [self.sep_idx, self.unk_idx, self.pad_idx, self.cls_idx]:
                                prev_pred = torch.multinomial(probs, num_samples=1)
                        prev_pred = prev_pred.item()
                        sentence.append(prev_pred)
                        if prev_pred == self.sep_idx:
                            break
                    predict = [idx2token[i] for i in sentence]
                    pre_str = "".join(predict[1:-1])
                    if len(pre_str) < 8:
                        for ban in self.ban_seq:
                            if ban in pre_str:
                                flag = True
                                self.top_k += 1
                                break

                    if count > 10:
                        flag = False

                self.top_k = self.config.get("top_k", 0)
                if 'response_ids' in batch.keys():
                    target = batch['response_ids']
                    reference = [idx2token[i] for i in target]
                else:
                    reference = []

                history = [idx2token[i] for i in history_ids]
                mp_list.append(["".join(history), "".join(predict[1:-1]), "".join(reference)])
                cnt += 1
                print("------------------------------------------")
                print(cnt)
                print("".join(history))
                print("{}".format("".join(predict[1:-1])))
                print("------------------------------------------")
        return mp_list

    def generate_sentences_bert2bert_entity(self, model_class, model_path, dataset, idx2token, rank, mp_list,
                                            model=None):
        if model is None:
            model = model_class(self.config)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = model.to(self.device)
        model.eval()

        if rank == 0:
            dataset = tqdm(dataset)
        cnt = 0
        for batch in dataset:  # batch size=1
            with torch.no_grad():
                history_ids = batch['history_ids']
                history_speaker = torch.tensor([batch['history_speaker']]).to(self.device) if self.config[
                    'use_token_type_ids'] else None
                flag = True
                count = 0
                while flag:
                    count += 1
                    flag = False
                    encode_step_outputs = model.encode_step(
                        history_ids=torch.tensor([history_ids]).to(self.device),
                        history_speaker=history_speaker,

                    )
                    encoded_history = encode_step_outputs[0]
                    history_mask = encode_step_outputs[1]
                    entity_info_before_cls = encode_step_outputs[2]

                    sentence = []
                    prev_pred = self.cls_idx
                    sentence.append(prev_pred)
                    # decoding loop
                    for i in range(self.max_len):
                        # response_mask = torch.tensor([[1] * len(sentence)]).to(self.device)

                        decode_step_outputs = model.decode_step(
                            encoded_history=encoded_history,
                            history_mask=history_mask,
                            response_ids=torch.tensor([sentence]).to(self.device),
                            # response_mask=response_mask,
                            entity_info_before_cls=entity_info_before_cls
                        )
                        logits = decode_step_outputs[0][-1]
                        if i < self.min_len:
                            logits[self.sep_idx] = float('-inf')
                        logits = self.top_filtering(logits)
                        probs = torch.softmax(logits, dim=-1)
                        prev_pred = torch.multinomial(probs, num_samples=1)
                        while prev_pred.item() in [self.unk_idx, self.pad_idx, self.cls_idx]:
                            prev_pred = torch.multinomial(probs, num_samples=1)
                        if i < self.min_len and prev_pred.item() == self.sep_idx:
                            while prev_pred.item() in [self.sep_idx, self.unk_idx, self.pad_idx, self.cls_idx]:
                                prev_pred = torch.multinomial(probs, num_samples=1)
                        prev_pred = prev_pred.item()
                        sentence.append(prev_pred)
                        if prev_pred == self.sep_idx:
                            break
                    predict = [idx2token[i] for i in sentence]
                    pre_str = "".join(predict[1:-1])
                    if len(pre_str) < 4:
                        for ban in self.ban_seq:
                            if ban in pre_str:
                                flag = True
                                self.top_k += 1
                                break

                    if count > 5:
                        flag = False

                self.top_k = self.config.get("top_k", 0)
                if 'response_ids' in batch.keys():
                    target = batch['response_ids']
                    reference = [idx2token[i] for i in target]
                else:
                    reference = []

                history = [idx2token[i] for i in history_ids]
                mp_list.append(["".join(history), "".join(predict[1:-1]), "".join(reference)])
                cnt += 1
                if cnt <= 20000:
                    print("------------------------------------------")
                    print(cnt)
                    print("".join(history))
                    print("{}".format("".join(predict[1:-1])))
                    print("------------------------------------------")
        return mp_list

    def multi_process_sample_generate(self, process_num, model, test_data):
        length = len(test_data)
        mgr = mp.Manager()
        mp_list = mgr.list()
        processes = []
        if self.config['model_name'] == "BERTGPT":
            target_func = self.generate_sentences_bertgpt
        elif self.config['model_name'] == "BERTGPTEntity":
            target_func = self.generate_sentences_entity_attention
        elif self.config['model_name'] == "BERT2BERTEntity":
            target_func = self.generate_sentences_bert2bert_entity
        else:
            target_func = None

        if process_num == 1:
            normal_list = list()
            mp_list = target_func(self.config['model_class'], None, test_data, self.idx2token, 0, normal_list, model)
        else:
            for rank in range(process_num):
                if rank == process_num - 1:
                    data = test_data[int((rank / process_num) * length):]
                else:
                    data = test_data[int((rank / process_num) * length): int(((rank + 1) / process_num) * length)]
                p = mp.Process(
                    target=target_func,
                    args=(self.config['model_class'], None, data, self.idx2token, rank, mp_list, model)
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

        return mp_list

    def beam_search_generate(self, model_class=None, model_path=""):
        test_data = pickle.load(open(self.config['test_data_path'], 'rb'))
        model = model_class(self.config)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = model.to(self.device)
        if self.config['model_name'] == "BERTGPT":
            lm_linear = False
        else:
            lm_linear = True

        cnt = 0
        for item in test_data:
            history_speaker = item['history_speaker'] if self.config['use_token_type_ids'] else None
            if self.config['model_name'] == "BERTGPT":
                complete_seqs, complete_seqs_scores = self.beam_search_bertgpt(
                    model, self.beam_size, item['history_ids'], history_speaker, len(self.idx2token),
                    lm_linear=lm_linear
                )
            else:
                complete_seqs, complete_seqs_scores = self.beam_search(
                    model, self.beam_size, item['history_ids'], history_speaker, len(self.idx2token),
                    lm_linear=lm_linear
                )
            best_idx = complete_seqs_scores.index(max(complete_seqs_scores))
            best_reply = complete_seqs[best_idx]
            cnt += 1
            print("-------------------------")
            print("{}".format("".join([self.idx2token[_] for _ in item['history_ids']])))
            print("{} ".format(cnt) + "".join([self.idx2token[_] for _ in best_reply[1:-1]]))
            print("-------------------------")

    def beam_search(self, model, beam_size, history, history_speaker, vocab_size, lm_linear=True):
        k = beam_size

        history = torch.tensor(history).repeat(k, 1).to(self.device)
        history_mask = torch.ones_like(history)
        if history_speaker is not None:
            history_speaker = history_speaker.repeat(k, 1).to(self.device)

        if self.config['model_name'] == "BERTGPT":
            encoder_outputs = model.encoder(
                input_ids=history,
                mask=history_mask
            )
        else:
            encoder_outputs = model.encoder(
                input_ids=history,
                attention_mask=history_mask,
                token_type_ids=history_speaker
            )

        encoder_hidden_states = encoder_outputs[0]

        # store top k previous words at each step
        # (k, 1)
        # k_prev_words = torch.tensor([model.cls_idx] * k).to(device)

        # store top k sequences, cls for init
        seqs = torch.tensor([[self.cls_idx]] * k).to(self.device)

        # store top k sequences' scores
        # (k, 1)
        top_k_scores = torch.zeros(k, 1).to(self.device)

        # store completed sequences
        complete_seqs = list()
        complete_seqs_scores = list()

        step = 1

        while True:
            decoder_outputs = model.decoder(
                input_ids=seqs,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=history_mask
            )

            decoder_hidden_states = decoder_outputs[0]
            if len(decoder_hidden_states.shape) == 2:
                decoder_hidden_states = decoder_hidden_states.unsqueeze(0)

            if lm_linear:
                vocab_logits = model.lm_linear(decoder_hidden_states[:, -1, :])
            else:
                vocab_logits = decoder_hidden_states[:, -1, :]

            # vocab_logits = top_filtering(vocab_logits, top_k=top_k)

            # scores: (s, vocab_size)
            scores = F.log_softmax(vocab_logits, dim=1)

            # top_k_scores[i] + scores[0],top_k_scores[i] + scores[1], ... top_k_scores[i] + scores[vocab_size1-1]
            scores = top_k_scores.expand_as(scores) + scores

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0)
                # sample_top = torch.multinomial(scores[0], k)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0)

            # top_k_words: (s)
            pre_word_index = top_k_words // vocab_size
            next_word_index = top_k_words % vocab_size

            # add new words to sequences
            seqs = torch.cat([seqs[pre_word_index], next_word_index.unsqueeze(1)], dim=1)

            incomplete_index = [i for i, next_word in enumerate(next_word_index) if next_word != self.sep_idx]

            complete_index = list(set(range(len(next_word_index.tolist()))) - set(incomplete_index))

            # set aside complete sequences
            if len(complete_index) > 0:
                complete_seqs.extend(seqs[complete_index].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_index].tolist())

            k -= len(complete_index)

            if k == 0:
                break

            # too long
            if step > self.max_len:
                complete_seqs.extend(seqs[incomplete_index].tolist())
                complete_seqs_scores.extend(top_k_scores[incomplete_index].tolist())
                break
            step += 1

            # incomplete sequences
            seqs = seqs[incomplete_index]

            encoder_hidden_states = encoder_hidden_states[pre_word_index[incomplete_index]]
            history_mask = history_mask[pre_word_index[incomplete_index]]
            # k_prev_words: (s, 1)
            # k_prev_words = next_word_index[incomplete_index]
            top_k_scores = top_k_scores[incomplete_index].unsqueeze(1)
        return complete_seqs, complete_seqs_scores

    def beam_search_bertgpt(self, model, beam_size, history, history_speaker, vocab_size, lm_linear=False):
        k = beam_size
        bak_history = torch.tensor(history).to(self.device)
        history = torch.tensor(history).repeat(k, 1).to(self.device)
        mask = torch.ones_like(history)
        if history_speaker is not None:
            history_speaker = history_speaker.repeat(k, 1).to(self.device)

        _, past = model.encoder(
            input_ids=history,
            mask=mask
        )
        # store top k previous words at each step
        # (k, 1)
        # k_prev_words = torch.tensor([model.cls_idx] * k).to(device)

        # store top k sequences, cls for init
        k_prev_words = torch.tensor([self.cls_idx] * k).to(self.device)
        seqs = torch.tensor([[self.cls_idx]] * k).to(self.device)
        # store top k sequences' scores
        # (k, 1)
        top_k_scores = torch.zeros(k, 1).to(self.device)

        # store completed sequences
        complete_seqs = list()
        complete_seqs_scores = list()

        step = 1

        while True:
            mask = F.pad(mask, [0, 1], "constant", 1.0)
            decoder_outputs = model.decoder(
                input_ids=seqs,
                mask=mask,
                past=past,
                past_length=0
            )

            decoder_hidden_states = decoder_outputs[0]
            if len(decoder_hidden_states.shape) == 2:
                decoder_hidden_states = decoder_hidden_states.unsqueeze(0)
            vocab_logits = decoder_hidden_states[:, -1, :]

            # vocab_logits = top_filtering(vocab_logits, top_k=top_k)

            # scores: (s, vocab_size)
            scores = F.log_softmax(vocab_logits, dim=1)

            # top_k_scores[i] + scores[0],top_k_scores[i] + scores[1], ... top_k_scores[i] + scores[vocab_size1-1]
            scores = top_k_scores.expand_as(scores) + scores

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0)
                # sample_top = torch.multinomial(scores[0], k)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0)

            # top_k_words: (s)
            pre_word_index = top_k_words // vocab_size
            next_word_index = top_k_words % vocab_size

            # add new words to sequences
            seqs = torch.cat([seqs[pre_word_index], next_word_index.unsqueeze(1)], dim=1)

            incomplete_index = [i for i, next_word in enumerate(next_word_index) if next_word != self.sep_idx]

            complete_index = list(set(range(len(next_word_index.tolist()))) - set(incomplete_index))

            # set aside complete sequences
            if len(complete_index) > 0:
                complete_seqs.extend(seqs[complete_index].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_index].tolist())

            k -= len(complete_index)

            if k == 0:
                break

            # too long
            if step > self.max_len:
                complete_seqs.extend(seqs[incomplete_index].tolist())
                complete_seqs_scores.extend(top_k_scores[incomplete_index].tolist())
                break
            step += 1

            # incomplete sequences
            seqs = seqs[incomplete_index]

            mask = mask[pre_word_index[incomplete_index]]
            # for layer_idx in range(len(past)):
            #     past[layer_idx] = past[layer_idx][pre_word_index[incomplete_index]]

            if len(complete_index) > 0:
                history = torch.tensor(bak_history).repeat(len(seqs), 1)
                _, past = model.encoder(
                    input_ids=history,
                    mask=torch.ones_like(history).to(self.device)
                )

            # k_prev_words = next_word_index[incomplete_index]
            top_k_scores = top_k_scores[incomplete_index].unsqueeze(1)

            # for s in seqs:
            #     print("".join([self.idx2token[_] for _ in s.tolist()]))
        return complete_seqs, complete_seqs_scores

    def bert2bert_entity(self, model_class, model_path, iterator, idx2token, model=None):
        if model is None:
            model = model_class(self.config)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = model.to(self.device)
        model.eval()
        cnt = 0

        result_list = []

        for batch in iterator:  # batch size=1
            with torch.no_grad():

                if not self.config.get('slide_window', False):
                    history_ids = batch['history_ids'].to(self.device)
                    history_speaker = batch['history_speaker'].to(self.device) if self.config[
                        'use_token_type_ids'] else None
                    history_mask = batch['history_mask'].to(self.device)
                    group_history_ids = None
                    group_history_mask = None
                    start_end_expand_as_batch_size = None
                    each_sample_chunks_num = None
                else:
                    history_ids = batch['history_ids']
                    history_speaker = None
                    group_history_ids = batch['group_history_ids'],
                    group_history_mask = batch['group_history_mask'],
                    for idx in range(len(batch['group_history_ids'])):
                        batch['group_history_ids'][idx] = batch['group_history_ids'][idx].to(self.device)
                        batch['group_history_mask'][idx] = batch['group_history_mask'][idx].to(self.device)
                    start_end_expand_as_batch_size = batch['start_end_expand_as_batch_size'],
                    each_sample_chunks_num = batch['each_sample_chunks_num'],
                    history_mask = None
                    group_history_ids = group_history_ids[0]
                    group_history_mask = group_history_mask[0]
                    start_end_expand_as_batch_size = start_end_expand_as_batch_size[0]
                    each_sample_chunks_num = each_sample_chunks_num[0]

                flag = True
                count = 0
                while flag:
                    count += 1
                    flag = False
                    encode_step_outputs = model.encode_step(
                        history_ids=history_ids,
                        history_speaker=history_speaker,

                        group_history_ids=group_history_ids,
                        group_history_mask=group_history_mask,
                        start_end_expand_as_batch_size=start_end_expand_as_batch_size,
                        each_sample_chunks_num=each_sample_chunks_num,
                    )
                    encoded_history = encode_step_outputs[0]
                    history_mask = encode_step_outputs[1]
                    entity_info_before_cls = encode_step_outputs[2]

                    sentence = []
                    prev_pred = self.cls_idx
                    sentence.append(prev_pred)

                    for i in range(self.max_len):
                        decode_step_outputs = model.decode_step(
                            encoded_history=encoded_history,
                            history_mask=history_mask,
                            response_ids=torch.tensor([sentence]).to(self.device),

                            entity_info_before_cls=entity_info_before_cls
                        )
                        logits = decode_step_outputs[0][-1]

                        if i < self.min_len:
                            logits[self.sep_idx] = float('-inf')

                        logits = self.top_filtering(logits)
                        probs = torch.softmax(logits, dim=-1)
                        prev_pred = torch.multinomial(probs, num_samples=1)

                        while prev_pred.item() in [self.unk_idx, self.pad_idx, self.cls_idx]:
                            prev_pred = torch.multinomial(probs, num_samples=1)

                        prev_pred = prev_pred.item()
                        sentence.append(prev_pred)
                        if prev_pred == self.sep_idx:
                            break
                    predict = [idx2token[i] for i in sentence]
                    pre_str = "".join(predict[1:-1])
                    if len(pre_str) < 4:
                        for ban in self.ban_seq:
                            if ban in pre_str:
                                flag = True
                                self.top_k += 1
                                break
                    if count > 2:
                        flag = False

                self.top_k = self.config.get("top_k", 0)
                if 'response_ids' in batch.keys():
                    target = batch['response_ids'][0].tolist()
                    reference = [idx2token[i] for i in target]
                else:
                    reference = []

                history = [idx2token[i] for i in history_ids[0].tolist()]
                result_list.append(["".join(history), "".join(predict[1:-1]), "".join(reference)])
                cnt += 1
                if cnt <= 20:
                    print("------------------------------------------")
                    print(cnt)
                    print("".join(history))
                    if reference:
                        print("target: {}".format("".join(reference)))
                    print("predict: {}".format("".join(predict[1:-1])))
                    print("------------------------------------------")
        return result_list

    def bertgpt_entity(self, model_class, model_path, iterator, idx2token, model=None):
        if model is None:
            model = model_class(self.config)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = model.to(self.device)
        model.eval()
        cnt = 0
        result_list = []
        for batch in iterator:  # batch size=1
            with torch.no_grad():

                if not self.config.get('slide_window', False):
                    history_ids = batch['history_ids'].to(self.device)
                    history_speaker = batch['history_speaker'].to(self.device) if self.config[
                        'use_token_type_ids'] else None
                    history_mask = batch['history_mask'].to(self.device)
                    group_history_ids = None
                    group_history_mask = None
                    start_end_expand_as_batch_size = None
                    each_sample_chunks_num = None
                else:
                    history_ids = batch['history_ids']
                    history_speaker = None
                    group_history_ids = batch['group_history_ids'],
                    group_history_mask = batch['group_history_mask'],
                    for idx in range(len(batch['group_history_ids'])):
                        batch['group_history_ids'][idx] = batch['group_history_ids'][idx].to(self.device)
                        batch['group_history_mask'][idx] = batch['group_history_mask'][idx].to(self.device)
                    start_end_expand_as_batch_size = batch['start_end_expand_as_batch_size'],
                    each_sample_chunks_num = batch['each_sample_chunks_num'],
                    history_mask = None
                    group_history_ids = group_history_ids[0]
                    group_history_mask = group_history_mask[0]
                    start_end_expand_as_batch_size = start_end_expand_as_batch_size[0]
                    each_sample_chunks_num = each_sample_chunks_num[0]

                flag = True
                count = 0
                while flag:
                    count += 1
                    flag = False

                    encode_step_outputs = model.encode_step(
                        history_ids=history_ids,
                        token_type_ids=history_speaker,
                        history_mask=history_mask,

                        group_history_ids=group_history_ids,
                        group_history_mask=group_history_mask,
                        start_end_expand_as_batch_size=start_end_expand_as_batch_size,
                        each_sample_chunks_num=each_sample_chunks_num,
                    )

                    encoded_history, past_key_values, history_mask, entity_loss = encode_step_outputs

                    sentence = []
                    prev_pred = self.cls_idx
                    sentence.append(prev_pred)
                    # decoding loop
                    for i in range(self.max_len):
                        cur_input = torch.tensor([sentence]).to(self.device)
                        decode_step_outputs = model.decode_step(
                            response_ids=cur_input,
                            response_mask=torch.ones_like(cur_input).to(self.device),
                            history_mask=history_mask,
                            past_key_values=past_key_values,
                            past_key_values_len=0
                        )
                        logits = decode_step_outputs[0][-1]

                        if i < self.min_len:
                            logits[self.sep_idx] = float('-inf')

                        logits = self.top_filtering(logits)
                        probs = torch.softmax(logits, dim=-1)
                        prev_pred = torch.multinomial(probs, num_samples=1)

                        while prev_pred.item() in [self.unk_idx, self.pad_idx, self.cls_idx]:
                            prev_pred = torch.multinomial(probs, num_samples=1)

                        prev_pred = prev_pred.item()
                        sentence.append(prev_pred)
                        if prev_pred == self.sep_idx:
                            break
                    predict = [idx2token[i] for i in sentence]
                    pre_str = "".join(predict[1:-1])
                    if len(pre_str) < 4:
                        for ban in self.ban_seq:
                            if ban in pre_str:
                                flag = True
                                self.top_k += 1
                                break

                    if count > 2:
                        flag = False

                self.top_k = self.config.get("top_k", 0)
                if 'response_ids' in batch.keys():
                    target = batch['response_ids'][0].tolist()
                    reference = [idx2token[i] for i in target]
                else:
                    reference = []

                history = [idx2token[i] for i in history_ids[0].tolist()]
                result_list.append(["".join(history), "".join(predict[1:-1]), "".join(reference)])
                cnt += 1
                if cnt <= 20:
                    print("------------------------------------------")
                    print(cnt)
                    print("".join(history))
                    if reference:
                        print("target: {}".format("".join(reference[1:-1])))
                    print("predict: {}".format("".join(predict[1:-1])))
                    print("------------------------------------------")
        return result_list
