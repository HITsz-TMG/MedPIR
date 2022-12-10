import torch
from tqdm import tqdm
import torch.multiprocessing as mp
import json
import torch.nn.functional as F
from src.model import BERTGPT


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
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
    logits = logits[0]
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(tokenizer, model, history,
                    max_length=100, min_length=1, no_sample=True, temperature=1.0, top_k=5, top_p=1.0):
    SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[PAD]", "[speaker1]", "[speaker2]"]
    cls, sep, pad, spk1, spk2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    special_tokens_ids = [cls, sep, pad, spk1, spk2]

    past_generate = [cls]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i in range(max_length):
        if len(history) + len(past_generate) > 512:
            history = [history[0]] + history[2:]

        input_ids = torch.tensor([history + past_generate]).cuda()
        token_type_ids = [response_speaker] * input_ids.shape[1]
        token_type_ids = torch.tensor([token_type_ids], dtype=torch.long).cuda()

        logits, *_ = model(input_ids, token_type_ids=token_type_ids)
        logits = logits[0, -1, :] / temperature
        logits = top_filtering(logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if no_sample else torch.multinomial(probs, 1)
        if i < min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() == sep:
            break
        past_generate.append(prev.item())

    return past_generate[1:]


def generate(model, dataloader, tokenizer, save_path=None):
    SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[PAD]"]
    cls, sep, pad = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    # special_tokens_ids = [cls, sep, pad, spk1, spk2]
    model.eval()
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterator = tqdm(dataloader)
    pred_sentences_ids, target_sentences_ids = [], []

    with torch.no_grad():
        for batch in iterator:
            batch_pred_sentences, batch_target_sentences = [], []
            history_len = batch["history_ids"][0]
            cur = sample_sequence(tokenizer, model, batch['history'][0].tolist())
            batch_pred_sentences.append(cur)

            pred_sentences_ids.extend(batch_pred_sentences)

    return pred_sentences_ids


def generate_sentences(model, model_path, config, dataset, idx2token, rank, mp_list):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    model = BERTGPT(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    encoder, decoder = model.encoder, model.decoder
    encoder.eval()
    decoder.eval()

    if rank == 0:
        dataset = tqdm(dataset)
    for batch in dataset:  # batch size=1
        with torch.no_grad():

            history_ids = batch['history_ids']
            history_speaker = batch['history_speaker']
            mask = [1] * len(history_ids)

            _, past = encoder(torch.tensor([history_ids]), torch.tensor([mask]),
                              token_type_ids=torch.tensor([history_speaker]))

            sentence = []

            prev_pred = 101
            sentence.append(prev_pred)

            # decoding loop
            past_length = 0
            mask = torch.tensor([mask])
            for i in range(100):
                mask = F.pad(mask, [0, 1], "constant", 1.0)
                logits, past = decoder(torch.tensor([[prev_pred]]), mask, past=past, past_length=past_length)
                logits = logits.squeeze(1)
                logits = top_filtering(logits)
                probs = torch.softmax(logits, dim=-1)
                prev_pred = torch.multinomial(probs, num_samples=1)
                prev_pred = prev_pred.item()
                sentence.append(prev_pred)
                if prev_pred == 102:
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


def sample_generate(
        model_path="./BERTGPT_save_debug.pt",
        config=None,
        vocab_path="",
        process_num=1,
        result_save_path=""
):
    import pickle
    test_data = pickle.load(open(config['test_data_path'], 'rb'))

    length = len(test_data)
    idx2token = dict()
    with open(vocab_path, 'r', encoding='utf-8') as reader:
        for idx, token in enumerate(list(reader.readlines())):
            idx2token[idx] = token.strip()

    mgr = mp.Manager()
    mp_list = mgr.list()
    processes = []
    for rank in range(process_num):
        if rank == process_num - 1:
            data = test_data[int((rank / process_num) * length):]
        else:
            data = test_data[int((rank / process_num) * length): int(((rank + 1) / process_num) * length)]
        # (model_path, dataset, idx2token, rank, mp_list)

        p = mp.Process(target=generate_sentences, args=(model_path, config, data, idx2token, rank, mp_list))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    Dialog_list = []
    with open(result_save_path, 'w', encoding='utf-8') as f:
        for s in mp_list:
            cases = dict()
            cases['input'] = s[0]
            cases['predict'] = s[1]
            cases['reference'] = s[2]
            Dialog_list.append(cases)
        json.dump(Dialog_list, f, ensure_ascii=False, indent=4)
