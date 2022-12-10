from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import torch
from torch.nn import functional as F
from transformers.generation_logits_process import LogitsWarper
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)
from tqdm import tqdm


class BanResponseWarper(LogitsWarper):
    def __init__(self, response_special_id: int = 99):
        self.response_special_id = response_special_id

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        for idx, item in enumerate(input_ids):
            if self.response_special_id in item:
                scores[idx][self.response_special_id] = float("-inf")
        return scores


class BeamSample:
    def __init__(self, config):
        self.config = dict(config)
        self.top_k = config.get("top_k", 0)
        self.top_p = config.get("top_p", 0)
        self.threshold = config.get("sampling_threshold", 0)
        self.beam_size = config.get("beam_size", 3)
        self.repetition_penalty = config.get("repetition_penalty", 1.0)
        self.length_penalty = config.get("length_penalty", 0)
        self.no_repeat_ngram_size = config.get("no_repeat_ngram_size", 0)
        self.encoder_no_repeat_ngram_size = config.get('encoder_no_repeat_ngram_size', 6)
        self.temperature = 1.0
        self.bad_words_ids = [[]]
        self.do_sample = True
        self.is_encoder_decoder = False
        self.min_len = config.get("min_len", 3)
        self.max_len = config.get("max_len", 120)
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

    def _get_logits_warper(
            self, top_k: int = None, top_p: float = None, temperature: float = None, num_beams: int = None,
            response_special_id: int = None) -> LogitsProcessorList:
        """
        This class returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
        :obj:`~transformers.LogitsWarper` instances used for multinomial sampling.
        """

        # init warp parameters
        top_k = top_k if top_k is not None else self.top_k
        top_p = top_p if top_p is not None else self.top_p
        temperature = temperature if temperature is not None else self.temperature
        # instantiate warpers list
        warpers = LogitsProcessorList()

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if response_special_id is not None:
            warpers.append(BanResponseWarper(response_special_id))
        if top_k is not None and top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
        if top_p is not None and top_p < 1.0 and top_p != 0:
            warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
        if temperature is not None and temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        return warpers

    def _get_logits_processor(
            self,
            repetition_penalty: float,
            no_repeat_ngram_size: int,
            bad_words_ids: List[List[int]],
            min_length: int,
            eos_token_id: int,
            prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
            num_beams: int,
            encoder_no_repeat_ngram_size: int,
            encoder_input_ids: torch.LongTensor,
    ) -> LogitsProcessorList:
        """
        This class returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
        :obj:`~transformers.LogitsProcessor` instances used to modify the scores of the language model head.
        """

        # init warp parameters
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.no_repeat_ngram_size
        )
        encoder_no_repeat_ngram_size = (
            encoder_no_repeat_ngram_size
            if encoder_no_repeat_ngram_size is not None
            else self.encoder_no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.bad_words_ids
        min_length = min_length if min_length is not None else self.min_len
        eos_token_id = eos_token_id if eos_token_id is not None else self.sep_idx
        # instantiate processors list
        processors = LogitsProcessorList()

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if repetition_penalty is not None and repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
            processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
        if encoder_no_repeat_ngram_size is not None and encoder_no_repeat_ngram_size > 0:
            processors.append(EncoderNoRepeatNGramLogitsProcessor(encoder_no_repeat_ngram_size, encoder_input_ids))
        # if bad_words_ids is not None:
        #     processors.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id))
        if min_length is not None and eos_token_id is not None and min_length > -1:
            processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
        # if prefix_allowed_tokens_fn is not None:
        #     processors.append(PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, num_beams))
        return processors

    @torch.no_grad()
    def generate(
            self,
            early_stopping: Optional[bool] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            model=None,
            data_iterator=None,
            prepare_input_for_encode_step=None,
    ) -> Dict:
        # set init values
        num_beams = self.beam_size
        max_length = self.max_len
        do_sample = self.do_sample
        pad_token_id = self.pad_idx
        bos_token_id = self.cls_idx
        eos_token_id = self.sep_idx

        is_beam_sample_gen_mode = (num_beams > 1) and do_sample is True

        model = model.to(self.device)

        logits_warper = self._get_logits_warper(
            top_k=self.top_k, top_p=self.top_p, temperature=self.temperature, num_beams=self.beam_size,
            response_special_id=99
        )
        special_ids = [eos_token_id, pad_token_id, bos_token_id]
        predict_result = {"predict": [], "reference": []}

        batch_size = 1
        cnt = 0

        for item in tqdm(data_iterator):
            logits_processor = self._get_logits_processor(
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                bad_words_ids=self.bad_words_ids,
                min_length=self.min_len,
                eos_token_id=eos_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                encoder_no_repeat_ngram_size=self.encoder_no_repeat_ngram_size,
                encoder_input_ids=item['history_ids'].to(self.device),
            )

            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                device=self.device,
                length_penalty=self.length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=self.beam_size
            )
            predict, b_score = self.beam_sample(
                item,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                model=model,
                prepare_input_for_encode_step=prepare_input_for_encode_step
            )
            cnt += 1
            print("-------------------------")
            print("{}".format("".join([self.idx2token[_] for _ in item['history_ids'][0].tolist()])))
            # if 'response_ids' in item.keys():
            #     cur_ref = "".join(
            #         [self.idx2token[_] for _ in item['response_ids'][0].tolist() if _ not in special_ids]
            #     )
            #     print("{}".format(cur_ref))
            # for bs, m in zip(b_score, predict):
            #     print("{:.3f}  ".format(bs) + "".join([self.idx2token[_] for _ in m if _ not in [
            #         eos_token_id, pad_token_id, bos_token_id]]))

            cur_pred = "".join([self.idx2token[_] for _ in predict[0] if _ not in special_ids])

            # if len(predict[0]) > len(predict[1]):
            #     cur_pred = "".join([self.idx2token[_] for _ in predict[0] if _ not in special_ids])
            # else:
            #     cur_pred = "".join([self.idx2token[_] for _ in predict[1] if _ not in special_ids])

            print("{} {}".format(cnt, cur_pred))
            predict_result['predict'].append(cur_pred)
            print("-------------------------")
        return predict_result

    def beam_sample(
            self,
            item,
            beam_scorer: BeamSearchScorer,
            logits_processor: Optional[LogitsProcessorList] = None,
            logits_warper: Optional[LogitsProcessorList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            model=None,
            prepare_input_for_encode_step: Callable = None
    ):
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.max_len
        pad_token_id = pad_token_id if pad_token_id is not None else self.pad_idx
        eos_token_id = eos_token_id if eos_token_id is not None else self.sep_idx

        batch_size = 1
        num_beams = self.beam_size

        seqs = torch.tensor([[self.cls_idx]] * num_beams).to(self.device)

        encode_step_outputs = model.encode_step(
            **prepare_input_for_encode_step(item, self.device, expand_batch_size=num_beams)
        )
        encoded_history, past_key_values, history_mask, entity_loss = encode_step_outputs
        # history_mask = torch.ones((num_beams, len(history_mask[0])), dtype=torch.long).to(self.device)

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=self.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))
        step = 1
        while step < max_length:

            decode_step_outputs = model.decode_step(
                response_ids=seqs,
                response_mask=torch.ones_like(seqs).to(self.device),
                history_mask=history_mask,
                past_key_values=past_key_values,
                past_key_values_len=0
            )
            logits = decode_step_outputs[:, -1, :]

            next_token_scores = F.log_softmax(logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(seqs, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            next_token_scores = logits_warper(seqs, next_token_scores)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            if step == 1:
                probs = F.softmax(next_token_scores[:, :vocab_size], dim=-1)
            else:
                probs = F.softmax(next_token_scores, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                seqs,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            seqs = torch.cat([seqs[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            step = step + 1

            if beam_scorer.is_done:
                break

        decoded = beam_scorer.finalize(
            seqs, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        return decoded["sequences"].tolist(), decoded["sequence_scores"].tolist()


class Greedy(BeamSample):
    def __init__(self, config):
        super(Greedy, self).__init__(config)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def greedy_search(
            self,
            item,
            logits_processor: Optional[LogitsProcessorList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            model=None,
            prepare_input_for_encode_step: Callable = None
    ):
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.max_len
        pad_token_id = pad_token_id if pad_token_id is not None else self.pad_idx
        eos_token_id = eos_token_id if eos_token_id is not None else self.sep_idx

        batch_size = len(item['history_ids'])
        num_beams = 1
        seqs = torch.tensor([[self.cls_idx]] * batch_size).to(self.device)
        inputs = prepare_input_for_encode_step(item, self.device, expand_batch_size=1)
        encode_step_outputs = model.encode_step(
            **inputs
        )
        encoded_history, past_key_values, history_mask, entity_loss = encode_step_outputs
        # history_mask = torch.ones((num_beams, len(history_mask[0])), dtype=torch.long).to(self.device)
        unfinished_sequences = torch.ones(seqs.shape[0], dtype=torch.long).to(self.device)
        step = 1
        while step < max_length:
            decode_step_outputs = model.decode_step(
                response_ids=seqs,
                response_mask=torch.ones_like(seqs).to(self.device),
                history_mask=history_mask,
                past_key_values=past_key_values,
                past_key_values_len=0
            )
            logits = decode_step_outputs

            next_token_logits = logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(seqs, next_token_logits)

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # add code that transfomers next_tokens to tokens_to_add
            next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)

            # add token and increase length by one
            seqs = torch.cat([seqs, next_tokens.unsqueeze(-1)], dim=-1)

            # update
            unfinished_sequences = unfinished_sequences.mul((~(next_tokens == eos_token_id)).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sequences.max() == 0:
                break

            # increase cur_len
            step += 1

        return seqs

    @torch.no_grad()
    def greedy_generate(
            self,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            model=None,
            data_iterator=None,
            prepare_input_for_encode_step=None,
    ):
        # init values
        max_length = self.max_len
        pad_token_id = self.pad_idx
        bos_token_id = self.cls_idx
        eos_token_id = self.sep_idx

        model = model.to(self.device)

        special_ids = [eos_token_id, pad_token_id, bos_token_id]
        predict_result = {"predict": [], "reference": []}
        cnt = 0
        for item in tqdm(data_iterator, desc='eval'):
            logits_processor = self._get_logits_processor(
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                bad_words_ids=self.bad_words_ids,
                min_length=self.min_len,
                eos_token_id=eos_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=0,
                encoder_no_repeat_ngram_size=self.encoder_no_repeat_ngram_size,
                encoder_input_ids=item['history_ids'].to(self.device),
            )

            predict = self.greedy_search(
                item,
                logits_processor=logits_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                model=model,
                prepare_input_for_encode_step=prepare_input_for_encode_step
            )

            cur_predict = []
            for p in predict.tolist():
                p_sent = "".join([self.idx2token[_] for _ in p if _ not in special_ids])
                cur_predict.append(p_sent)
            predict_result['predict'].extend(cur_predict)
            # print(cur_predict)

            if 'response_ids' in item.keys():
                cur_reference = []
                for r in item['response_ids'].tolist():
                    r_sent = "".join([self.idx2token[_] for _ in r if _ not in special_ids])
                    cur_reference.append(r_sent)
                predict_result['reference'].extend(cur_reference)

                # print(cur_reference)
            # print("")

        return predict_result
