import copy
import time
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
import json

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
        self.recall_eos_idx = self.token2idx['[SummaryEnd]']

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
            num_beams: int,
            encoder_no_repeat_ngram_size: int,
            encoder_input_ids: torch.LongTensor,
            prefix_allowed_tokens_fn=None,
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
            gen_prefix=False
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
        special_ids = [eos_token_id, pad_token_id, bos_token_id, 98, 40, 41]
        predict_result = {"predict": [], "reference": []}

        batch_size = 1
        cnt = 0
        encoder_no_repeat_ngram_size = self.encoder_no_repeat_ngram_size if self.config['model_name'] \
                                                                            not in ["HRED", "Seq2Seq"] else 0
        for item in tqdm(data_iterator, ncols=50):
            if self.config['model_name'] == "DialogGPT":
                encoder_input_ids = item['input_ids'].to(self.device)
            else:
                encoder_input_ids = item['history_ids'].to(self.device)
            logits_processor = self._get_logits_processor(
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                bad_words_ids=self.bad_words_ids,
                min_length=self.min_len,
                eos_token_id=eos_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
                encoder_input_ids=encoder_input_ids,
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

            if self.config['model_name'] == 'DialogGPT':

                # tmp = [_ for _ in predict[0] if _ != self.pad_idx]
                # tmp_sent = "".join([self.idx2token[_] for _ in tmp])
                # sep_positions = []
                # for idx, i in enumerate(tmp):
                #     if i == self.sep_idx:
                #         sep_positions.append(idx)
                # sep_positions = sorted(sep_positions)
                #
                # if tmp[-1] == self.sep_idx:
                #     tmp = tmp[sep_positions[-2] + 1:sep_positions[-1]]
                # else:
                #     tmp = tmp[sep_positions[-1]:]
                # cur_pred = "".join([self.idx2token[_] for _ in tmp if _ not in special_ids])

                # print('-' * 30)
                # print(tmp_sent)
                # print(cur_pred)
                # print('-' * 30)

                tmp = [_ for _ in predict[0] if _ != self.pad_idx]
                revers_tmp = list(reversed(tmp))
                r_spk_idx = revers_tmp.index(41)
                tmp = tmp[len(tmp) - r_spk_idx:]
                cur_pred = "".join([self.idx2token[_] for _ in tmp if _ not in special_ids])

            else:
                cur_pred = "".join([self.idx2token[_] for _ in predict[0] if _ not in special_ids])

            # if len(predict[0]) > len(predict[1]):
            #     cur_pred = "".join([self.idx2token[_] for _ in predict[0] if _ not in special_ids])
            # else:
            #     cur_pred = "".join([self.idx2token[_] for _ in predict[1] if _ not in special_ids])

            # print("{} {}".format(cnt, cur_pred))
            predict_result['predict'].append(cur_pred)

            if 'response_ids' in item.keys():
                cur_reference = []
                for r in item['response_ids'].tolist():
                    r_sent = "".join([self.idx2token[_] for _ in r if _ not in special_ids])
                    cur_reference.append(r_sent)
                predict_result['reference'].extend(cur_reference)
            if 'target_ids' in item.keys():
                cur_reference = []
                for r in item['target_ids'].tolist():
                    r_sent = "".join([self.idx2token[_] for _ in r if _ not in special_ids])
                    cur_reference.append(r_sent)
                predict_result['reference'].extend(cur_reference)
                print("-" * 30)
                print(cur_pred)
                print(cur_reference[0])
                print("-" * 30)

            if self.config['model_name'] == "DialogGPT":
                print("-" * 30)
                print("".join([self.idx2token[_] for _ in item['input_ids'][0].tolist()]))
                # print(tmp_sent)
                print("refe {}".format(predict_result['reference'][-1]))
                print("pred {}".format(predict_result['predict'][-1]))
                print("-" * 30)

            if self.config['model_name'] == "Reorganize" and cnt <= 5:
                print("-" * 30)
                print("raw responses and entities:")
                # "references_with_entities": combined_references_with_entities,
                raw_and_ent = [self.idx2token[_] for _ in item['references_with_entities'][0].tolist()]
                raw_and_ent = "".join(raw_and_ent).split('[SEP]')
                for rn in raw_and_ent:
                    print(rn)
                print("refe {}".format(predict_result['reference'][-1]))
                print("pred {}".format(predict_result['predict'][-1]))
                print("-" * 30)

            if cnt < 20 and self.config['model_name'] == "Summary":
                print("-" * 30)
                print("history:")
                text = "".join([self.idx2token[_] for _ in item['history_ids'][0].tolist()])
                text = text.replace('[SEP]', '\n')
                print(text)
                print("pred {}".format(predict_result['predict'][-1]))
                print("-" * 30)

            # print("-------------------------")
        return predict_result

    @staticmethod
    def get_rsep_pos(seqs, rsep_idx):
        result = [-1] * len(seqs)
        for sid in range(len(seqs)):
            for tid, token_id in enumerate(seqs[sid]):
                if token_id == rsep_idx:
                    result[sid] = tid
        return result

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
            prepare_input_for_encode_step: Callable = None,

            refresh_rsep_pos=False,
    ):
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.max_len
        pad_token_id = pad_token_id if pad_token_id is not None else self.pad_idx
        eos_token_id = eos_token_id if eos_token_id is not None else self.sep_idx

        batch_size = 1
        num_beams = self.beam_size

        seqs = torch.tensor([[self.cls_idx]] * num_beams).to(self.device)

        beam_next_tokens = torch.tensor([self.cls_idx] * num_beams).to(self.device)

        if self.config['model_name'] == "GenSummaryEntityResponse":
            encode_step_inputs = prepare_input_for_encode_step(
                item, self.device, expand_batch_size=num_beams,
                summary_gate_open=self.config['summary_gate_open'],
                entity_gate_open=self.config['entity_gate_open'],
                recall_gate_network=self.config['recall_gate_network'],
            )
        else:
            encode_step_inputs = prepare_input_for_encode_step(item, self.device, expand_batch_size=num_beams)
        encode_step_outputs = model.encode_step(
            **encode_step_inputs
        )
        if self.config['model_name'] == "DialogGPT":
            seqs = encode_step_inputs['input_ids']
            # token_type_ids = encode_step_inputs['token_type_ids']
            generate_seqs = torch.tensor([[41]] * self.beam_size, dtype=torch.long).to(self.device)
            max_length = min(250, max_length)

        se_idx = self.token2idx['[SummaryEnd]']
        if self.config['model_name'] == "GenSummaryEntityResponse":
            # summary_mask = encode_step_inputs['summary_mask']
            # entity_mask = encode_step_inputs['entity_mask']
            # recall_prefix = pad_sequence(item['prefix'], batch_first=True, padding_value=pad_token_id)
            # seqs = recall_prefix.to(self.device)
            if item.get('prefix') is not None:
                seqs = item['prefix'][0].unsqueeze(dim=0).repeat(num_beams, 1).to(self.device)
                max_length = 250
                wo_prefix_seqs = torch.tensor([[se_idx] * num_beams]).to(self.device)

        # encoded_history, past_key_values, history_mask, entity_loss = encode_step_outputs
        # history_mask = torch.ones((num_beams, len(history_mask[0])), dtype=torch.long).to(self.device)

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=self.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))
        step = 1
        last_step_states = None

        rsep_position = [-1] * num_beams

        while step < max_length:
            if self.config["model_name"] in ["HRED", "Seq2Seq"]:
                if last_step_states is None:
                    last_step_states = encode_step_outputs[1]
                decode_step_outputs = model.decode_step(
                    last_step_states=last_step_states,
                    cur_input_token=beam_next_tokens.unsqueeze(dim=-1),
                )
                logits, last_step_states = decode_step_outputs
                logits = logits[:, -1, :]
            elif self.config['model_name'] == "DialogGPT":
                if seqs.shape[1] > 300:
                    seqs = seqs[:, -300:]
                    # seqs = torch.cat([seqs[:, 0:1], seqs[:, -299:]], dim=1)
                    # if token_type_ids is not None:
                    #     token_type_ids = torch.cat([token_type_ids[:, 0:1], token_type_ids[:, -299:]], dim=1)
                decode_step_outputs = model.decode_step(
                    input_ids=seqs
                )
                logits = decode_step_outputs[0]
                logits = logits[:, -1, :]
            elif self.config['model_name'] == "Reorganize":
                decode_step_outputs = model.decode_step(
                    response_ids=seqs,
                    response_mask=torch.ones_like(seqs).to(self.device),
                    history_mask=encode_step_outputs[2],
                    past_key_values=encode_step_outputs[1],
                    past_key_values_len=0,

                    outer_encoded_hidden_states=encode_step_outputs[3],
                    outer_attention_mask=encode_step_outputs[4]
                )
                logits = decode_step_outputs
                logits = logits[:, -1, :]
            elif self.config['model_name'] == "GenSummaryEntityResponse":
                if refresh_rsep_pos:
                    rsep_position = self.get_rsep_pos(seqs, se_idx)
                decode_step_outputs = model.decode_step(
                    response_ids=seqs,
                    response_mask=torch.ones_like(seqs).to(self.device),
                    outer_encoded_hidden_states=encode_step_outputs[0],
                    outer_attention_mask=encode_step_outputs[1],
                    summary_hidden=encode_step_outputs[2],
                    entity_hidden=encode_step_outputs[3],
                    summary_mask=encode_step_outputs[4],
                    # entity_mask=entity_mask,
                    rsep_position=rsep_position
                )
                logits = decode_step_outputs
                logits = logits[:, -1, :]
            else:
                if refresh_rsep_pos:
                    rsep_position = self.get_rsep_pos(seqs, se_idx)
                decode_step_outputs = model.decode_step(
                    response_ids=seqs,
                    response_mask=torch.ones_like(seqs).to(self.device),
                    history_mask=encode_step_outputs[2],
                    past_key_values=encode_step_outputs[1],
                    past_key_values_len=0
                )
                logits = decode_step_outputs[:, -1, :]

            next_token_scores = F.log_softmax(logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            if self.config['model_name'] == "DialogGPT":
                if generate_seqs.shape != (1, 0):
                    next_token_scores = logits_processor(generate_seqs, next_token_scores)
            else:
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
            if self.config['model_name'] == "DialogGPT":
                generate_seqs = torch.cat([generate_seqs[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                # if token_type_ids is not None:
                #     token_type_ids = torch.cat(
                #         [token_type_ids[beam_idx, :], torch.ones_like(beam_next_tokens).unsqueeze(-1)], dim=-1)

            step = step + 1

            # if len(seqs[0]) > 100:
            #     print("")

            if beam_scorer.is_done:
                break

        decoded = beam_scorer.finalize(
            seqs, beam_scores, next_tokens, next_indices,
            pad_token_id=pad_token_id, eos_token_id=eos_token_id, max_length=max_length
        )

        return decoded["sequences"].tolist(), decoded["sequence_scores"].tolist()

    def with_prefix_beam_sample(
            self,
            item,
            beam_scorer: BeamSearchScorer,
            logits_processor: Optional[LogitsProcessorList] = None,
            logits_warper: Optional[LogitsProcessorList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            model=None,
            prepare_input_for_encode_step: Callable = None,

            rsep_position=None,
    ):
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.max_len
        pad_token_id = pad_token_id if pad_token_id is not None else self.pad_idx
        eos_token_id = eos_token_id if eos_token_id is not None else self.sep_idx

        batch_size = 1
        num_beams = self.beam_size

        seqs = torch.tensor([[self.cls_idx]] * num_beams).to(self.device)

        beam_next_tokens = torch.tensor([self.cls_idx] * num_beams).to(self.device)
        model_name = self.config['model_name']

        if model_name == "GenSummaryEntityResponse":
            encode_step_inputs = prepare_input_for_encode_step(
                item, self.device, expand_batch_size=num_beams,
                summary_gate_open=self.config['summary_gate_open'],
                entity_gate_open=self.config['entity_gate_open'],
                recall_gate_network=self.config['recall_gate_network'],
            )
        elif model_name == "BERTGPTEntity":
            encode_step_inputs = prepare_input_for_encode_step(item, self.device, expand_batch_size=num_beams)
        else:
            raise NotImplemented("model name {} is not implemented".format(model_name))

        encode_step_outputs = model.encode_step(
            **encode_step_inputs
        )
        seqs = item['prefix'][0].unsqueeze(dim=0).repeat(num_beams, 1).to(self.device)
        max_length = 250
        se_idx = self.token2idx['[SummaryEnd]']
        wo_prefix_seqs = torch.tensor([[se_idx]] * num_beams).to(self.device)

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=self.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))
        step = 1

        refresh_rsep_pos = False
        if rsep_position is None:
            rsep_position = [-1] * num_beams
            refresh_rsep_pos = True

        while step < max_length:
            if model_name == "GenSummaryEntityResponse":
                decode_step_outputs = model.decode_step(
                    response_ids=seqs,
                    response_mask=torch.ones_like(seqs).to(self.device),
                    outer_encoded_hidden_states=encode_step_outputs[0],
                    outer_attention_mask=encode_step_outputs[1],
                    summary_hidden=encode_step_outputs[2],
                    entity_hidden=encode_step_outputs[3],
                    summary_mask=encode_step_outputs[4],
                    rsep_position=rsep_position,
                    # entity_mask=entity_mask,
                )
                logits = decode_step_outputs
                logits = logits[:, -1, :]
            elif model_name == "BERTGPTEntity":
                decode_step_outputs = model.decode_step(
                    response_ids=seqs,
                    response_mask=torch.ones_like(seqs).to(self.device),
                    history_mask=encode_step_outputs[2],
                    past_key_values=encode_step_outputs[1],
                    past_key_values_len=0
                )
                logits = decode_step_outputs[:, -1, :]
            else:
                raise NotImplemented("model name {} is not implemented".format(model_name))

            next_token_scores = F.log_softmax(logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            # 1.
            # next_token_scores = logits_processor(seqs, next_token_scores)
            next_token_scores = logits_processor(wo_prefix_seqs, next_token_scores)

            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            # 2.
            # next_token_scores = logits_warper(seqs, next_token_scores)
            next_token_scores = logits_warper(wo_prefix_seqs, next_token_scores)

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
            # 3.
            # beam_outputs = beam_scorer.process(
            #     seqs,
            #     next_token_scores,
            #     next_tokens,
            #     next_indices,
            #     pad_token_id=pad_token_id,
            #     eos_token_id=eos_token_id,
            # )wo_prefix_seqs
            beam_outputs = beam_scorer.process(
                wo_prefix_seqs,
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
            # 4. add
            wo_prefix_seqs = torch.cat([wo_prefix_seqs[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            step = step + 1

            if beam_scorer.is_done:
                break

        decoded = beam_scorer.finalize(
            wo_prefix_seqs, beam_scores, next_tokens, next_indices,
            pad_token_id=pad_token_id, eos_token_id=eos_token_id, max_length=max_length
        )

        return decoded["sequences"].tolist(), decoded["sequence_scores"].tolist()

    def with_prefix_generate(
            self,
            early_stopping: Optional[bool] = None,
            model=None,
            data_iterator=None,
            prepare_input_for_encode_step=None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            gen_prefix=True
    ):
        num_beams = self.beam_size
        max_length = self.max_len
        pad_token_id = self.pad_idx
        bos_token_id = self.cls_idx
        eos_token_id = self.sep_idx

        model = model.to(self.device)

        logits_warper = self._get_logits_warper(
            top_k=self.top_k, top_p=self.top_p,
            temperature=self.temperature, num_beams=self.beam_size,
            response_special_id=99
        )
        unk_id = self.token2idx['[UNK]']
        special_ids = [eos_token_id, pad_token_id, bos_token_id, 98, 40, 41]
        predict_result = {"predict": [], "reference": []}

        batch_size = 1
        cnt = 0
        encoder_no_repeat_ngram_size = self.encoder_no_repeat_ngram_size
        tmp_file = "./cikm_predict_result_{}.json".format(time.strftime("%m-%d-%H-%M"))
        # tmp_pred = json.load(open("cikm_predict_result_08-03-13-14.json", 'r', encoding='utf-8'))
        tmp_pred = dict()
        text_prefix_predict = None
        for item in tqdm(data_iterator, ncols=50):
            if str(cnt) in tmp_pred.keys():
                print("{} continue".format(cnt))
                cnt += 1
                continue
            # for sent in item['sentences_ids'].tolist():
            #     sent_text = [_ for _ in sent if _ != pad_token_id]
            #     print("".join([self.idx2token[_] for _ in sent_text]))

            encoder_input_ids = item['history_ids'].to(self.device)
            if gen_prefix:
                logits_processor = self._get_logits_processor(
                    repetition_penalty=self.repetition_penalty,
                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                    bad_words_ids=self.bad_words_ids,
                    min_length=self.min_len,
                    eos_token_id=self.token2idx['[SummaryEnd]'],
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    num_beams=num_beams,
                    encoder_no_repeat_ngram_size=None,
                    encoder_input_ids=None
                )
                beam_scorer = BeamSearchScorer(
                    batch_size=batch_size,
                    # max_length=250,
                    num_beams=num_beams,
                    device=self.device,
                    length_penalty=self.length_penalty,
                    num_beam_hyps_to_keep=self.beam_size
                )
                del item['prefix']
                prefix_predict, b_score = self.beam_sample(
                    item,
                    beam_scorer,
                    logits_processor=logits_processor,
                    logits_warper=logits_warper,
                    max_length=250,
                    pad_token_id=pad_token_id,
                    eos_token_id=self.token2idx['[SummaryEnd]'],
                    model=model,
                    prepare_input_for_encode_step=prepare_input_for_encode_step,
                    refresh_rsep_pos=True
                )
                prefix_predict = prefix_predict[0]
                prefix_predict = [_ for _ in prefix_predict if _ != pad_token_id and _ != unk_id]
                item['prefix'] = [torch.tensor(prefix_predict)]
                text_prefix_predict = "".join([self.idx2token[i] for i in prefix_predict])
                # print("-" * 30)
                # print(text_prefix_predict)

            logits_processor = self._get_logits_processor(
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                bad_words_ids=self.bad_words_ids,
                min_length=self.min_len,
                eos_token_id=eos_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
                encoder_input_ids=encoder_input_ids,
            )
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                # max_length=510,
                num_beams=num_beams,
                device=self.device,
                length_penalty=self.length_penalty,
                num_beam_hyps_to_keep=self.beam_size
            )
            predict, b_score = self.with_prefix_beam_sample(
                item,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                max_length=510,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                model=model,
                prepare_input_for_encode_step=prepare_input_for_encode_step
            )
            # pred_token_list = [self.idx2token[_] for _ in predict[0]]
            history_str = "".join([self.idx2token[_] for _ in encoder_input_ids[0].tolist()])
            cur_pred = "".join([self.idx2token[_] for _ in predict[0][1:] if _ not in special_ids])
            print()
            print("-" * 30)
            print("history:")
            print(history_str)
            print("recall:")
            print(text_prefix_predict) if text_prefix_predict is not None else None
            print("response:")
            print(cur_pred)
            # print(pred_token_list)
            print("-" * 30)
            predict_result['predict'].append(cur_pred)

            tmp_pred[cnt] = cur_pred
            json.dump(tmp_pred, open(tmp_file, 'w', encoding='utf-8'), ensure_ascii=False)

            cnt += 1
        return predict_result

    @torch.no_grad()
    def generate_one_item(self, item, model, prepare_input_for_encode_step):
        beam_scorer = BeamSearchScorer(
            batch_size=1,
            max_length=self.max_len,
            num_beams=self.beam_size,
            device=self.device,
            length_penalty=self.length_penalty,
            do_early_stopping=None,
            num_beam_hyps_to_keep=self.beam_size
        )
        if self.config['model_name'] == "DialogGPT":
            encoder_input_ids = item['input_ids'].to(self.device)
        else:
            encoder_input_ids = item['history_ids'].to(self.device)
        logits_warper = self._get_logits_warper(
            top_k=self.top_k, top_p=self.top_p, temperature=self.temperature, num_beams=self.beam_size,
            response_special_id=99
        )
        logits_processor = self._get_logits_processor(
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            bad_words_ids=self.bad_words_ids,
            min_length=self.min_len,
            eos_token_id=self.sep_idx,
            prefix_allowed_tokens_fn=None,
            num_beams=self.beam_size,
            encoder_no_repeat_ngram_size=self.encoder_no_repeat_ngram_size,
            encoder_input_ids=encoder_input_ids,
        )
        predict, b_score = self.beam_sample(
            item,
            beam_scorer,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            max_length=self.max_len,
            pad_token_id=self.pad_idx,
            eos_token_id=self.sep_idx,
            model=model,
            prepare_input_for_encode_step=prepare_input_for_encode_step
        )
        cur_pred = "".join([self.idx2token[_] for _ in predict[0] if _ not in
                            [self.sep_idx, self.pad_idx, self.cls_idx, 98]])

        return cur_pred


class Greedy(BeamSample):
    def __init__(self, config):
        super(Greedy, self).__init__(config)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def greedy_search(
            self,
            item,
            logits_processor=None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            model=None,
            prepare_input_for_encode_step: Callable = None,
            with_prefix=False,
            two_processor=False,
    ):
        # init values
        # logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        # logits_processor2 = copy.deepcopy(logits_processor)
        logits_processor, response_logits_processor = logits_processor["recall"], logits_processor["response"]
        max_length = max_length if max_length is not None else self.max_len
        pad_token_id = pad_token_id if pad_token_id is not None else self.pad_idx
        eos_token_id = eos_token_id if eos_token_id is not None else self.sep_idx

        if self.config['model_name'] == "DialogGPT":
            batch_size = len(item['input_ids'])
        else:
            batch_size = len(item['history_ids'])

        seq_mask = None
        seqs = torch.tensor([[self.cls_idx]] * batch_size).to(self.device)
        if self.config['model_name'] == "DialogGPT":
            seqs = item['input_ids'].to(self.device)
            generate_seqs = torch.tensor([[]] * batch_size).to(self.device)
        elif self.config['model_name'] == 'GenSummaryEntityResponse':
            # recall_prefix = pad_sequence(item['prefix'], batch_first=True, padding_value=pad_token_id)
            # seqs = recall_prefix.to(self.device)
            # seqs = item['prefix'][0].unsqueeze(dim=0).to(self.device)
            if with_prefix:
                seqs = item['prefix'][0].unsqueeze(dim=0).to(self.device)
            max_length = 500
        elif self.config['model_name'] == "BERTGPTEntity" and with_prefix is True:
            seqs = item['prefix'][0].unsqueeze(dim=0).to(self.device)
            max_length = 250
        next_tokens = torch.tensor([self.cls_idx] * batch_size).to(self.device)
        inputs = prepare_input_for_encode_step(item, self.device, expand_batch_size=1)
        encode_step_outputs = model.encode_step(**inputs)
        # encoded_history, past_key_values, history_mask, entity_loss = encode_step_outputs
        # history_mask = torch.ones((num_beams, len(history_mask[0])), dtype=torch.long).to(self.device)
        unfinished_sequences = torch.ones(seqs.shape[0], dtype=torch.long).to(self.device)
        step = 1
        last_step_states = None

        # input_ids = item['input_ids']
        # response_ids = item['response_ids']
        # for idx in range(len(input_ids)):
        #     print("".join([self.idx2token[_] for _ in input_ids[idx].tolist()]))
        #     print("".join([self.idx2token[_] for _ in response_ids[idx].tolist()]))
        rsep_position = [-1]

        past_key_and_values = None
        summary_is_end_tags = [False] * batch_size
        summary_end_idx = self.token2idx['[SummaryEnd]']
        summary_cache = [[] for _ in range(batch_size)]
        response_cache = [[] for _ in range(batch_size)]
        while step < max_length:

            if self.config["model_name"] in ["HRED", "Seq2Seq"]:
                if last_step_states is None:
                    last_step_states = encode_step_outputs[1]
                decode_step_outputs = model.decode_step(
                    last_step_states=last_step_states,
                    cur_input_token=next_tokens.unsqueeze(dim=-1),
                )
                logits, last_step_states = decode_step_outputs
                next_token_logits = logits[:, -1, :]
            elif self.config['model_name'] == "DialogGPT":
                if seqs.shape[1] > 300:
                    seqs = torch.cat([seqs[:, 0:1], seqs[:, -299:]], dim=1)
                decode_step_outputs = model.decode_step(
                    input_ids=seqs
                )
                logits = decode_step_outputs[0]
                next_token_logits = logits[:, -1, :]
            elif self.config['model_name'] == "Reorganize":
                decode_step_outputs = model.decode_step(
                    response_ids=seqs,
                    response_mask=torch.ones_like(seqs).to(self.device),
                    history_mask=encode_step_outputs[2],
                    past_key_values=encode_step_outputs[1],
                    past_key_values_len=0,

                    outer_encoded_hidden_states=encode_step_outputs[3],
                    outer_attention_mask=encode_step_outputs[4]
                )
                logits = decode_step_outputs
                next_token_logits = logits[:, -1, :]
            elif self.config['model_name'] == "HireFusionModel":
                decode_step_outputs = model.decode_step(
                    response_ids=seqs,
                    response_mask=torch.ones_like(seqs).to(self.device),
                    history_mask=encode_step_outputs[2],
                    past_key_values=encode_step_outputs[1],
                    past_key_values_len=0,

                    entity_mem=encode_step_outputs[4],
                    summary_mem=encode_step_outputs[5],
                )
                logits = decode_step_outputs
                next_token_logits = logits[:, -1, :]
            elif self.config['model_name'] == "GenSummaryEntityResponse":
                # mask_response = (seqs != pad_token_id).long()
                cur_use_key_and_values = self.config['use_past_key_and_values'] and (past_key_and_values is not None)
                cur_input_seqs = seqs if not cur_use_key_and_values else seqs[..., -1:]
                # if seq_mask is not None:
                #     seq_mask = torch.cat([seq_mask, torch.ones(seqs.shape[0], 1).to(self.device)], dim=1)
                decode_step_outputs = model.decode_step(
                    response_ids=cur_input_seqs,
                    response_mask=torch.ones_like(seqs).to(self.device),
                    outer_encoded_hidden_states=encode_step_outputs[0],
                    outer_attention_mask=encode_step_outputs[1],
                    summary_hidden=encode_step_outputs[2],
                    entity_hidden=encode_step_outputs[3],
                    summary_mask=encode_step_outputs[4],
                    rsep_position=rsep_position,
                    return_past_key_and_values=self.config['use_past_key_and_values'],
                    past_key_and_values=past_key_and_values
                )
                if self.config['use_past_key_and_values']:
                    logits = decode_step_outputs[0]
                    past_key_and_values = decode_step_outputs[1]
                else:
                    logits = decode_step_outputs

                if self.config.get("model_recall_strategy"):
                    summary_is_end_tags = [0 if summary_end_idx in seq else 1 for seq in seqs]
                    select_logits = []
                    for seq_idx, select_idx in enumerate(summary_is_end_tags):
                        select_logits.append(logits[select_idx][seq_idx])
                    logits = torch.stack(select_logits).to(self.device)
                next_token_logits = logits[:, -1, :]

            else:
                decode_step_outputs = model.decode_step(
                    response_ids=seqs,
                    response_mask=torch.ones_like(seqs).to(self.device),
                    history_mask=encode_step_outputs[2],
                    past_key_values=encode_step_outputs[1],
                    past_key_values_len=0
                )
                logits = decode_step_outputs
                next_token_logits = logits[:, -1, :]

            # pre-process distribution

            if two_processor:
                next_tokens_scores = []
                for seq, logits in zip(seqs.tolist(), next_token_logits):
                    if summary_end_idx in seq:
                        seq = seq[seq.index(summary_end_idx):]
                        seq = torch.tensor([seq]).to(self.device)
                        after_process = response_logits_processor(seq, logits.unsqueeze(dim=0))  # 1, vocab_size
                    else:
                        seq = torch.tensor([seq]).to(self.device)
                        after_process = logits_processor(seq, logits.unsqueeze(dim=0))  # 1, vocab_size
                    next_tokens_scores.append(after_process)
                next_tokens_scores = torch.cat(next_tokens_scores)
            else:
                next_tokens_scores = logits_processor(seqs, next_token_logits)  # batch_size, vocab_size

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # add code that transfomers next_tokens to tokens_to_add
            next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)

            # add token and increase length by one
            seqs = torch.cat([seqs, next_tokens.unsqueeze(-1)], dim=-1)

            if self.config['rsep_as_associate']:
                if next_tokens.item() == self.token2idx['[SummaryEnd]']:
                    rsep_position = [len(seqs[0])]

            if self.config['model_name'] == "DialogGPT":
                generate_seqs = torch.cat([generate_seqs, next_tokens.unsqueeze(-1)], dim=-1)
            # update
            unfinished_sequences = unfinished_sequences.mul((~(next_tokens == eos_token_id)).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sequences.max() == 0:
                break

            # increase cur_len
            step += 1

        if self.config['model_name'] == "DialogGPT":
            return generate_seqs

        return seqs

    def convert_ids_to_tokens(self, ids):
        return [self.idx2token[_] for _ in ids]

    @torch.no_grad()
    def greedy_generate(
            self,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            model=None,
            data_iterator=None,
            prepare_input_for_encode_step=None,
            with_prefix=False,
            two_processor=False
    ):
        # init values
        max_length = self.max_len
        pad_token_id = self.pad_idx
        bos_token_id = self.cls_idx
        eos_token_id = self.sep_idx

        model = model.to(self.device)

        special_ids = [eos_token_id, pad_token_id, bos_token_id, 98]
        predict_result = {"predict": [], "reference": []}
        cnt = 0
        encoder_no_repeat_ngram_size = self.encoder_no_repeat_ngram_size if self.config['model_name'] not in \
                                                                            ["HRED", 'Seq2Seq'] else 0
        for item in tqdm(data_iterator, desc='eval'):
            if self.config['model_name'] == "DialogGPT":
                encoder_input_ids = item['input_ids'].to(self.device)
            # elif self.config['model_name'] == "GenSummaryEntityResponse":
            #     encoder_input_ids = item['input_for_crossattention'].to(self.device)
            else:
                encoder_input_ids = item['history_ids'].to(self.device)
            recall_logits_processor = self._get_logits_processor(
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                bad_words_ids=self.bad_words_ids,
                min_length=self.min_len,
                eos_token_id=eos_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=0,
                encoder_no_repeat_ngram_size=0,
                encoder_input_ids=encoder_input_ids,
            )
            response_logits_processor = self._get_logits_processor(
                repetition_penalty=1,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                bad_words_ids=self.bad_words_ids,
                min_length=self.min_len,
                eos_token_id=eos_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=0,
                encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
                encoder_input_ids=encoder_input_ids,
            )

            predict = self.greedy_search(
                item,
                logits_processor={"recall": recall_logits_processor, "response": response_logits_processor},
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                model=model,
                prepare_input_for_encode_step=prepare_input_for_encode_step,
                with_prefix=with_prefix,
                two_processor=two_processor,
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
            elif 'target_ids' in item.keys():
                cur_reference = []
                for r in item['target_ids'].tolist():
                    r_sent = "".join([self.idx2token[_] for _ in r if _ not in special_ids])
                    cur_reference.append(r_sent)
                predict_result['reference'].extend(cur_reference)

                # print(cur_reference)
            # print("")
            #
            # if self.config['model_name'] == "DialogGPT":
            #     for idx, (i, j) in enumerate(zip(predict_result['reference'], predict_result['predict'])):
            #         print("-" * 30)
            #         print("".join([self.idx2token[_] for _ in item['input_ids'][idx].tolist()]))
            #         print("refe {}".format(i))
            #         print("pred {}".format(j))
            #         print("-" * 30)

        return predict_result
