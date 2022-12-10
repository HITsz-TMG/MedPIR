def prepare_input_for_greedy_generate_BertGPT(batch, device, expand_batch_size=1):
    history_ids = batch['history_ids'].to(device)
    history_mask = batch['history_mask'].to(device)
    history_speaker = batch['history_speaker'].to(device) if 'history_speaker' in batch.keys() else None
    kv_inputs = {
        "history_ids": history_ids,
        "token_type_ids": history_speaker,
        "history_mask": history_mask,
    }
    return kv_inputs


def prepare_input_for_encode_step_BertGPT(batch, device, expand_batch_size=1):
    assert len(batch['history_ids']) == 1
    history_ids = batch['history_ids'].to(device)
    history_mask = batch['history_mask'].to(device)
    history_speaker = batch['history_speaker'].to(device) if 'history_speaker' in batch.keys() else None
    if expand_batch_size != 1:
        history_ids = history_ids.repeat(expand_batch_size, 1)
        history_mask = history_mask.repeat(expand_batch_size, 1)
        if history_speaker is not None:
            history_speaker = history_speaker.repeat(expand_batch_size, 1)
    kv_inputs = {
        "history_ids": history_ids,
        "token_type_ids": history_speaker,
        "history_mask": history_mask,
    }
    return kv_inputs


def prepare_input_for_encode_step_Reorganize(batch, device, expand_batch_size=1):
    # this is method-3
    assert len(batch['history_ids']) == 1
    history_ids = batch['references_with_entities'].to(device)
    history_mask = batch['references_mask'].to(device)
    token_type_ids = batch['token_type_ids'].to(device)
    input_for_crossattention = batch["history_ids"].to(device)
    crossattention_mask = batch["history_mask"].to(device)
    if expand_batch_size != 1:
        history_ids = history_ids.repeat(expand_batch_size, 1)
        history_mask = history_mask.repeat(expand_batch_size, 1)
        token_type_ids = token_type_ids.repeat(expand_batch_size, 1)
        input_for_crossattention = input_for_crossattention.repeat(expand_batch_size, 1)
        crossattention_mask = crossattention_mask.repeat(expand_batch_size, 1)

    kv_inputs = {
        "history_ids": history_ids,
        "token_type_ids": token_type_ids,
        "history_mask": history_mask,
        "input_for_crossattention": input_for_crossattention,
        "crossattention_mask": crossattention_mask
    }
    return kv_inputs


def prepare_input_for_greedy_generate_HRED(batch, device, expand_batch_size=1):
    history_ids = batch['history_ids'].to(device)
    history_mask = batch['history_mask'].to(device)
    kv_inputs = {
        "history_ids": history_ids,
        "history_mask": history_mask,
        "utterance_num": batch['utterance_num']
    }
    return kv_inputs


def prepare_input_for_encode_step_HRED(batch, device, expand_batch_size=1):
    assert len(batch['history_ids']) == 1
    history_ids = batch['history_ids'].to(device)
    history_mask = batch['history_mask'].to(device)
    if expand_batch_size != 1:
        history_ids = history_ids.repeat(expand_batch_size, 1, 1)
        history_mask = history_mask.repeat(expand_batch_size, 1, 1)
    kv_inputs = {
        "history_ids": history_ids,
        "history_mask": history_mask,
        "utterance_num": batch['utterance_num'] * expand_batch_size
    }
    return kv_inputs


def prepare_input_for_greedy_generate_Seq2Seq(batch, device, expand_batch_size=1):
    history_ids = batch['history_ids'].to(device)
    history_mask = batch['history_mask'].to(device)
    kv_inputs = {
        "history_ids": history_ids,
        "history_mask": history_mask,
    }
    return kv_inputs


def prepare_input_for_encode_step_Seq2Seq(batch, device, expand_batch_size=1):
    assert len(batch['history_ids']) == 1
    history_ids = batch['history_ids'].to(device)
    history_mask = batch['history_mask'].to(device)
    if expand_batch_size != 1:
        history_ids = history_ids.repeat(expand_batch_size, 1, 1)
        history_mask = history_mask.repeat(expand_batch_size, 1, 1)
    kv_inputs = {
        "history_ids": history_ids,
        "history_mask": history_mask,
    }
    return kv_inputs


def prepare_input_for_greedy_generate_GPT(batch, device, expand_batch_size=1):
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


def prepare_input_for_encode_step_GPT(batch, device, expand_batch_size=1):
    input_ids = batch['input_ids'].to(device)
    if batch['token_type_ids'] is not None:
        token_type_ids = batch['token_type_ids'].to(device)
    else:
        token_type_ids = None

    if expand_batch_size != 1:
        input_ids = input_ids.repeat(expand_batch_size, 1)
        if batch['token_type_ids'] is not None:
            token_type_ids = token_type_ids.repeat(expand_batch_size, 1)
        else:
            token_type_ids = None
    kv_inputs = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
    }
    return kv_inputs


def prepare_input_for_summary_generation(batch, device, expand_batch_size=1):
    history_ids = batch['history_ids'].to(device)
    history_mask = batch['history_mask'].to(device)
    history_spk = batch['history_spk'].to(device)
    if expand_batch_size != 1:
        history_ids = history_ids.repeat(expand_batch_size, 1)
        history_mask = history_mask.repeat(expand_batch_size, 1)
        history_spk = history_spk.repeat(expand_batch_size, 1)
    kv_inputs = {
        "history_ids": history_ids,
        "token_type_ids": history_spk,
        "history_mask": history_mask,
    }

    return kv_inputs



def prepare_input_for_GenSummaryEntityResponse(
        batch, device, expand_batch_size=1,
        summary_gate_open=False,
        entity_gate_open=False,
        recall_gate_network=None,
):
    assert len(batch['history_ids']) == 1

    # response_mask = batch['target_mask'].to(device)
    # response_ids = batch['target_ids'].to(device)
    input_for_crossattention = batch['history_ids'].to(device)
    crossattention_mask = batch['history_mask'].to(device)

    if expand_batch_size != 1:
        input_for_crossattention = input_for_crossattention.repeat(expand_batch_size, 1)
        crossattention_mask = crossattention_mask.repeat(expand_batch_size, 1)
        # response_mask = response_mask.repeat(expand_batch_size, 1)
        # response_ids = response_ids.repeat(expand_batch_size, 1)

    kv_inputs = {
        "input_for_crossattention": input_for_crossattention,
        "crossattention_mask": crossattention_mask,
        # "response_mask": response_mask,
        # "response_ids": response_ids,
    }

    if summary_gate_open:
        summary_ids = batch['summary_ids'].to(device)
        summary_mask = batch['summary_mask'].to(device)
        if expand_batch_size != 1:
            summary_ids = summary_ids.repeat(expand_batch_size, 1)
            summary_mask = summary_mask.repeat(expand_batch_size, 1)
        kv_inputs.update({
            "summary_ids": summary_ids,
            "summary_mask": summary_mask,
        })
    if entity_gate_open:
        entity_ids = batch['entity_ids'].to(device)
        entity_mask = batch['entity_mask'].to(device)
        if expand_batch_size != 1:
            entity_ids = entity_ids.repeat(expand_batch_size, 1)
            entity_mask = entity_mask.repeat(expand_batch_size, 1)
        kv_inputs.update({
            "entity_ids": entity_ids,
            "entity_mask": entity_mask,
        })
    if recall_gate_network == "GAT":
        sentences_ids = batch['sentences_ids'].to(device)
        sentences_mask = batch['sentences_mask'].to(device)
        sentences_num = batch['sentences_num']
        adjacent_matrix = batch['adjacent_matrix'].to(device)
        head_type = batch['head_type'].to(device)
        edge_type = batch['edge_type'].to(device)
        target_recall = batch['target_recall'].to(device)
        if expand_batch_size != 1:
            sentences_ids = sentences_ids.repeat(expand_batch_size, 1)
            sentences_mask = sentences_mask.repeat(expand_batch_size, 1)
            sentences_num = sentences_num * expand_batch_size
            adjacent_matrix = adjacent_matrix.repeat(expand_batch_size, 1, 1)
            head_type = head_type.repeat(expand_batch_size, 1)
            edge_type = edge_type.repeat(expand_batch_size, 1, 1)
            target_recall = target_recall.repeat(expand_batch_size, 1)
        kv_inputs.update({
            "sentences_ids": sentences_ids,
            "sentences_mask": sentences_mask,
            "sentences_num": sentences_num,
            "adjacent_matrix": adjacent_matrix,
            "head_type": head_type,
            "edge_type": edge_type,
            "target_recall": target_recall,
        })

    return kv_inputs


def prepare_input_for_RecallBERTGPT(batch, device, expand_batch_size=1):

    assert len(batch['history_ids']) == 1
    history_ids = batch['history_ids'].to(device)
    history_mask = batch['history_mask'].to(device)
    if expand_batch_size != 1:
        history_ids = history_ids.repeat(expand_batch_size, 1)
        history_mask = history_mask.repeat(expand_batch_size, 1)
    kv_inputs = {
        "history_ids": history_ids,
        "history_mask": history_mask,
    }
    return kv_inputs
