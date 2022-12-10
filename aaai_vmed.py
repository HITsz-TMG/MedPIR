from src.model import BERTGPTEntity
from main_config import config
import torch
from cikm_dataset.summary_response_dataset import SummaryResponseDataset
from cikm_generate_utils.generator import BeamSample
from cikm_generate_utils import prepare_input_utils
from cikm_model.vmed import VSumDialog
from cikm_trainer.vmed_trainer import VMedTrainer
import pickle


def train_vmed():
    use_summary_entity_crossattention = True
    config["summary_entity_encoder"] = use_summary_entity_crossattention

    with_entity = False
    with_summary = True  # Decoder with summary
    decoder_with_entity = True
    config['model_name'] = "GenSummaryEntityResponse"

    dev_dataset = SummaryResponseDataset(
        vocab_path=config['vocab_path'],
        data_path=config['dev_data_path'],
        summary_data_path=config['dialogue_summary_dev_path'],
        data_type="dev",
        config=config,
        with_entity=with_entity,
        with_summary=with_summary,
        decoder_with_entity=decoder_with_entity,
        vmed=True
    )
    train_dataset = SummaryResponseDataset(
        vocab_path=config['vocab_path'],
        data_path=config['dev_data_path'],
        summary_data_path=config['dialogue_summary_dev_path'],
        data_type="dev",
        config=config,
        with_entity=with_entity,
        with_summary=with_summary,
        decoder_with_entity=decoder_with_entity,
        vmed=True
    )
    # for i in dev_dataset.get_dataloader(batch_size=5, use_vmed=True):
    #     print(i)

    config.update({
        'dataset_class': dev_dataset,
    })

    model = VSumDialog(config)

    trainer = VMedTrainer(
        train_dataset,
        model,
        dev_dataset=dev_dataset,
        config=config,
        save_root="./cikm_save/SummaryAndResponse",
    )

    trainer.train()


if __name__ == '__main__':
    train_vmed()
