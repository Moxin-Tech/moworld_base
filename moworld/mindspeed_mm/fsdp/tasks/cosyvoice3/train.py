import logging

from torchdata.stateful_dataloader import StatefulDataLoader
from mindspeed.fsdp.utils.log import print_rank

from mindspeed_mm.fsdp.data import build_mm_dataset
from mindspeed_mm.fsdp.params.argument import Arguments, parse_args
from mindspeed_mm.fsdp.train.trainer import Trainer
from mindspeed_mm.fsdp.utils.device import get_device_type
from mindspeed_mm.config.config_manager import ConfigManager


def get_cosyvoice_dataloader(args):
    """Build training dataloader with proper parallel partitioning."""
    print_rank(logging.info, "Prepare data")
    data_config = args.data

    datasets = build_mm_dataset(data_config.dataset_param)
    dataloader_param = data_config.dataloader_param.to_dict()
    train_dataloader = StatefulDataLoader(
        datasets,
        batch_size=None,
        pin_memory=dataloader_param.get('pin_memory'),
        pin_memory_device=get_device_type(),
        num_workers=dataloader_param.get('num_workers'),
        prefetch_factor=dataloader_param.get('prefetch_factor')
    )

    return train_dataloader


if __name__ == "__main__":
    # Entry point for training script
    args = ConfigManager(config_class=Arguments).load_and_parse()
    trainer = Trainer(args=args, dataloader_provider=get_cosyvoice_dataloader)
    trainer.train()
