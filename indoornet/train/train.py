# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse

from outdoornet.models.model_wrapper import ModelWrapper
from outdoornet.models.model_checkpoint import ModelCheckpoint
from outdoornet.trainers.trainer import HorovodTrainer
from outdoornet.dataandgeo.utils import parse_train_file
from outdoornet.dataandgeo.utils import set_debug, filter_args_create
from outdoornet.loss import WandbLogger


def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='PackNet-SfM training script')
    parser.add_argument('file', type=str, help='Input file (.ckpt or .yaml)')
    args = parser.parse_args()
    assert args.file.endswith(('.ckpt', '.yaml')), \
        'You need to provide a .ckpt of .yaml file'
    return args


def train(file):
    """
    Monocular depth estimation training script.

    Parameters
    ----------
    file : str
        Filepath, can be either a
        **.yaml** for a yacs configuration file or a
        **.ckpt** for a pre-trained checkpoint file.
    """
    # Initialize horovod
    #hvd_init()

    # Produce configuration and checkpoint from filename
    config, ckpt = parse_train_file(file)

    # Set debug if requested
    set_debug(config.debug)

    # Wandb Logger
    # logger = None if config.wandb.dry_run  \
    #     else filter_args_create(WandbLogger, config.wandb)
    logger = None if config.wandb.dry_run \
             else filter_args_create(WandbLogger, config.wandb)
    # model checkpoint
    checkpoint = None if config.checkpoint.filepath is ''  else \
        filter_args_create(ModelCheckpoint, config.checkpoint)

    # Initialize model wrapper
    model_wrapper = ModelWrapper(config, resume=ckpt, logger=logger)

    # Create trainer with args.arch parameters
    trainer = HorovodTrainer(**config.arch, checkpoint=checkpoint)

    # Train model
    trainer.fit(model_wrapper)


if __name__ == '__main__':
    # args = parse_args()
    # train(args.file)
    train(file='overfit_kitti1.yaml')
    #train(file='train_kitti.yaml')