from typing import List, Optional

import hydra
from omegaconf import DictConfig
import torch
from lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)

from src.utils import utils

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule.instance._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule.instance)

    # Init lightning model
    log.info(f"Instantiating model <{config.model.instance._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model.instance)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))


    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger)
        
    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    if not config.only_test :
        if config.get("ckpt_path"):
            ckpt_path = config.get("ckpt_path")
            if config.load_just_weights :
                log.info(f"Start of training from checkpoint {ckpt_path} using only the weights !")
                checkpoint = torch.load(ckpt_path)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                ckpt_path = None
            else :
                log.info(f"Start of training from checkpoint {ckpt_path} !")
        else :
            log.info("Starting training from scratch!")
            ckpt_path = None
        
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


        # Evaluate model on test set, using the best model achieved during training

        if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
            log.info("Starting testing!")
            ckpt_path = trainer.checkpoint_callback.best_model_path
            if ckpt_path == "":
                log.warning("Best ckpt not found! Using current weights for testing...")
                ckpt_path = None
            log.info(f"Best ckpt path: {ckpt_path}")
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    else :
        if config.get("ckpt_path"):
            ckpt_path = config.ckpt_path
            log.info(f"Starting testing with {ckpt_path}!")
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        else :
            raise Exception("Give a checkpoint to test")
        

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]

    return None
