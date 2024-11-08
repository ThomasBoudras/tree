from typing import Any

import torch
from lightning import LightningModule
import os
import numpy as np

class Module(LightningModule):
    def __init__(
        self,
        network,
        loss,
        train_metrics,
        val_metrics,
        test_metrics,
        scheduler,
        optimizer,
        predictions_save_dir=None,        
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.network = network.instance
        self.loss = loss
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.predictions_save_dir = predictions_save_dir
    
    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        
        return {"optimizer": optimizer}

    def forward(self, x: torch.Tensor):
        return self.network(x)

    def step(self, batch: Any, stage, metrics_function):
        inputs, targets , meta_data = batch
        preds = self.forward(inputs)
        
        loss = self.loss(preds, targets)
        self.log(
                name=os.path.join(stage, "loss"),
                value=loss, 
                on_step=True,
                on_epoch=True, 
                prog_bar=False
                )
        
        if metrics_function :
            metrics_function.update(preds, targets, meta_data)
        
        return loss, preds, targets
    

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch=batch, stage="train", metrics_function=self.train_metrics)
        return {"loss": loss, "preds": preds, "targets": targets}


    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch=batch, stage="val", metrics_function=self.val_metrics)
        return {"loss": loss, "preds": preds, "targets": targets}


    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch=batch, stage="val", metrics_function=self.test_metrics)
        return {"loss": loss, "preds": preds, "targets": targets}


    def final_step(self, stage, metrics_function):
        if metrics_function :
            metrics = metrics_function.compute()
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for metric_name, metric_value in value.items():
                        self.log(
                        f"{stage}/{key}/{metric_name}",
                        metric_value,
                        sync_dist=True,
                        on_step=False,
                        on_epoch=True,
                    )

                else :
                    self.log(
                        f"{stage}/{key}",
                        value,
                        sync_dist=True,
                        on_step=False,
                        on_epoch=True,
                    )

            metrics_function.reset()

    def on_train_epoch_end(self):
        self.final_step("train", self.train_metrics)
    
    def on_validation_epoch_end(self):
        self.final_step("val", self.val_metrics)

    def on_test_epoch_end(self):
        self.final_step("val", self.val_metrics)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # At predict time, there are (normally) only inputs, no targets

        if (len(batch) == 2) and (
            not isinstance(batch, torch.Tensor)
        ):  # handle the case of a batch of size 2
            res = self(batch[0])
        else:
            res = self(batch)
        if self.predictions_save_dir is not None:
            np.save(
                os.path.join(self.predictions_save_dir, str(batch_idx) + ".npy"),
                res.cpu().numpy().astype(np.float16),
            )
            if len(batch) == 2:  # save also labels
                np.save(
                    os.path.join(self.predictions_save_dir, str(batch_idx) + "_label.npy"),
                    batch[1].cpu().numpy().astype(np.float16),
                )
            return None
        else:
            raise Exception("Please give a name for the prediction file ")