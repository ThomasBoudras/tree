from typing import Any

import torch
from lightning import LightningModule
import os
import numpy as np
import torch.nn.functional as F

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
        predictions_save_dir= None,
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
        self.compute_initial_metrics_state = False
        self.initialisation_state = False
        self.initial_metrics = {}
        self.initial_metrics["val/loss"] = []
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

    def forward(self, x: torch.Tensor, meta_data):
        return self.network(x, meta_data)

    def step(self, batch: Any, stage, metrics_function):
        inputs, targets , meta_data = batch
        preds = self.forward(inputs, meta_data)

        if preds.shape[-1] != targets.shape[-1] :
            scale_factor = targets.shape[-1] / preds.shape[-1] 
            preds = F.interpolate(preds, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        
        loss = self.loss(preds, targets)

        if self.compute_initial_metrics_state and stage == "val":
            self.initial_metrics["val/loss"].append(loss)   
        else :
            if self.initialisation_state and stage =="val" :
                loss = self.initial_metrics["val/loss"]
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
            if self.compute_initial_metrics_state :
                self.initial_metrics["val/loss"] = torch.mean(torch.stack(self.initial_metrics["val/loss"])).item()
                for key, value in metrics.items():
                    if isinstance(value, dict):
                        for metric_name, metric_value in value.items():
                            self.initial_metrics[f"{stage}/{key}/{metric_name}"] = metric_value
                    else :
                        self.initial_metrics[f"{stage}/{key}"] = value

            else :
                if self.initialisation_state:
                    for key, value in metrics.items():
                        if isinstance(value, dict):
                            for metric_name in value.keys():
                                metrics[key][metric_name] = self.initial_metrics[f"{stage}/{key}/{metric_name}"]
                        else:
                            metrics[key] = self.initial_metrics[f"{stage}/{key}"]
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
        self.final_step("val", self.test_metrics)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # At predict time, there are (normally) only inputs, no targets
        input, meta_data = batch
        res = self(input, meta_data)
        
        if res.isnan().sum() != 0 :
            raise Exception("predictions with nan")
        
        target_nb_pixel = meta_data["target_nb_pixel"]
        if res.shape[-1] != target_nb_pixel :
            scale_factor = target_nb_pixel.item() / res.shape[-1] 
            res = F.interpolate(res, scale_factor=scale_factor, mode='bilinear', align_corners=False)

        if self.predictions_save_dir is not None:
            np.save(
                os.path.join(self.predictions_save_dir, str(batch_idx) + ".npy"),
                res.cpu().numpy().astype(np.float16),
            )

        else:
            raise Exception("Please give a name for the prediction dir ")