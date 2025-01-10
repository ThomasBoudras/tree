from lightning import Trainer
from src.utils import utils

log = utils.get_logger(__name__)

class defaultTrainer(Trainer):
    def fit(self, model, datamodule=None, ckpt_path=None):
        if ckpt_path:
            log.info("Start fit from a checkpoint, recalculate loss validation values")
            model.compute_initial_metrics_state = True
            self.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
            model.compute_initial_metrics_state = False
            model.initialisation_state = True

        super().fit(model, datamodule=datamodule, ckpt_path=ckpt_path)