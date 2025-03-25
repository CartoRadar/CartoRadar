import os
import torch

from detectron2.data import (
    build_detection_train_loader,
    build_detection_test_loader,
)
from detectron2.engine import DefaultTrainer
from dataloader.mapper import RfMapper
from engine.hooks import ImageVisHook
from evaluation.evaluator import DepthEvaluator


class RfDepthTrainer(DefaultTrainer):
    """The customized trainer for training depth and surface normal."""

    def __init__(self, cfg):
        cfg.MODEL.PIXEL_MEAN = [0 for _ in range(256)]
        cfg.MODEL.PIXEL_STD = [1 for _ in range(256)]
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        """Take in the config file and return a torch Dataloader for training"""
        mapper = RfMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """Take in the config file and return a torch Dataloader for testing"""
        mapper = RfMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """Build evaluator(s) for metrics"""
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        os.makedirs(output_folder, exist_ok=True)
        ret = [DepthEvaluator(output_folder)]
        return ret

    def build_hooks(self):
        """Overwrite this function so that new hooks can be added.

        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        ret = super().build_hooks()
        ret.append(ImageVisHook(self.cfg))
        return ret

    @classmethod
    def build_optimizer(cls, cfg, model):
        """Returns a torch.optim.Optimizer"""
        if cfg.SOLVER.NAME == 'SGD':
            args = {
                "params": model.parameters(),
                "lr": cfg.SOLVER.BASE_LR,
                "momentum": cfg.SOLVER.MOMENTUM,
                "nesterov": cfg.SOLVER.NESTEROV,
                "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
            }
            optim_cls = torch.optim.SGD
        elif cfg.SOLVER.NAME == 'AdamW':
            args = {"params": model.parameters(), "lr": cfg.SOLVER.BASE_LR, "weight_decay": 0.00001}
            optim_cls = torch.optim.AdamW
        else:
            raise NameError(f'Unrecognize solver name: {cfg.SOLVER.NAME}')
        return optim_cls(**args)


def get_trainer_class(cfg):
    """Get the trainer class according to the dataset."""
    return RfDepthTrainer
