"""Custom Evaluators.

- DepthEvaluator: depth_l1_mean, depth_l1_median, depth_l1_80, depth_l1_90
                  psnr, ssim
- SnEvaluator: sn_angle_mean, sn_angle_median, sn_angle_80, sn_angle_90
"""

import os
import logging
import numpy as np
from scipy import stats
from collections import OrderedDict

import torch
import torch.nn.functional as F
from detectron2.evaluation import DatasetEvaluator
from dataloader.iqa import masked_psnr, masked_ssim


class DepthEvaluator(DatasetEvaluator):
    """Evaluate depth l1 metrics."""

    def __init__(self, output_dir=None):
        """
        Args:
            output_dir (str): an output directory to dump results.
        """
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._output_dir = output_dir

        self._l1_mean = []
        self._l1_median = []
        self._l1_80 = []
        self._l1_90 = []
        self._psnr = []
        self._ssim = []
        self._inr_precision = []
        self._inr_recall = []

    def reset(self):
        self._l1_mean.clear()
        self._l1_median.clear()
        self._l1_80.clear()
        self._l1_90.clear()
        self._psnr.clear()
        self._ssim.clear()
        self._inr_precision.clear()
        self._inr_recall.clear()

    def process(self, inputs, outputs):
        """This is the evaluator for in-range regression + oor classification problem.
        For the Laplace distribution
        Process the pair of inputs and outputs.
        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): tuple (depth_results, features)
        """
        # for input, output in zip(inputs, outputs[0][0]):
        outputs = outputs[0]  # select results from (results, features)
        for i in range(len(inputs)):
            gt_depth = inputs[i]['depth'].unsqueeze(0)  # (1, 1, 64, 512)
            pred_depth = outputs[0][i].cpu().unsqueeze(0)  # (1, 1, 64, 512)
            pred_depth_oor = outputs[2][i].cpu().unsqueeze(0)  # (1, 1, 64, 512)

            mask_valid = gt_depth > 0
            mask_inr = mask_valid & (gt_depth < 0.96)

            # depth estimation performance
            l1_loss = F.l1_loss(pred_depth, gt_depth, reduction='none')[mask_inr]
            self._l1_mean.append(l1_loss.sum() / mask_inr.sum())
            self._l1_median.append(l1_loss.quantile(0.5))
            self._l1_80.append(l1_loss.quantile(0.8))
            self._l1_90.append(l1_loss.quantile(0.9))
            self._psnr.append(masked_psnr(pred_depth, gt_depth, mask_inr, data_range=1.0))
            self._ssim.append(masked_ssim(pred_depth, gt_depth, mask_inr, data_range=1.0))

            # in-range and out-of-range classification performance
            TP = torch.sum((pred_depth_oor < 0.5) & mask_inr)
            FP = torch.sum((pred_depth_oor < 0.5) & (gt_depth > 0.96))
            FN = torch.sum((pred_depth_oor > 0.5) & mask_inr)
            self._inr_precision.append(TP / (TP + FP))
            self._inr_recall.append(TP / (TP + FN))

    def evaluate(self):
        depth_res = {
            'depth_l1_mean': np.mean(self._l1_mean),
            'depth_l1_median': np.mean(self._l1_median),
            'depth_l1_80': np.mean(self._l1_80),
            'depth_l1_90': np.mean(self._l1_90),
            'depth_psnr': np.mean(self._psnr),
            'depth_ssim': np.mean(self._ssim),
            'cls_precision': np.mean(self._inr_precision),
            'cls_recall': np.mean(self._inr_recall),
        }

        results = OrderedDict({"depth": depth_res})
        self._logger.info(results)

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "depth_evaluation.pth")
            with open(file_path, "wb") as f:
                torch.save(results, f)

        return results
