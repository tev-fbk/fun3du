import os
import sys

sys.path.append(os.getcwd())
import json
import pickle
from typing import List, Optional

import numpy as np

from utils.metrics import (
    compute_3d_ap,
    compute_3d_ar,
    compute_mean_iou,
    compute_mean_recalls,
    compute_scores,
)

from .metrics import *


class Segment3DEvaluator(object):
    """
    Helper class used to evaluate mask metrics
    """

    def __init__(self, exp_tag: str):

        super().__init__()
        self.exp_tag = exp_tag
        self.AP_TH = torch.linspace(0.5, 0.95, 10)
        self.metrics = {}

        self.metrics["visit_id"] = []
        self.metrics["annot_id"] = []
        self.metrics["pred_count"] = []
        self.metrics["Prc"] = []
        self.metrics["mAP"] = []
        self.metrics["AP25"] = []
        self.metrics["AP50"] = []
        self.metrics["Rec"] = []
        self.metrics["mAR"] = []
        self.metrics["AR25"] = []
        self.metrics["AR50"] = []
        self.metrics["mIoU"] = []

    def register(
        self,
        visit_ids: List[str],
        annot_ids: List[str],
        gt_masks: Tensor,
        pred_masks: Tensor,
    ):
        """
        Register metrics mAP,AP50,AP25,mAR,AR50,AR25
        """

        assert (
            gt_masks.shape == pred_masks.shape
        ), f"Shapes not correponding: {gt_masks.shape} ad {pred_masks.shape}"

        # beacuse function only processes lists
        if len(gt_masks.shape) == 1:
            gt_masks = gt_masks.unsqueeze(0)
            pred_masks = pred_masks.unsqueeze(0)

        # compute ap and relative recalls
        valid_pred = [torch.count_nonzero(pred_mask).item() for pred_mask in pred_masks]
        ap_i = compute_3d_ap(gt_masks, pred_masks)
        ious = compute_mean_iou(gt_masks, pred_masks)
        map = compute_mean_recalls(ap_i, self.AP_TH)
        ap_rec = compute_scores(ap_i, [0.25, 0.50])
        ap_50 = ap_rec[0.50]
        ap_25 = ap_rec[0.25]

        # compute ar and relative recalls
        ar_i = compute_3d_ar(gt_masks, pred_masks)
        mar = compute_mean_recalls(ar_i, self.AP_TH)
        ar_rec = compute_scores(ar_i, [0.25, 0.50])
        ar_50 = ar_rec[0.50]
        ar_25 = ar_rec[0.25]
        self.metrics["visit_id"].extend(visit_ids)
        self.metrics["annot_id"].extend(annot_ids)
        self.metrics["pred_count"].extend(valid_pred)

        self.metrics["Prc"].extend(list(ap_i.numpy().astype(np.float64)))
        self.metrics["mAP"].extend(list(map.numpy().astype(np.float64)))
        self.metrics["AP50"].extend(list(ap_50.numpy().astype(np.float64)))
        self.metrics["AP25"].extend(list(ap_25.numpy().astype(np.float64)))

        self.metrics["Rec"].extend(list(ar_i.numpy().astype(np.float64)))
        self.metrics["mAR"].extend(list(mar.numpy().astype(np.float64)))
        self.metrics["AR50"].extend(list(ar_50.numpy().astype(np.float64)))
        self.metrics["AR25"].extend(list(ar_25.numpy().astype(np.float64)))

        self.metrics["mIoU"].extend(list(ious.numpy().astype(np.float64)))

    def save(self, file):
        to_save = dict()
        for k, v in self.metrics.items():
            to_save[k] = v  # assumes is a list already
        json.dump(to_save, file)

    def get_means(self):
        """
        Returns mean of each metric registered so far
        """
        means = {}
        for name, value in self.metrics.items():
            if name not in ["visit_id", "annot_id"]:
                mean = np.asarray(value).mean()
                means[name] = mean

        return means

    def get_latex_str(self):
        """
        Returns mean of each metric in format for a latex table
        """

        means = self.get_means()

        latex_str = f"{self.exp_tag} & {means['mAP']*100:.2f} & {means['AP50']*100:.2f} & {means['AP25']*100:.2f} & {means['mAR']*100:.2f} & {means['AR50']*100:.2f} & {means['AR25']*100:.2f} & {means['mIoU']*100:.2f} \\\\ \n"

        return latex_str
