import math
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import entropy
from torch import Tensor
from torch.nn.functional import cosine_similarity


def get_image_polar_coords(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make polar coordinates mask with a fixed size
    """
    h, w = shape
    xs, ys = np.meshgrid(np.arange(0, w, 1), np.arange(0, h, 1))
    xs = (xs - int(w / 2)) / (w / 2)
    ys = (ys - int(h / 2)) / (h / 2)

    arctans = np.arctan2(ys, xs)
    module = np.sqrt(np.power(xs, 2) + np.power(ys, 2))

    return torch.tensor(module), torch.tensor(arctans)


M_1920_1440, A_1920_1440 = get_image_polar_coords((1920, 1440))
M_1440_1920, A_1440_1920 = get_image_polar_coords((1440, 1920))


def get_mask_score(mask: np.ndarray, n_bins: Optional[int] = 30) -> Tuple[float, float]:
    """
    Compute polar coordinate score for the mask
    """
    if mask.shape == (1920, 1440):
        module, arctans = M_1920_1440, A_1920_1440
    elif mask.shape == (1440, 1920):
        module, arctans = M_1440_1920, A_1440_1920
    else:
        print("Must compute for shape ", mask.shape)
        module, arctans = get_image_polar_coords(mask.shape)

    if torch.count_nonzero(mask) > 0:

        m_arctans = arctans[mask == 1]
        m_mod = module[mask == 1]

        hist_arc, _ = torch.histogram(
            m_arctans, bins=n_bins, range=(-torch.pi, torch.pi)
        )
        hist_mod, bins_mod = torch.histogram(
            m_mod, bins=n_bins, range=(0, math.sqrt(2))
        )
        arc_dist = torch.ones(n_bins)

        mod_dist = torch.zeros(n_bins)
        max_mod = torch.max(m_mod)
        max_bin = torch.min(torch.nonzero(max_mod <= bins_mod)[:, 0])
        # get the bin before that, if it isnt the last
        mod_dist[0:max_bin] = 1
        arc_score = entropy(hist_arc, arc_dist)
        mod_score = entropy(hist_mod, mod_dist)
    else:
        arc_score, mod_score = 0.0, 0.0

    return arc_score, mod_score


def multimask_to_idxs(masks: torch.Tensor) -> List[set]:
    mask_idxs = torch.unique(masks)
    idxs = list()
    for mask_idx in mask_idxs:
        idx_i = set(torch.nonzero(masks == mask_idx).squeeze(1).tolist())
        idxs.append(idx_i)

    return idxs


def masks_to_idxs(masks: torch.Tensor) -> List[set]:
    idxs = list()
    for mask in masks:
        idx_i = set(torch.nonzero(mask).squeeze(1).tolist())
        idxs.append(idx_i)

    return idxs


def one_to_multimask_iou_idx(idx: set, masks: torch.Tensor):
    """
    IoU between a mask and multiple masks, the latter is a spatial segmentation mask with multiple indexes
    """
    idxs = multimask_to_idxs(masks)
    ious = one_to_many_iou_idx(idx, idxs)

    return ious


def many_to_many_iou_idx(idxs2: List[set], idxs1: List[set]) -> torch.Tensor:

    """
    IoU between a mask and multiple masks, all specified as set of indexes
    """

    iou_list = np.zeros((len(idxs1), len(idxs2)))

    for i, idx1 in enumerate(idxs1):
        for j, idx2 in enumerate(idxs2):

            inter = idx1.intersection(idx2)
            union = idx1.union(idx2)
            iou = len(inter) / len(union) if len(union) > 0 else 0.0
            iou_list[i, j] = iou

    return torch.tensor(iou_list)


def one_to_many_iou_idx(idx1: set, idxs: List[set]):

    """
    IoU between a mask and multiple masks, all specified as set of indexes
    """

    iou_list = list()

    for idx in idxs:

        inter = idx1.intersection(idx)
        union = idx1.union(idx)
        iou = len(inter) / len(union) if len(union) > 0 else 0.0
        iou_list.append(iou)

    return torch.tensor(iou_list)


def cross_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Given two vectors [N1,D] and [N2,D], returns [N1,N2] cosine similarity matrix
    """

    N1, N2 = x1.shape[0], x2.shape[0]
    assert x1.shape[1] == x2.shape[1]

    c_matrix = torch.zeros((N1, N2))
    for i in range(N1):
        c_matrix[i, :] = cosine_similarity(x2, x1[i, :])

    return c_matrix


def one_to_many_iou(mask1: Tensor, mask2: Tensor) -> Tensor:
    """
    Compute IoU among all pairs (maybe too much memory)
    """

    inter = mask2.unsqueeze(1) * mask1.unsqueeze(0)
    union = mask2.unsqueeze(1) + mask1.unsqueeze(0) - inter
    ious = inter.sum(dim=-1) / union.sum(dim=-1)

    return ious


def compute_mean_iou(gt_mask: Tensor, pred_mask: Tensor) -> Tuple[Tensor]:
    """
    Given N ground-truth masks and N pred_masks, returns resulting mIoUs
    """

    inter = pred_mask * gt_mask
    union = pred_mask + gt_mask - inter
    ious = inter.sum(dim=-1) / union.sum(dim=-1)

    ious = torch.where(torch.isnan(ious), 0.0, ious)
    return ious


def compute_mean_recalls(values: Tensor, thresholds: Tensor) -> Tensor:
    """
    Given a tensor of N scores, and a tensor of M thresholds (assumed incremental), the average recall with varying thresholds is returned for each of the N sample.
    """

    matrix = (values.unsqueeze(1) >= thresholds.unsqueeze(0)).to(torch.float)
    return matrix.mean(dim=1)


def compute_scores(values: Tensor, thresholds: Optional[list] = [0.25, 0.5]) -> dict:
    """
    Given a list of N scores, computes recall based on a list of M thresholds.
    Results is a dictionary with M keys, each with a list of N positive/negative score
    """

    recalls = dict()
    for th in thresholds:
        recalls[th] = (values >= th).to(torch.float)
    return recalls


def compute_3d_ap(gt_mask: Tensor, pred_mask: Tensor) -> Tensor:
    """
    Return precision of point clouds (can be B,N or N)
    Only binary are allowed
    """

    assert gt_mask.shape == pred_mask.shape, " Different shape!"

    if len(gt_mask.shape) == 1:
        gt_mask = gt_mask.unsqueeze(0)
        pred_mask = pred_mask.unsqueeze(0)
        to_squeeze = True
    else:
        to_squeeze = False

    tp = torch.logical_and(gt_mask, pred_mask).to(torch.uint8)
    positives = pred_mask.count_nonzero(dim=1)
    precision = torch.where(positives > 0, tp.count_nonzero(dim=1) / positives, 0)

    if to_squeeze:
        precision = precision.squeeze(0)

    return precision


def compute_3d_ar(gt_mask: Tensor, pred_mask: Tensor) -> Tensor:
    """
    Return recall of point clouds (can be B,N or N)
    Only binary are allowed
    """

    assert gt_mask.shape == pred_mask.shape, " Different shape!"

    if len(gt_mask.shape) == 1:
        gt_mask = gt_mask.unsqueeze(0)
        pred_mask = pred_mask.unsqueeze(0)
        to_squeeze = True
    else:
        to_squeeze = False

    tp = torch.logical_and(gt_mask, pred_mask).to(torch.uint8)
    positives = gt_mask.count_nonzero(dim=1)
    recall = torch.where(positives > 0, tp.count_nonzero(dim=1) / positives, 0)

    if to_squeeze:
        recall = recall.squeeze(0)

    return recall
