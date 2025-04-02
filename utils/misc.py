import os
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor


def np_normalize(array: np.ndarray) -> np.ndarray:
    """
    Normalizes across first axis probably
    """

    amin, amax = array.min(), array.max()
    m_array = array.copy()

    return (m_array - amin) / (amax - amin)


def boxes2mask(boxes: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Given a list of boxes (xyxy) and the image size in shape (h,w), paints the binary mask
    """
    h, w = size
    mask = np.zeros((h, w))
    for box in boxes:
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = 1

    return mask


def args2dict(config: DictConfig, init_key=None) -> dict:
    """
    Recursively converts an Hydra configuration file in a one-level dictionary
    """
    cur_dict = {}
    for k, v in config.items():

        if isinstance(v, DictConfig):
            if init_key:
                sub_dict = args2dict(v, init_key=f"{init_key}.{k}")
            else:
                sub_dict = args2dict(v, init_key=k)
            cur_dict.update(sub_dict)
        else:
            if init_key:
                cur_dict[f"{init_key}.{k}"] = v
            else:
                cur_dict[k] = v

    return cur_dict


def sort_alphanumeric(lst: str):
    def sort_key(s):
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split("([0-9]+)", s)
        ]

    return sorted(lst, key=sort_key)


def mask_idxs2bin(mask: Tensor) -> Tensor:
    """
    Given a mask with integer values, transforms it in a set of binary masks.
    Only makes sense if the indexes are integet but continuous.
    """

    idxs = torch.unique(mask)
    mask_list = torch.zeros((idxs.shape[0], *mask.shape))
    for i, idx in enumerate(idxs):
        mask_list[i, mask == idx] = 1
    return mask_list


def mask_to_box(mask: np.array) -> np.array:
    """
    Given a mask, return the respective bounding box, considering nonzero indexes
    Box is returned in format YXHW
    """
    idxs_y, idxs_x = np.nonzero(mask)
    # print(mask_idxs.shape)
    max_y, min_y = idxs_y.max(), idxs_y.min()
    max_x, min_x = idxs_x.max(), idxs_x.min()
    # print(max_y,min_y,max_x,min_x)
    box = np.asarray([min_y, min_x, (max_y - min_y), (max_x - min_x)])

    return box


def square_bbox(box: np.array, ratio: Optional[int] = None) -> np.array:
    """
    Computes squared bounding box, in format YXHW
    """

    y, x, h, w = box
    h, w = max(h, 2), max(w, 2)
    # get centers
    cx, cy = x + w / 2, y + h / 2
    # recompute first point
    new_h = max(h, 2)
    new_w = max(w, 2)
    if ratio is not None:
        new_h, new_w = new_h * ratio, new_w * ratio
    x = cx - new_w / 2
    y = cy - new_h / 2

    # square bbox
    max_dim = max(new_h, new_w)
    # recompute if needed
    if new_w < max_dim:
        extra_w = max_dim - new_w
        x = x - extra_w / 2.0

    elif new_h < max_dim:
        extra_h = max_dim - new_h
        y = y - extra_h / 2.0

    return np.asarray([int(y), int(x), int(max_dim), int(max_dim)])


def set_seed(seed: int):
    print("SETTING SEED: ", seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def make_frame_data(
    frames: dict,
    mask_list: List,
    masks_n: List,
    score_list: List,
    point_list: List,
    args: DictConfig,
) -> dict:
    """
    Uniforms the format of frames data given by molmo
    """

    new_format = {
        "frame_ids": [],
        "video_ids": [],
        "depth_paths": [],
        "poses": [],
        "intrinsics": [],
        "func_masks": [],
        "func_scores": [],
        "points": [],
        "parent_masks": [],
        "parent_scores": [],
    }

    dataset = zip(
        mask_list,
        score_list,
        point_list,
        masks_n,
        frames["masks"],
        frames["scores"],
        frames["frame_ids"],
        frames["video_ids"],
        frames["depth_paths"],
        frames["intrinsics"],
        frames["poses"],
    )

    for (
        func_masks_i,
        func_scores_i,
        func_points_i,
        func_mask_n,
        parent_mask,
        parent_score,
        frame_id,
        video_id,
        depth_path,
        intrinsic,
        pose,
    ) in dataset:

        # at least one func_mask per parent_mask
        if func_mask_n > 0:
            for i in range(func_mask_n):
                # compute molmo score accordingly
                new_format["func_scores"].append(func_scores_i[i])
                # must append a record for each.
                new_format["func_masks"].append(func_masks_i[i])
                new_format["points"].append(func_points_i[i])
                new_format["depth_paths"].append(depth_path)
                new_format["frame_ids"].append(frame_id)
                new_format["video_ids"].append(video_id)
                new_format["intrinsics"].append(intrinsic)
                new_format["poses"].append(pose)
                new_format["parent_masks"].append(parent_mask)
                new_format["parent_scores"].append(parent_score)

        # we have a parent mask, but no molmo frame
        else:
            new_format["func_masks"].append(np.zeros(parent_mask.shape))
            new_format["func_scores"].append(0)
            new_format["points"].append(np.asarray([0, 0]))
            new_format["depth_paths"].append(depth_path)
            new_format["frame_ids"].append(frame_id)
            new_format["video_ids"].append(video_id)
            new_format["intrinsics"].append(intrinsic)
            new_format["poses"].append(pose)
            new_format["parent_masks"].append(parent_mask)
            new_format["parent_scores"].append(parent_score)

    for k, v in new_format.items():
        # mask cannot be a np array, because they can have different shapes
        if "masks" not in k:
            new_format[k] = np.asarray(v)

    return new_format
