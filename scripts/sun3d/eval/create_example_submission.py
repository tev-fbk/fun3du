"""
This script reads the ground truth (GT) data in the evaluation format and
generates an example submission that achieves a perfect score on the benchmark's validation set.
"""

import os
import sys

sys.path.append(os.getcwd())
import argparse

import numpy as np
from tqdm import tqdm

from scripts.sun3d.eval.eval_utils.rle import rle_encode


def main(gt_dir, write_dir):

    write_gt_dir_pred_masks = os.path.join(write_dir, "predicted_masks")

    os.makedirs(write_dir, exist_ok=True)
    os.makedirs(write_gt_dir_pred_masks, exist_ok=True)

    scene_names = sorted([el for el in os.listdir(gt_dir) if el.endswith(".txt")])

    for scene_name in tqdm(scene_names):
        scene_id = scene_name.split(".")[0]
        gt = np.loadtxt(os.path.join(gt_dir, scene_id + ".txt")).astype(int)
        obj_ids = np.unique(gt)
        lines = []
        for idx, obj_id in enumerate(obj_ids):
            if obj_id == 0:
                continue
            obj_mask = (gt == obj_id).astype(np.uint8)  # get mask for object

            if obj_id == 255:
                obj_mask = (gt == obj_id).astype(np.uint8)
                obj_id = obj_ids[idx - 1] + 1  # prev + 1

            rle_mask = rle_encode(obj_mask)
            obj_mask_path = os.path.join(
                write_gt_dir_pred_masks,
                scene_id + "_" + str(obj_id - 1).zfill(3) + ".txt",
            )
            with open(obj_mask_path, "w", encoding="utf-8") as file:
                file.write(rle_mask)
            lines.append(
                "predicted_masks/"
                + scene_id
                + "_"
                + str(obj_id - 1).zfill(3)
                + ".txt"
                + " 1.00"
            )

        scene_rel_file_write_path = os.path.join(write_dir, scene_id + ".txt")
        with open(scene_rel_file_write_path, "w") as f:
            f.write("\n".join(lines))
    print(f"Submissions saved to: {write_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gt_dir",
        help="Specify the directory where the ground-truth data is stored.",
        required=True,
    )

    parser.add_argument(
        "--write_dir",
        help="Specify the directory where you want to save your predictions in the submission format.",
        required=True,
    )

    args = parser.parse_args()

    main(args.gt_dir, args.write_dir)
