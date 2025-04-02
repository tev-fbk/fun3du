# Based on https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts
# Adapted, July 2023

import os

import numpy as np
from benchmark_scripts.eval_utils.eval_script_inst import evaluate


def main(pred_dir, gt_dir):
    scene_names = sorted(el[:-4] for el in os.listdir(gt_dir) if el.endswith(".txt"))

    preds = {}
    for scene_name in scene_names[:]:

        # Load predictions based on the requested submission format
        file_path = os.path.join(pred_dir, scene_name + ".txt")  # {SCENE_ID}.txt file
        scene_pred_mask_list = np.loadtxt(file_path, dtype=str)  # (num_masks, 2)
        assert (
            scene_pred_mask_list.shape[1] == 2
        ), "Each line should have 2 values: instance mask path and confidence score."

        pred_masks = []
        pred_scores = []
        for mask_path, conf_score in scene_pred_mask_list:
            # Read mask and confidence score for each instance mask
            pred_mask = np.loadtxt(
                os.path.join(pred_dir, mask_path), dtype=int
            )  # Values: 0 for the background, 1 for the instance
            pred_masks.append(pred_mask)
            pred_scores.append(float(conf_score))

        assert len(pred_masks) == len(
            pred_scores
        ), "Number of masks and confidence scores should be the same."

        # Aggregate masks and scores for each scene - pred_class is always 1 (we only have one semantic class, 'object', referring to the query object)
        preds[scene_name] = {
            "pred_masks": (
                np.vstack(pred_masks).T if len(pred_masks) > 0 else np.zeros((1, 1))
            ),
            "pred_scores": (
                np.vstack(pred_scores) if len(pred_masks) > 0 else np.zeros(1)
            ),
            "pred_classes": (
                np.ones(len(pred_masks), dtype=np.int64)
                if len(pred_masks) > 0
                else np.ones(1, dtype=np.int64)
            ),
        }

    # Run evaluation script
    ap_dict = evaluate(preds, gt_dir)
    del ap_dict["classes"]
    print(ap_dict)


if __name__ == "__main__":
    pred_dir = "PATH/TO/RESULTS"  # Predictions in the submission format, containing '<SCENE_ID>.txt' files and 'predicted_masks' folder including all masks
    gt_dir = "PATH/TO/GT"  # Folder containing <SCENE_ID>.txt files for gt annotations, see https://github.com/OpenSun3D/OpenSun3D.github.io/tree/main/challenge/benchmark_data/gt_development_scenes
    main(pred_dir, gt_dir)
