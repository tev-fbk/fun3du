import os
import sys

sys.path.append(os.getcwd())
from os.path import join
from typing import List, Tuple, Union

import numpy as np
from numpy import ndarray

from utils.misc import np_normalize
from utils.sun3d.data_parser import DataParser


def get_prompt_data(visit_id: str, desc_id: str, llm_annot: dict) -> Tuple[str, str]:
    """
    Returns context and functional object
    """

    context_object, func_object = None, None  # these are top object and functional
    try:
        context_object = llm_annot["acted_on_object_hierarchy"][0].lower()
        func_object = llm_annot["acted_on_object"].lower()

    except:
        print(f"No contextual or func object for {visit_id},{desc_id}.")

    return context_object, func_object


def sample_scored_frames(
    parser: DataParser,
    visit_id: str,
    obj: str,
    n_samples: int,
    modality: str,
    mask_type: str,
) -> dict:

    """
    Samples N images across all views for a visit with a specified object.
    """

    (
        cand_videos,
        cand_frames,
        all_masks,
        all_scores_owl,
        all_scores_mod,
        all_scores_angle,
    ) = filter_scored_masks(parser, visit_id, mask_type, obj)

    if cand_videos.shape[0] == 0:
        return {
            "rgb_paths": np.asarray([]),
            "depth_paths": np.asarray([]),
            "video_ids": np.asarray([]),
            "frame_ids": np.asarray([]),
            "intrinsics": np.asarray([]),
            "poses": np.asarray([]),
            "masks": [],
            "scores_owl": np.asarray([]),
            "scores_mod": np.asarray([]),
            "scores_angle": np.asarray([]),
            "scores": np.asarray([]),
        }

    # -1 because these are actually distances.
    all_scores_mod = 1 - np_normalize(all_scores_mod)
    all_scores_angle = 1 - np_normalize(all_scores_angle)
    all_scores_mask = 0.5 * (all_scores_mod + all_scores_angle)
    choosen_score = 0.5 * (
        all_scores_owl + 0.25 * all_scores_mod + 0.25 * all_scores_angle
    )

    sampling = False
    # print(cand_videos,cand_frames,cand_boxes)
    if modality == "random":
        sampling = True
        actual_samples = min(cand_videos.shape[0], n_samples)
        sub_idxs = np.random.choice(
            np.arange(cand_videos.shape[0]), actual_samples, replace=False
        )
        cand_videos = cand_videos[sub_idxs]
        cand_frames = cand_frames[sub_idxs]
        cand_scores_owl = all_scores_owl[sub_idxs]
        cand_scores_mod = all_scores_mod[sub_idxs]
        cand_scores_angle = all_scores_angle[sub_idxs]

    elif "score" in modality:
        sampling = True
        actual_samples = min(cand_videos.shape[0], n_samples)
        if modality == "score_mean":
            choosen_score = 0.5 * (all_scores_mask + all_scores_owl)
        elif modality == "score_mask":
            choosen_score = all_scores_mask
        elif modality == "score_mod":
            choosen_score = all_scores_mod
        elif modality == "score_angle":
            choosen_score = all_scores_angle
        elif modality == "score_owl":
            choosen_score = all_scores_owl
        else:
            raise RuntimeError("Score of type {} not supported.".format(modality))

        sub_idxs = np.argsort(-choosen_score)[:n_samples]
        choosen_score = choosen_score
        cand_videos = cand_videos[sub_idxs]
        cand_frames = cand_frames[sub_idxs]
        cand_scores_owl = all_scores_owl[sub_idxs]
        cand_scores_mod = all_scores_mod[sub_idxs]
        cand_scores_angle = all_scores_angle[sub_idxs]

    else:
        sampling = False

    # list of selected samples
    sel_rgb, sel_depth, sel_intrinsics, sel_poses = (list(), list(), list(), list())

    # dictionaries of data for each video
    poses_t, rgb_paths_t, depth_paths_t, intrinsics_t = (dict(), dict(), dict(), dict())
    for video_id in np.unique(cand_videos):

        # save video-related information in dictionary
        poses_t[video_id] = parser.get_camera_trajectory(visit_id, video_id)
        rgb_paths_t[video_id] = parser.get_rgb_frames(visit_id, video_id)
        depth_paths_t[video_id] = parser.get_depth_frames(visit_id, video_id)
        intrinsics_t[video_id] = parser.get_camera_intrinsics(visit_id, video_id)

    for frame_id, video_id in zip(cand_frames, cand_videos):

        sel_rgb.append(rgb_paths_t[video_id][frame_id])
        sel_depth.append(depth_paths_t[video_id][frame_id])
        intrinsic = parser.read_camera_intrinsics(
            intrinsics_t[video_id][frame_id], format="matrix"
        )
        sel_intrinsics.append(intrinsic)
        sel_poses.append(parser.get_nearest_pose(frame_id, poses_t[video_id]))

    if sampling:
        cand_masks = [all_masks[idx] for idx in sub_idxs]
    else:
        cand_masks = all_masks

    # necessary to get all boxes

    return {
        "rgb_paths": np.asarray(sel_rgb),
        "depth_paths": np.asarray(sel_depth),
        "video_ids": cand_videos,
        "frame_ids": cand_frames,
        "intrinsics": np.asarray(sel_intrinsics),
        "poses": np.asarray(sel_poses),
        "masks": cand_masks,
        "scores_owl": cand_scores_owl,
        "scores_mod": cand_scores_mod,
        "scores_angle": cand_scores_angle,
        "scores": choosen_score[sub_idxs],
    }


def filter_scored_masks(
    parser: DataParser, visit_id: str, mask_type: str, query_object: str
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Filters scored masks by only retaining frames with a specific object
    """
    video_ids, frame_ids, masks, scores_conf, scores_mod, scores_angle = (
        list(),
        list(),
        list(),
        list(),
        list(),
        list(),
    )
    query_object = query_object.lower()
    mask_data = parser.get_mask_index(visit_id, mask_type)

    if query_object in mask_data["objects"].keys():
        obj_set = set(mask_data["objects"][query_object])

        # list of video_id/frame_id pairw with the given object with the given description
        video_frames_ids = list(obj_set)

        for video_frame_id in video_frames_ids:

            video_id, frame_id = video_frame_id.split(" ")
            masks_i, scores_owl_i, scores_angle_i, scores_mod_i = parser.get_pred_mask(
                visit_id, video_id, mask_type, frame_id, query_object
            )
            # if this happened, the masked index is wrong
            assert masks_i.shape[0] > 0, scores_owl_i.shape[0] > 0
            for mask_i, score_owl_i, score_angle_i, score_mod_i in zip(
                masks_i, scores_owl_i, scores_angle_i, scores_mod_i
            ):
                video_ids.append(video_id)
                frame_ids.append(frame_id)
                masks.append(mask_i)
                scores_conf.append(score_owl_i)
                scores_angle.append(score_angle_i)
                scores_mod.append(score_mod_i)

    video_ids = np.asarray(video_ids)
    frame_ids = np.asarray(frame_ids)
    scores_conf = np.asarray(scores_conf)
    scores_angle = np.asarray(scores_angle)
    scores_mod = np.asarray(scores_mod)

    return video_ids, frame_ids, masks, scores_conf, scores_mod, scores_angle


def get_visit_to_videos(root: str, split: str) -> dict:
    """
    Given a split, returns a dict associating each visit id to the list of video ids
    """

    visit_to_videos = dict()

    with open(join(root, f"benchmark_file_lists/{split}_set.csv")) as f:
        # skip csv header
        visit_video = f.readlines()[1:]

    for line in visit_video:
        visit_id, video_id = line.strip("\n").split(",")
        if visit_id not in visit_to_videos.keys():
            visit_to_videos[visit_id] = list()
        visit_to_videos[visit_id].append(video_id)

    return visit_to_videos


def pad_detection_predictions(
    boxes: List, scores: List, labels: List
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    List of lists of GDino prediction for each frame.
    Pads them to the largest value. Non-valid detections are set with scores=-1
    """

    assert len(boxes) == len(scores) and len(scores) == len(labels)
    num_frames = len(boxes)
    max_boxes = max([len(frame_boxes) for frame_boxes in boxes])

    new_boxes = np.zeros((num_frames, max_boxes, 4))
    new_scores = np.zeros((num_frames, max_boxes))
    new_labels = list()
    for i, (boxes_i, scores_i, labels_i) in enumerate(zip(boxes, scores, labels)):
        assert len(boxes_i) == len(scores_i) and len(scores_i) == len(labels_i)

        if len(labels_i) < max_boxes:
            for _ in range(max_boxes - len(labels_i)):
                labels_i.append("empty")

        boxes_i, scores_i, labels_i = (
            np.asarray(boxes_i),
            np.asarray(scores_i),
            np.asarray(labels_i),
        )
        num_boxes = boxes_i.shape[0]
        new_boxes[i, :num_boxes] = boxes_i
        new_scores[i, :num_boxes] = scores_i
        new_labels.append(labels_i)

    new_labels = np.asarray(new_labels)
    return new_boxes.astype(np.uint16), new_scores, new_labels
