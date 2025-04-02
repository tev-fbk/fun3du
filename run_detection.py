import os
import sys

import hydra
from omegaconf import DictConfig

sys.path.append(os.getcwd())
import argparse
import json
import math
from os.path import join
from typing import List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from utils import io
from utils.hf_models import init_detection, process_detection
from utils.metrics import get_mask_score
from utils.misc import sort_alphanumeric
from utils.sun3d.data_parser import DataParser


def pad_into_array(strs: List[List[str]]) -> np.ndarray:

    "Pads a list of list of strings into an array of strings"

    max_l = max([len(l) for l in strs])
    new_strs = list()
    for str in strs:
        new_str = str.copy()
        if len(new_str) < max_l:
            new_str.extend(["nan" for _ in range(max_l - len(new_str))])
        new_strs.append(new_str)

    return np.asarray(new_strs)


def make_mask_index(args: DictConfig):
    """
    Makes a json file with mask data, for faster access when looking for a specific object.
    """

    root, split = args.dataset.root, args.dataset.split
    start, end = args.dataset.start, args.dataset.end
    parser = DataParser(root, split)

    visits2videos = io.get_visit_to_videos(root, split)
    visits = sort_alphanumeric(list(visits2videos.keys()))

    start = 0 if args.dataset.start is None else int(args.dataset.start)
    end = len(visits) if args.dataset.end is None else int(args.dataset.end)
    visit_ids = visits[start:end]

    print(
        f"Processing {end-start} visits (split {args.dataset.split}), from {visit_ids[0]} to {visit_ids[-1]}"
    )


    for visit_id in tqdm(visit_ids):
        video_list = io.get_visit_to_videos(root, split)[visit_id]
        visit_dict = {"desc_ids": {}, "objects": {}}

        # desc_data = parser.get_descriptions(visit_id)
        # llm_data = parser.get_llm_data(visit_id)
        for video_id in video_list:
            frame_ids = list(parser.get_rgb_frames(visit_id, video_id).keys())

            for frame_id in frame_ids:

                try:
                    mask_data = parser.read_owl_mask(
                        visit_id, video_id, args.mask_type, frame_id
                    )
                except:
                    # print("Not found")
                    continue

                cur_key = f"{video_id} {frame_id}"
                for label, desc_list, mask in zip(
                    mask_data["labels"], mask_data["desc_ids"], mask_data["masks"]
                ):
                    if np.count_nonzero(mask) == 0:
                        continue
                    # work on label presence
                    if label not in visit_dict["objects"].keys():
                        visit_dict["objects"][label] = list()
                    if cur_key not in visit_dict["objects"][label]:
                        visit_dict["objects"][label].append(cur_key)
                    # work on desc_list presence
                    for desc_id in desc_list:
                        if desc_id != "nan":
                            if desc_id not in visit_dict["desc_ids"].keys():
                                visit_dict["desc_ids"][desc_id] = list()
                            if cur_key not in visit_dict["desc_ids"][desc_id]:
                                visit_dict["desc_ids"][desc_id].append(cur_key)

        with open(
            os.path.join(
                root, split, visit_id, f"{visit_id}_{args.mask_type}_masks.json"
            ),
            "w",
        ) as f:
            json.dump(visit_dict, f)


def run_owl2_rsam(args: argparse.Namespace, bs: int = 20):
    """
    Runs detection on a set of visits, looking for the objects mentioned in the LLM-extracted data
    Saves masks and related info (e.g., scores for retrieval)
    """
    root, split = args.dataset.root, args.dataset.split
    device = "cuda" if torch.cuda.is_available() else "cpu"

    owl_m, owl_p, sam_m, sam_p = init_detection(device)

    visits2videos = io.get_visit_to_videos(root, split)
    visits = sort_alphanumeric(list(visits2videos.keys()))

    start = 0 if args.dataset.start is None else int(args.dataset.start)
    end = len(visits) if args.dataset.end is None else int(args.dataset.end)
    visits = visits[start:end]

    print(
        f"Running detection for {end-start} visits (split {args.dataset.split}), from {visits[0]} to {visits[-1]}"
    )
    parser = DataParser(root, split)

    # iterate over visits
    for visit_id in tqdm(visits):

        print("Processing ", visit_id)
        visit_masks = dict()
        video_list = visits2videos[visit_id]

        llm_prompts = parser.get_llm_prompts(visit_id, args.llm_type)
        prompts = [p.lower() for p in list(llm_prompts.keys())]
        # iterate over videos in each visit
        for video_id in video_list:

            os.mkdir(
                join(
                    args.dataset.root,
                    args.dataset.split,
                    visit_id,
                    video_id,
                    args.mask_type,
                )
            )

            visit_masks[video_id] = dict()

            rgb_data = parser.get_rgb_frames(visit_id, video_id)
            frame_ids = list(rgb_data.keys())
            rgb_paths = list(rgb_data.values())

            NB = math.ceil(len(frame_ids) / bs)
            # iterate over batches of images
            for i_b in range(NB):
                paths_i = rgb_paths[i_b * bs : (i_b + 1) * bs]
                frames_i = frame_ids[i_b * bs : (i_b + 1) * bs]

                # get batch ready
                imgs = [Image.open(path) for path in paths_i]

                all_imgs_data = process_detection(
                    owl_m, owl_p, sam_m, sam_p, imgs, prompts
                )

                # iterate over data about images
                for frame_id, frame_data in zip(frames_i, all_imgs_data):
                    mask_desc_ids = list()

                    # if not, than this is an empty mask, thus not saved!
                    if isinstance(frame_data["labels"], np.ndarray):
                        mod_scores = np.zeros(frame_data["scores"].shape)
                        angle_scores = np.zeros(frame_data["scores"].shape)

                        for i, (label, mask) in enumerate(
                            zip(frame_data["labels"], frame_data["masks"])
                        ):
                            mask = torch.tensor(mask)
                            angle_scores[i], mod_scores[i] = get_mask_score(mask)

                            if label in llm_prompts.keys():
                                # appending the LIST of desc_ids
                                mask_desc_ids.append(llm_prompts[label])
                        frame_data["desc_ids"] = pad_into_array(mask_desc_ids)
                        path = join(
                            args.dataset.root,
                            args.dataset.split,
                            visit_id,
                            video_id,
                            args.mask_type,
                            f"{video_id}_{frame_id}.npz",
                        )
                        np.savez_compressed(
                            path,
                            masks=frame_data["masks"],
                            labels=frame_data["labels"],
                            scores=frame_data["scores"],
                            mod_scores=mod_scores,
                            angle_scores=angle_scores,
                            desc_ids=frame_data["desc_ids"],
                        )
                    else:
                        print(
                            "Empty frame {},{},{}".format(visit_id, video_id, frame_id)
                        )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(args: DictConfig):

    run_owl2_rsam(args, bs=10)
    make_mask_index(args)


if __name__ == "__main__":
    main()
