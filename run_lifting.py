import os
import sys

import torch

sys.path.append(os.getcwd())
from os.path import join

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from utils import io
from utils.misc import np_normalize, sort_alphanumeric, args2dict
from utils.sun3d.data_parser import DataParser
from utils.sun3d.fusion_util import PointCloudToImageMapper

def get_present_annots(path: str) -> dict:
    """
    Returns dict of present visits -> list of annot.
    To be used with the experiment path.
    """

    visit2desc = dict()
    for file in os.listdir(join(path, "pcds")):

        visit_id, desc_id = os.path.splitext(file)[0].split("_")
        if visit_id not in visit2desc.keys():
            visit2desc[visit_id] = list()
        visit2desc[visit_id].append(desc_id)

    return visit2desc


def post_process_pcd(
    parser: DataParser, pcd: torch.Tensor, mask_data: dict  # used for processing
) -> torch.Tensor:
    """
    Converts a point-cloud heatmap in a binary point cloud mask according to evaluation modality.
    """

    # get complete point cloud and mask
    n_points = pcd.shape[0]
    # prob functional mask
    acc_f = np.zeros((n_points), dtype=np.float16)
    c_f = np.zeros((n_points), dtype=np.uint8)

    dataset = zip(
        mask_data["frame_ids"],
        mask_data["video_ids"],
        mask_data["depth_paths"],
        mask_data["intrinsics"],
        mask_data["poses"],
        mask_data["masks_f"],
    )

    for i, (frame_id, video_id, depth_path, intrinsic, pose, mask_f) in enumerate(
        dataset
    ):

        depth = parser.read_depth_frame(depth_path)
        h, w = depth.shape
        whole_mask = np.ones(depth.shape)

        mapper = PointCloudToImageMapper((w, h))

        mapping_fo = mapper.compute_multi_masked_mapping(
            pose,
            pcd,
            np.stack([mask_f, whole_mask], axis=0),
            depth,
            intrinsic,
            "cuda",
        )
        valid_f = mapping_fo[0, :, -1] == 1

        if np.count_nonzero(valid_f) > 0:
            # compute F
            valid_fy = mapping_fo[0, valid_f, 0]
            valid_fx = mapping_fo[0, valid_f, 1]

            acc_f[valid_f] += mask_f[valid_fy, valid_fx]
            c_f[valid_f] += 1

    N_MASKS = mask_data["frame_ids"].shape[0]

    record = {"acc_f": acc_f, "c_f": c_f, "n_views": N_MASKS}

    return record


def get_visit_stuff(parser: DataParser, visit_id: str, videos: dict) -> dict:

    """
    For each visit returns a dict mapping video to data
    Data contains rgb frames, depth frames, camera intrinsics, camera poses
    """

    frames_data = dict()
    for video_id in videos:
        frames_data[video_id] = {
            "rgb_paths": parser.get_rgb_frames(visit_id, video_id),
            "depth_paths": parser.get_depth_frames(visit_id, video_id),
            "intrinsics": parser.get_camera_intrinsics(visit_id, video_id),
            "poses": parser.get_camera_trajectory(visit_id, video_id),
        }

    return frames_data


def get_prediction(
    exp: str,
    parser: DataParser,
    visit_id: str,
    frame_folder: str,
    desc_id: str,
    visit_data: dict,
) -> dict:
    """
    Reads prediction, and stuff needed for lifting
    """
    try:
        path = join("exps", exp, frame_folder, f"{visit_id}_{desc_id}.npz")
        data = np.load(path)
        if data["frame_ids"].shape[0] == 1:
            if data["frame_ids"][0] == 0:
                print("Nothing found in {} {}".format(visit_id, desc_id))
                return None

        all_mask_data = {k: data[k] for k in list(data.keys())}
        all_mask_data["depth_paths"] = list()
        all_mask_data["rgb_paths"] = list()
        all_mask_data["intrinsics"] = list()
        all_mask_data["poses"] = list()
    except:
        print(f"Error on {visit_id},{desc_id}")
        return None

    if "orig_dims" in all_mask_data.keys():

        new_f = list()

        for mask_f, orig_dim in zip(
            all_mask_data["masks_f"],
            all_mask_data["orig_dims"],
        ):
            if mask_f.shape[0] != orig_dim[0] or mask_f.shape[1] != orig_dim[1]:
                mask_f = torch.tensor(mask_f).unsqueeze(0).unsqueeze(0).to(torch.float)

                mask_f = (
                    torch.nn.functional.interpolate(
                        mask_f, (orig_dim[0], orig_dim[1]), mode="nearest"
                    )
                    .squeeze()
                    .to(torch.uint8)
                    .numpy()
                )

            new_f.append(mask_f)

        all_mask_data["masks_f"] = new_f

    for frame_id, video_id in zip(data["frame_ids"], data["video_ids"]):
        # read all the stuff
        all_mask_data["depth_paths"].append(
            visit_data[video_id]["depth_paths"][frame_id]
        )
        all_mask_data["rgb_paths"].append(visit_data[video_id]["rgb_paths"][frame_id])
        intrinsic = parser.read_camera_intrinsics(
            visit_data[video_id]["intrinsics"][frame_id], format="matrix"
        )
        pose = parser.get_nearest_pose(frame_id, visit_data[video_id]["poses"])
        all_mask_data["intrinsics"].append(intrinsic)
        all_mask_data["poses"].append(pose)

    return all_mask_data


def save_record(path: str, n_points: int, data: dict):

    if data is not None:
        np.savez_compressed(path, acc_f=data["acc_f"], n_views=data["n_views"])
    else:
        np.savez_compressed(path, acc_f=np.zeros(n_points), n_views=np.asarray([1]))


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run_lifting(args: DictConfig) -> dict:
    """
    Lifts the masks obtained in the previous step
    """
    arg_dict = args2dict(args)
    for k, v in arg_dict.items():
        print(f"{k} : {v}")

    parser = DataParser(args.dataset.root, args.dataset.split)
    visits = sort_alphanumeric(parser.get_visits())
    start = 0 if args.dataset.start is None else int(args.dataset.start)
    end = len(visits) if args.dataset.end is None else int(args.dataset.end)
    visit_ids = visits[start:end]

    print(
        f"Running multi-view agreement on {end-start} visits (split {args.dataset.split}), from {visit_ids[0]} to {visit_ids[-1]}"
    )
    visits2videos = io.get_visit_to_videos(args.dataset.root, args.dataset.split)
    os.makedirs(
        os.path.join(args.exp_root, args.exp_name, args.pcds_folder), exist_ok=True
    )

    # iterate over visits
    for visit_id in tqdm(visit_ids):

        # get cropped pcd and move it to np
        video_list = visits2videos[visit_id]
        pcd = parser.get_laser_scan(visit_id)
        pcd = parser.get_cropped_laser_scan(visit_id, pcd)

        scene_xyz = np.asarray(pcd.points)
        proc_pcd = torch.tensor(scene_xyz).cuda()
        scene_rgb = np.asarray(pcd.colors)
        full_pcd = np.concatenate([scene_xyz, scene_rgb], axis=1)

        desc_data = parser.get_descriptions_list(visit_id)

        # get data needed for lifting
        visit_data = get_visit_stuff(parser, visit_id, video_list)

        for desc_id in desc_data.keys():
            path = os.path.join(
                args.exp_root,
                args.exp_name,
                args.pcds_folder,
                f"{visit_id}_{desc_id}.npz",
            )

            if os.path.exists(path):
                print(
                    "Prediction {},{} exists in {}. Skipping.".format(
                        visit_id, desc_id, args.exp_name
                    )
                )
                continue

            # NB: predicted mask refer to CROPPED point cloud
            mask_data = get_prediction(
                args.exp_name, parser, visit_id, args.frame_folder, desc_id, visit_data
            )
            if mask_data is not None:
                pred_mask = post_process_pcd(parser, proc_pcd, mask_data)
            else:
                pred_mask = None
            save_record(path, full_pcd.shape[0], pred_mask)


if __name__ == "__main__":

    run_lifting()
