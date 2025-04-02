"""
Projects the RGB color of the iPad camera frames on the 3D points of the laser scan

This example script demonstrates how to utilize the data assets and helper scripts in the SceneFun3D Toolkit.
"""

import os
import sys

sys.path.append(os.getcwd())
import argparse
import os

import numpy as np
import open3d as o3d
import pandas as pd
from plyfile import PlyData, PlyProperty
from tqdm import tqdm

from utils.data_parser import DataParser
from utils.fusion_util import PointCloudToImageMapper
from utils.pc_process import pc_estimate_normals
from utils.viz import viz_3d

##################
### PARAMETERS ###
##################
use_interpolation = True
time_distance_threshold = 0.2
frame_distance_threshold = np.inf  # 0.1
visibility_threshold = 0.25
cut_bound = 5

vis_result = True
##################
##################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split", choices=["dev", "test"], help="Specify the split of the data"
    )

    parser.add_argument(
        "--data_dir", default="data", help="Specify the path of the data"
    )

    parser.add_argument(
        "--video_id_csv",
        help="Specify the path of the .csv file which contains the {visit_id, video_id} list",
    )

    parser.add_argument(
        "--continue_video_id",
        default="0",
        help="Specify the video_id to start processing from in the video_id_csv",
    )

    parser.add_argument(
        "--coloring_asset",
        default="wide",
        choices=["wide", "lowres_wide"],
        help="Specify the RGB data asset to use for projecting the color to the laser scan",
    )

    parser.add_argument(
        "--crop_extraneous",
        action="store_true",
        help="Specify whether to crop the extraneous points from the laser scan",
    )

    parser.add_argument(
        "--save_as_float32",
        action="store_true",
        help="Specify whether to store the output point cloud",
    )

    args = parser.parse_args()

    assert args.video_id_csv is not None, "video_id_csv must be specified"

    continue_visit_id = int(args.continue_video_id)

    split = args.split

    use_interpolation = True if args.coloring_asset == "wide" else False

    df = pd.read_csv(args.video_id_csv)

    number_of_visits = len(df.index)

    dataParser = DataParser(args.data_dir, split=split)

    for index, row in df.iterrows():
        print(f"Venue {index} / {number_of_visits}")

        visit_id = str(row["visit_id"])
        video_id = str(row["video_id"])

        print(f"Processing video_id {video_id} (visit_id: {visit_id}) ...")

        pcd = dataParser.get_laser_scan(visit_id)
        if args.crop_extraneous:
            pcd = dataParser.get_cropped_laser_scan(visit_id, pcd)

        refined_transform = dataParser.get_refined_transform(visit_id, video_id)
        pcd.transform(refined_transform)

        locs_in = np.array(pcd.points)
        n_points = locs_in.shape[0]

        poses_from_traj = dataParser.get_camera_trajectory(visit_id, video_id)

        frame_ids, rgb_frame_paths, intrinsics = dataParser.get_frame_id_and_intrinsic(
            visit_id, video_id, asset_type=args.coloring_asset, format="rgb"
        )

        _, depth_frame_paths, _ = dataParser.get_frame_id_and_intrinsic(
            visit_id, video_id, asset_type=args.coloring_asset, format="depth"
        )

        w, h, _, _, _, _ = intrinsics[list(intrinsics.keys())[0]]

        w = int(w)
        h = int(h)

        point2img_mapper = PointCloudToImageMapper(
            image_dim=(w, h),
            intrinsics=intrinsics,
            visibility_threshold=visibility_threshold,
            cut_bound=cut_bound,
        )

        counter = np.zeros((n_points, 1))
        sum_features = np.zeros((n_points, 3))
        n = len(frame_ids)
        progress_bar = tqdm(range(0, n), desc=f"Frames processing")
        skipped_frames = []
        for idx, frame_id in enumerate(frame_ids):
            if idx % 10 == 0:
                progress_bar.update(10)

            pose = dataParser.get_nearest_pose(
                frame_id,
                poses_from_traj,
                use_interpolation=use_interpolation,
                time_distance_threshold=time_distance_threshold,
            )

            if pose is None:
                # print(f"Skipping frame {idx}.")
                skipped_frames.append(frame_id)
                continue

            depth = dataParser.read_depth_frame(depth_frame_paths[frame_id])
            color = dataParser.read_rgb_frame(rgb_frame_paths[frame_id], normalize=True)

            # calculate the 3d-2d mapping based on the depth
            mapping = np.ones([n_points, 4], dtype=int)
            mapping[:, 1:4] = point2img_mapper.compute_mapping(
                frame_id, pose, locs_in, depth
            )

            if mapping[:, 3].sum() == 0:  # no points corresponds to this image, skip
                continue

            mask = mapping[:, 3]
            feat_2d_3d = color[mapping[:, 1], mapping[:, 2], :]

            counter[mask != 0] += 1
            sum_features[mask != 0] += feat_2d_3d[mask != 0]

        progress_bar.close()
        print(f"{len(skipped_frames)} frames were skipped because of unmet conditions.")
        counter[counter == 0] = 1e-5
        feat_bank = sum_features / counter
        feat_bank[feat_bank[:, 0:3] == [0.0, 0.0, 0.0]] = 169.0 / 255

        pcd.colors = o3d.utility.Vector3dVector(feat_bank)

        if args.save_as_float32:
            # save the output point cloud
            output_path = os.path.join(
                args.data_dir,
                split,
                visit_id,
                video_id,
                f"{video_id}_color_proj_{args.coloring_asset}.ply",
            )
            o3d.io.write_point_cloud(output_path, pcd)

            # convert float64 to float32
            plydata = PlyData.read(output_path)
            vertex = plydata["vertex"]
            vertex.properties = (
                PlyProperty("x", "float32"),
                PlyProperty("y", "float32"),
                PlyProperty("z", "float32"),
                PlyProperty("red", "uchar"),
                PlyProperty("green", "uchar"),
                PlyProperty("blue", "uchar"),
            )

            os.remove(output_path)
            plydata.write(output_path)

        if vis_result:
            pcd = pc_estimate_normals(pcd)
            viz_3d([pcd], show_coordinate_system=False)
