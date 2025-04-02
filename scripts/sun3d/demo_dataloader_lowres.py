# Based on https://github.com/apple/ARKitScenes/blob/main/threedod/benchmark_scripts/data_prepare_online.py
# Adapted, July 2023

import os

import numpy as np
import open3d as o3d
import pyviz3d.visualizer as viz

from utils.benchmark.opensun3d_challenge_utils import FrameReaderDataLoaderLowRes

if __name__ == "__main__":
    arkitscenes_root_dir = "data"  # "PATH/TO/ARKITSCENES/DOWNLOAD/DIR"
    data_type = "ChallengeDevelopmentSet"  # or "ChallengeTestSet"
    scene_ids = [
        "42445173",
        "42445677",
        "42445935",
        "42446478",
        "42446588",
    ]  # Dev Scenes
    scene_id = scene_ids[4]
    img_folder = "lowres_wide"

    # Set up dataloader
    data_root_dir = os.path.join(arkitscenes_root_dir, data_type)
    scene_dir = os.path.join(data_root_dir, scene_id)
    print(os.path.abspath(scene_dir))
    try:
        loader = FrameReaderDataLoaderLowRes(root_path=scene_dir)
    except FileNotFoundError:
        print(">>> Error: Data not found. Did you download it?")
        print(
            ">>> Try: python download_data_opensun3d.py --data_type=challenge_development_set --download_dir=data"
        )
        exit()

    print("Number of frames to load: ", len(loader))

    # Visualize RGB-D point clouds (for subset)
    v = viz.Visualizer()
    for i in [2560, 2620, 2680, 2800]:
        frame = loader[i]
        frame_id = frame["frame_id"]  # str
        depth = frame["depth"]  # (h, w)
        image = frame["image"]  # (h, w, 3), [0-255], uint8
        image_path = frame["image_path"]  # str
        intrinsics = frame["intrinsics"]  # (3,3) - camera intrinsics
        pose = frame["pose"]  # (4,4) - camera pose
        pcd = frame[
            "pcd"
        ]  # (n, 3) - backprojected point coordinates in world coordinate system! not camera coordinate system! backprojected and transformed to world coordinate system
        pcd_color = frame["color"]  # (n,3) - color values for each point
        v.add_points(f"rgb;{i}", pcd, pcd_color * 255, point_size=5, visible=True)

    # Draw camera trajectory
    trajectory = []
    for i in range(len(loader))[::10]:
        pose = loader[i]["pose"]  # (4,4) - camera pose
        camera_position_i = pose @ np.array([0.0, 0.0, 0.0, 1.0])
        trajectory.append(camera_position_i[0:3])
    v.add_polyline(
        "Trajectory",
        positions=np.vstack(trajectory),
        edge_width=0.005,
        color=np.array([255, 0, 0]),
    )

    # Add ARKit Reconstructed Mesh
    pcd = o3d.io.read_point_cloud(os.path.join(scene_dir, f"{scene_id}_3dod_mesh.ply"))
    v.add_points(
        f"Mesh ARKit",
        np.asarray(pcd.points),
        np.asarray(pcd.colors) * 255,
        point_size=20,
        visible=True,
    )

    v.save("viz")
