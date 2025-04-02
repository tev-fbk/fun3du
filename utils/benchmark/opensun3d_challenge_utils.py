# Based on https://github.com/apple/ARKitScenes/blob/main/threedod/benchmark_scripts/utils/tenFpsDataLoader.py
# Adapted by Ayca Takmaz, July 2023

import bisect
import copy
import glob
import json
import os
import pdb

import benchmark_scripts.utils.pc_utils as pc_utils
import cv2
import numpy as np
from benchmark_scripts.utils.rotation import convert_angle_axis_to_matrix3
from benchmark_scripts.utils.taxonomy import ARKitDatasetConfig, class_names


def TrajStringToMatrix(traj_str):
    """convert traj_str into translation and rotation matrices
    Args:
        traj_str: A space-delimited file where each line represents a camera position at a particular timestamp.
        The file has seven columns:
        * Column 1: timestamp
        * Columns 2-4: rotation (axis-angle representation in radians)
        * Columns 5-7: translation (usually in meters)

    Returns:
        ts: translation matrix
        Rt: rotation matrix
    """
    # line=[float(x) for x in traj_str.split()]
    # ts = line[0];
    # R = cv2.Rodrigues(np.array(line[1:4]))[0];
    # t = np.array(line[4:7]);
    # Rt = np.concatenate((np.concatenate((R, t[:,np.newaxis]), axis=1), [[0.0,0.0,0.0,1.0]]), axis=0)
    tokens = traj_str.split()
    assert len(tokens) == 7
    ts = tokens[0]
    # Rotation in angle axis
    angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
    r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))
    # Translation
    t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    Rt = np.linalg.inv(extrinsics)
    return (ts, Rt)


def st2_camera_intrinsics(filename):
    w, h, fx, fy, hw, hh = np.loadtxt(filename)
    return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])


def generate_point(
    rgb_image,
    depth_image,
    intrinsic,
    subsample=1,
    world_coordinate=True,
    pose=None,
):
    """Generate 3D point coordinates and related rgb feature
    Args:
        rgb_image: (h, w, 3) rgb
        depth_image: (h, w) depth
        intrinsic: (3, 3)
        subsample: int
            resize stride
        world_coordinate: bool
        pose: (4, 4) matrix
            transfer from camera to world coordindate
    Returns:
        points: (N, 3) point cloud coordinates
            in world-coordinates if world_coordinate==True
            else in camera coordinates
        rgb_feat: (N, 3) rgb feature of each point
    """
    intrinsic_4x4 = np.identity(4)
    intrinsic_4x4[:3, :3] = intrinsic

    u, v = np.meshgrid(
        range(0, depth_image.shape[1], subsample),
        range(0, depth_image.shape[0], subsample),
    )
    d = depth_image[v, u]
    d_filter = d != 0
    mat = np.vstack(
        (
            u[d_filter] * d[d_filter],
            v[d_filter] * d[d_filter],
            d[d_filter],
            np.ones_like(u[d_filter]),
        )
    )
    new_points_3d = np.dot(np.linalg.inv(intrinsic_4x4), mat)[:3]
    if world_coordinate:
        new_points_3d_padding = np.vstack(
            (new_points_3d, np.ones((1, new_points_3d.shape[1])))
        )
        world_coord_padding = np.dot(pose, new_points_3d_padding)
        new_points_3d = world_coord_padding[:3]

    rgb_feat = rgb_image[v, u][d_filter]

    return new_points_3d.T, rgb_feat


class FrameReaderDataLoaderLowRes(object):
    def __init__(
        self,
        class_names=class_names,
        root_path=None,
        gt_path=None,
        logger=None,
        frame_rate=1,
        with_color_image=True,
        subsample=1,
        world_coordinate=True,
        time_dist_limit=0.3,
    ):
        """
        Args:
            class_names: list of str
            root_path: path with all info for a scene_id
                color, color_2det, depth, label, vote, ...
            gt_path: xxx.json
                just to get correct floor height
            an2d_root: path to scene_id.json
                or None
            logger:
            frame_rate: int
            subsample: int
            world_coordinate: bool
        """
        self.root_path = root_path

        # pipeline does box residual coding here
        self.num_class = len(class_names)

        self.dc = ARKitDatasetConfig()

        depth_folder = os.path.join(self.root_path, "lowres_depth")
        if not os.path.exists(depth_folder):
            self.frame_ids = []
        else:
            depth_images = sorted(glob.glob(os.path.join(depth_folder, "*.png")))
            self.frame_ids = [os.path.basename(x) for x in depth_images]
            self.frame_ids = [x.split(".png")[0].split("_")[1] for x in self.frame_ids]
            self.video_id = depth_folder.split("/")[-2]  # depth_folder.split('/')[-3]
            self.frame_ids = [x for x in self.frame_ids]
            self.frame_ids.sort()
            self.intrinsics = {}

        traj_file = os.path.join(self.root_path, "lowres_wide.traj")
        with open(traj_file) as f:
            self.traj = f.readlines()
        # convert traj to json dict
        poses_from_traj = {}
        for line in self.traj:
            traj_timestamp = line.split(" ")[0]
            poses_from_traj[
                f"{round(float(traj_timestamp), 3):.3f}"
            ] = TrajStringToMatrix(line)[1].tolist()

        if os.path.exists(traj_file):
            # self.poses = json.load(open(traj_file))
            self.poses = poses_from_traj
        else:
            self.poses = {}

        self.frame_ids_new, self.closest_poses = find_closest_pose_from_timestamp(
            self.frame_ids, list(self.poses.keys()), time_dist_limit=time_dist_limit
        )
        print("Number of original frames:", len(self.frame_ids))
        print(
            "Number of frames with poses:", len(self.frame_ids_new)
        )  # 4682 frames to 4663 frames
        assert len(self.frame_ids_new) == len(self.closest_poses)

        self.frame_ids = self.frame_ids_new

        # get intrinsics
        for frame_id in self.frame_ids:
            intrinsic_fn = os.path.join(
                self.root_path,
                "lowres_wide_intrinsics",
                f"{self.video_id}_{frame_id}.pincam",
            )
            if not os.path.exists(intrinsic_fn):
                intrinsic_fn = os.path.join(
                    self.root_path,
                    "lowres_wide_intrinsics",
                    f"{self.video_id}_{float(frame_id) - 0.001:.3f}.pincam",
                )
            if not os.path.exists(intrinsic_fn):
                intrinsic_fn = os.path.join(
                    self.root_path,
                    "lowres_wide_intrinsics",
                    f"{self.video_id}_{float(frame_id) + 0.001:.3f}.pincam",
                )
            if not os.path.exists(intrinsic_fn):
                print("frame_id", frame_id)
                print(intrinsic_fn)
            self.intrinsics[frame_id] = st2_camera_intrinsics(intrinsic_fn)

        self.frame_rate = frame_rate
        self.subsample = subsample
        self.with_color_image = with_color_image
        self.world_coordinate = world_coordinate

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        """
        Returns:
            frame: a dict
                {frame_id}: str
                {depth}: (h, w)
                {image}: (h, w)
                {image_path}: str
                {intrinsics}: np.array 3x3
                {pose}: np.array 4x4
                {pcd}: np.array (n, 3)
                    in world coordinate
                {color}: (n, 3)
        """
        frame_id = self.frame_ids[idx]
        frame = {}
        frame["frame_id"] = frame_id
        fname = "{}_{}.png".format(self.video_id, frame_id)
        # fname = "{}.png".format(frame_id)
        depth_image_path = os.path.join(self.root_path, "lowres_depth", fname)
        if not os.path.exists(depth_image_path):
            print(depth_image_path)

        image_path = os.path.join(self.root_path, "lowres_wide", fname)

        if not os.path.exists(depth_image_path):
            print(depth_image_path, "does not exist")
        frame["depth"] = cv2.imread(depth_image_path, -1)
        frame["image"] = cv2.cvtColor(
            cv2.imread(
                image_path,
            ),
            cv2.COLOR_BGR2RGB,
        )

        frame["image_path"] = image_path
        depth_height, depth_width = frame["depth"].shape
        im_height, im_width, im_channels = frame["image"].shape

        frame["intrinsics"] = copy.deepcopy(self.intrinsics[frame_id])

        if str(frame_id) in self.poses.keys():
            frame_pose = np.array(self.poses[str(frame_id)])
        else:
            frame_pose = np.array(self.poses[self.closest_poses[idx]])
        frame["pose"] = copy.deepcopy(frame_pose)

        if depth_height != im_height:
            frame["image"] = np.zeros([depth_height, depth_width, 3])  # 288, 384, 3
            frame["image"][48 : 48 + 192, 64 : 64 + 256, :] = cv2.cvtColor(
                cv2.imread(image_path), cv2.COLOR_BGR2RGB
            )

        (m, n, _) = frame["image"].shape
        depth_image = frame["depth"] / 1000.0  # rescale to obtain depth in meters
        rgb_image = frame["image"] / 255.0

        pcd, rgb_feat = generate_point(
            rgb_image,
            depth_image,
            frame["intrinsics"],
            self.subsample,
            self.world_coordinate,
            frame_pose,
        )

        frame["pcd"] = pcd
        frame["color"] = rgb_feat
        frame["depth"] = depth_image  # depth in meters
        return frame


def find_closest_pose_from_timestamp(
    image_timestamps, pose_timestamps, time_dist_limit=0.3
):
    closest_poses = []
    new_frame_ids = []

    for image_ts in image_timestamps:
        index = bisect.bisect_left(pose_timestamps, image_ts)

        if index == 0:
            closest_pose = pose_timestamps[index]
        elif index == len(pose_timestamps):
            closest_pose = pose_timestamps[index - 1]
        else:
            diff_prev = abs(float(image_ts) - float(pose_timestamps[index - 1]))
            diff_next = abs(float(image_ts) - float(pose_timestamps[index]))

            if diff_prev < diff_next:
                closest_pose = pose_timestamps[index - 1]
            else:
                closest_pose = pose_timestamps[index]

        if abs(float(closest_pose) - float(image_ts)) >= time_dist_limit:
            pass
            # print("Warning: Closest pose is not close enough to image timestamp:", image_ts, closest_pose)
        else:
            closest_poses.append(closest_pose)
            new_frame_ids.append(image_ts)

    return new_frame_ids, closest_poses


def pcd_accumulate_wrapper(loader, grid_size=0.05):
    """
    Args:
        loader: FrameReaderDataLoader
    Returns:
        world_pc: (N, 3)
            xyz in world coordinate system
        world_sem: (N, d)
            semantic for each point
        grid_size: float
            keep only one point in each (g_size, g_size, g_size) grid
    """
    world_pc, world_rgb, poses = np.zeros((0, 3)), np.zeros((0, 3)), []
    for i in range(len(loader)):
        frame = loader[i]
        print(f"{i}/{len(loader)}", frame["image_path"])
        image_path = frame["image_path"]
        pcd = frame["pcd"]  # in world coordinate
        pose = frame["pose"]
        rgb = frame["color"]

        world_pc = np.concatenate((world_pc, pcd), axis=0)
        world_rgb = np.concatenate((world_rgb, rgb), axis=0)

        choices = pc_utils.down_sample(world_pc, 0.05)
        world_pc = world_pc[choices]
        world_rgb = world_rgb[choices]

    return world_pc, world_rgb, poses
