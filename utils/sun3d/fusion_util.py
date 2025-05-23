"""
2D/3D fusion utils

adapted from https://github.com/pengsongyou/openscene/blob/main/scripts/feature_fusion/fusion_util.py
"""

import glob
import math
import os

import numpy as np
import torch


def make_intrinsic(fx, fy, mx, my):
    """Create camera intrinsics."""

    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic


def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    """Adjust camera intrinsics."""

    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(
        math.floor(
            image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])
        )
    )
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


class PointCloudToImageMapper(object):
    def __init__(self, image_dim, visibility_threshold=0.25, cut_bound=0):

        self.image_dim = image_dim
        self.vis_thres = visibility_threshold
        self.cut_bound = cut_bound

    def compute_multi_masked_mapping(
        self, camera_to_world, coords, mask_list, depth, intrinsic, device
    ):
        """
        Same thing as masked mapping, but batches a list of N masks for better efficiency!

        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        depth = torch.tensor(depth).to(device)
        mask_list = torch.tensor(mask_list).to(device)
        intrinsic = torch.tensor(intrinsic).to(device)
        camera_to_world = torch.tensor(camera_to_world).to(device)
        mapping = torch.zeros(
            (mask_list.shape[0], 3, coords.shape[0]), dtype=torch.int
        ).to(device)
        coords_new = torch.cat(
            [coords, torch.ones([coords.shape[0], 1]).to(device)], dim=1
        ).transpose(1, 0)
        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = torch.linalg.inv(camera_to_world)
        p = torch.matmul(world_to_camera, coords_new)
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = torch.round(p).to(torch.int)  # simply round the projected coordinates
        inside_mask = (
            (pi[0] >= self.cut_bound)
            * (pi[1] >= self.cut_bound)
            * (pi[0] < self.image_dim[0] - self.cut_bound)
            * (pi[1] < self.image_dim[1] - self.cut_bound)
        )

        for i, mask in enumerate(mask_list):
            _depth = depth.clone()
            _depth = torch.where(mask <= 0, 0, _depth)
            depth_cur = _depth[pi[1][inside_mask], pi[0][inside_mask]]
            occlusion_mask = (
                torch.abs(
                    depth[pi[1][inside_mask], pi[0][inside_mask]] - p[2][inside_mask]
                )
                <= self.vis_thres * depth_cur
            )
            inside_mask_i = inside_mask.clone()
            inside_mask_i[inside_mask == True] = occlusion_mask

            mapping[i][0][inside_mask_i] = pi[1][inside_mask_i]
            mapping[i][1][inside_mask_i] = pi[0][inside_mask_i]
            mapping[i][2][inside_mask_i] = 1

        return mapping.transpose(2, 1).cpu().numpy()

    def compute_masked_mapping(
        self, camera_to_world, coords, mask, depth, intrinsic, device
    ):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        depth = torch.tensor(depth).to(device)
        mask = torch.tensor(mask).to(device)
        intrinsic = torch.tensor(intrinsic).to(device)
        mapping = torch.zeros((3, coords.shape[0]), dtype=torch.int).to(device)
        coords_new = torch.cat(
            [coords, torch.ones([coords.shape[0], 1]).to(device)], dim=1
        ).transpose(1, 0)
        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = torch.tensor(np.linalg.inv(camera_to_world)).to(device)
        p = torch.matmul(world_to_camera, coords_new)
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = torch.round(p).to(torch.int)  # simply round the projected coordinates
        inside_mask = (
            (pi[0] >= self.cut_bound)
            * (pi[1] >= self.cut_bound)
            * (pi[0] < self.image_dim[0] - self.cut_bound)
            * (pi[1] < self.image_dim[1] - self.cut_bound)
        )

        _depth = depth.clone()
        _depth = torch.where(mask != 1, 0, _depth)
        depth_cur = _depth[pi[1][inside_mask], pi[0][inside_mask]]
        occlusion_mask = (
            torch.abs(depth[pi[1][inside_mask], pi[0][inside_mask]] - p[2][inside_mask])
            <= self.vis_thres * depth_cur
        )

        inside_mask[inside_mask == True] = occlusion_mask

        mapping[0][inside_mask] = pi[1][inside_mask]
        mapping[1][inside_mask] = pi[0][inside_mask]
        mapping[2][inside_mask] = 1

        return mapping.transpose(1, 0).cpu().numpy()

    def compute_mapping(self, camera_to_world, coords, depth=None, intrinsic=None):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """

        mapping = np.zeros((3, coords.shape[0]), dtype=int)
        coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coords_new)
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = np.round(p).astype(int)  # simply round the projected coordinates
        inside_mask = (
            (pi[0] >= self.cut_bound)
            * (pi[1] >= self.cut_bound)
            * (pi[0] < self.image_dim[0] - self.cut_bound)
            * (pi[1] < self.image_dim[1] - self.cut_bound)
        )
        if depth is not None:
            depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
            occlusion_mask = (
                np.abs(
                    depth[pi[1][inside_mask], pi[0][inside_mask]] - p[2][inside_mask]
                )
                <= self.vis_thres * depth_cur
            )

            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2] > 0  # make sure the depth is in front
            inside_mask = front_mask * inside_mask
        mapping[0][inside_mask] = pi[1][inside_mask]
        mapping[1][inside_mask] = pi[0][inside_mask]
        mapping[2][inside_mask] = 1

        return mapping.T
