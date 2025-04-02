import math
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
from torch import Tensor


def pcd_valid_points(xyz: Tensor) -> Tensor:
    """
    Check if pcd is valid (no inf and no nans). Return indexes of valid
    """

    isnan = np.isnan(xyz).sum(axis=1)
    isinf = np.isinf(xyz).sum(axis=1)
    is_valid = 1 - (isnan + isinf)

    return np.nonzero(is_valid == 1)


def np_to_o3d(xyz: np.asarray, rgb: Optional[np.ndarray]) -> o3d.geometry.PointCloud:

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)

    return pcd


def o3d_to_np(pcd) -> np.ndarray:

    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors) if pcd.has_colors() else None

    if rgb is not None:
        xyz = np.concatenate([xyz, rgb], axis=1)

    return xyz


def np_transform_pcd(pcd: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:

    pcd = pcd.astype(np.float16)
    r = r.astype(np.float16)
    t = t.astype(np.float16)
    rot_pcd = np.dot(np.asarray(pcd), r.T) + t
    return rot_pcd
