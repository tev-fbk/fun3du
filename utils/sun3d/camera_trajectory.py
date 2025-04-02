import open3d as o3d

from utils.sun3d.viz import viz_3d


class CameraTrajectory:

    RADIUS = 0.05
    RESOLUTION = 2

    def __init__(self):
        self.cameras = []
        self.line_set = None

    def add_camera(self, camera_center):
        cam_vis = o3d.geometry.TriangleMesh.create_sphere(
            radius=self.RADIUS, resolution=self.RESOLUTION
        )
        cam_vis.paint_uniform_color([0, 0, 1])
        cam_vis.translate(camera_center)

        self.cameras.append(cam_vis)

    def calculate_line_set(self):
        points = []
        lines = []
        for idx, cam in enumerate(self.cameras):
            center = cam.get_center()
            points.append(center)
            lines.append([idx, idx + 1])

        lines.pop()
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        self.line_set = line_set

    def get_camera_trajectory(self):
        return [self.cameras, self.line_set]

    def viz_camera_trajectory(self, show_coordinate_system=True):
        camera_trajectory = self.get_camera_trajectory()
        camera_trajectory_to_plot = [*camera_trajectory[0], camera_trajectory[1]]
        viz_3d(camera_trajectory_to_plot, show_coordinate_system=show_coordinate_system)
