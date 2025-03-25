import numpy as np
import torch.nn as nn
import torch
from scipy.spatial.transform import Rotation as RotLib

import ATE.transformations as tfs
import ATE.align_trajectory as align
from typing import Tuple
import g2o

"""
This file contains the LearnPose class, PoseGraphOptimization class, pose-realted helper functinoas and functions for the pose-related metrics.
LearnPose is a class that represents the pose of each frame, which allows the pose optmization by setting the requires_trad to True.
PoseGraphOptimization is a class that creats and optimizes the pose graph.
"""

# Define the LearnPose class
class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, cfg, init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param cfg: config argument options
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        self.original_init_c2w = None
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w.share_memory_(), requires_grad=False)
            self.original_init_c2w = nn.Parameter(init_c2w.clone(), requires_grad=False)

        self.r = nn.Parameter(
            torch.zeros(size=(num_cams, 3), dtype=torch.float32).share_memory_(),
            requires_grad=learn_R,
        )  # (N, 3)
        self.t = nn.Parameter(
            torch.zeros(size=(num_cams, 3), dtype=torch.float32).share_memory_(),
            requires_grad=learn_t,
        )  # (N, 3)

        elevation_angles = np.pi / 4 - np.pi / 2 / (64 - 1) * np.arange(64, dtype=np.float32)
        azimuth_angles = -2 * np.pi / 512 * np.arange(512, dtype=np.float32)
        elevation_grid, azimuth_grid = np.meshgrid(elevation_angles, azimuth_angles, indexing="ij")
        self.local_dir = np.vstack(
            (
                (np.cos(azimuth_grid) * np.cos(elevation_grid))[None, ...],
                (np.sin(azimuth_grid) * np.cos(elevation_grid))[None, ...],
                np.sin(elevation_grid)[None, ...],
            )
        )
        self.local_dir = torch.tensor(self.local_dir).cuda().view(3, -1).T
        self.pose_graph = None

    def forward(self, cam_id):
        self.r.data[0] = torch.zeros_like(self.r.data[0])
        self.t.data[0] = torch.zeros_like(self.t.data[0])
        cam_id = int(cam_id)
        r = self.r[cam_id]  # (3, ) axis-angle
        t = self.t[cam_id]  # (3, )
        c2w = make_c2w(r, t)  # (4, 4)
        return c2w

    def get_globalPose(self, cam_id):
        self.r.data[0] = torch.zeros_like(self.r.data[0])
        self.t.data[0] = torch.zeros_like(self.t.data[0])
        cam_id = int(cam_id)
        r = self.r[cam_id]  # (3, ) axis-angle
        t = self.t[cam_id]  # (3, )
        c2w = make_c2w(r, t)  # (4, 4)

        # learn a delta pose between init pose and target pose, if a init pose is provided
        if self.init_c2w is not None:
            c2w = self.init_c2w[cam_id] @ c2w

        return c2w

    def get_initPose(self, cam_id):
        return self.original_init_c2w[cam_id]

    def get_t(self):
        self.r.data[0] = torch.zeros_like(self.r.data[0])
        self.t.data[0] = torch.zeros_like(self.t.data[0])
        return self.t

    def return_pose(self):
        return [self.get_globalPose(i) for i in range(self.num_cams)]

    def get_init_delta(self, idx):
        return torch.linalg.inv(self.original_init_c2w.data[idx - 1]) @ self.original_init_c2w.data[idx]

    def get_frames(self, frame_idx):
        self.r.data[0] = torch.zeros_like(self.r.data[0])
        self.t.data[0] = torch.zeros_like(self.t.data[0])
        delta_r = self.r[frame_idx.cpu().long()]
        delta_t = self.t[frame_idx.cpu().long()]
        frames_c2w = make_c2w_batchify(delta_r, delta_t)
        # c2w = make_c2w(delta_r[3], delta_t[3])
        # print(c2w, frames_c2w[3])
        init_poses = self.init_c2w[frame_idx.cpu().long()]
        # print(init_poses.shape, frames_c2w.shape)
        c2w = init_poses @ frames_c2w
        return c2w.to(frame_idx.device)

    def get_frames_shift(self, frame_idx):
        self.r.data[0] = torch.zeros_like(self.r.data[0])
        self.t.data[0] = torch.zeros_like(self.t.data[0])
        delta_r = self.r[frame_idx.cpu().int()]
        delta_t = self.t[frame_idx.cpu().int()]
        frames_c2w = make_c2w_batchify(delta_r, delta_t)
        return frames_c2w.to(frame_idx.device)

    def get_localDir(self, elevation, azimuth):
        elevation_angle = torch.pi / 4 - torch.pi / 2 / (64 - 1) * elevation
        azimuth_angle = -2 * torch.pi / 512 * azimuth
        local_dir = torch.cat(
            [
                torch.cos(azimuth_angle) * torch.cos(elevation_angle),
                torch.sin(azimuth_angle) * torch.cos(elevation_angle),
                torch.sin(elevation_angle),
            ],
            dim=-1,
        )
        return local_dir

    def provide_gt(self, pose):
        self.gt_pose = pose.numpy()


# define the PoseGraphOptimization class
class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=20):
        super().initialize_optimization()
        super().set_verbose(False)
        super().optimize(max_iterations)

    def add_vertex(self, id: int, pose: np.ndarray, fixed=False):
        v_se3 = g2o.VertexSE3()
        v_se3.set_id(id)
        v_se3.set_estimate(g2o.Isometry3d(pose))
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_edge(
        self,
        vertices: Tuple[int, int],
        measurement: np.ndarray,
        information=np.identity(6),
        robust_kernel=None,
    ):
        edge = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(g2o.Isometry3d(measurement))  # relative pose
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, id: int) -> np.ndarray:
        return self.vertex(id).estimate().matrix()


def construct_odom_graph(poses0: np.ndarray) -> PoseGraphOptimization:
    """Construct the pose graph for odometry.
    Args:
        pose0: the initial odometry poses. Nx4x4
    Returns:
        graph: the constructed pose graph
    """
    G = PoseGraphOptimization()
    for i, pose in enumerate(poses0):
        G.add_vertex(i, pose, fixed=(i == 0))
    for i in range(len(poses0) - 1):
        T_ij = np.linalg.inv(poses0[i]) @ poses0[i + 1]
        G.add_edge([i, i + 1], T_ij)
    return G



# from this point on, the functions are for the pose-related metrics
def make_c2w(r, t):
    """
    :param r:  (3, ) axis-angle             torch tensor
    :param t:  (3, ) translation vector     torch tensor
    :return:   (4, 4)
    """
    R = Exp(r)  # (3, 3)
    c2w = torch.cat([R, t.unsqueeze(1)], dim=1)  # (3, 4)
    c2w = convert3x4_4x4(c2w)  # (4, 4)
    return c2w


def make_c2w_batchify(r, t):
    """
    :param r:  [N, 3] axis-angle             torch tensor
    :param t:  [N, 3] translation vector     torch tensor
    :return:   [N, 4, 4]
    """
    R = Exp_batchify(r)  # [N, 3, 3]
    t = t.unsqueeze(-1)  # [N, 3, 1]
    c2w = torch.cat([R, t], dim=2)  # [N, 3, 4]
    c2w = convert3x4_4x4(c2w)  # [N, 4, 4]
    return c2w


def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    return R


def Exp_batchify(r):
    """so(3) vector to SO(3) matrix for batch
    :param r: [N, 3] axis-angle, torch tensor
    :return:  [N, 3, 3]
    """
    skew_r = vec2skew_batchify(r)  # [N, 3, 3]
    norm_r = (r.norm(dim=1, keepdim=True) + 1e-15)[..., None]
    eye = torch.eye(3, dtype=torch.float32, device=r.device).unsqueeze(0).expand(r.shape[0], -1, -1)
    R = (
        eye
        + (torch.sin(norm_r) / norm_r) * skew_r
        + ((1 - torch.cos(norm_r)) / (norm_r**2)) * torch.matmul(skew_r, skew_r)
    )
    return R


def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat(
                [
                    input,
                    torch.tensor([[0, 0, 0, 1]], dtype=input.dtype, device=input.device),
                ],
                dim=0,
            )  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0, 0, 0, 1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([zero, -v[2:3], v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([v[2:3], zero, -v[0:1]])
    skew_v2 = torch.cat([-v[1:2], v[0:1], zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)


def vec2skew_batchify(v):
    """
    :param v:  [N, 3] torch tensor
    :return:   [N, 3, 3]
    """
    zeros = torch.zeros(v.shape[0], 1, dtype=torch.float32, device=v.device)
    skew_v = torch.cat(
        [
            zeros,
            -v[:, 2:3],
            v[:, 1:2],
            v[:, 2:3],
            zeros,
            -v[:, 0:1],
            -v[:, 1:2],
            v[:, 0:1],
            zeros,
        ],
        dim=1,
    ).reshape(-1, 3, 3)
    return skew_v


def align_ate_c2b_use_a2b(traj_a, traj_b, traj_c=None):
    """Align c to b using the se3 from a to b.
    :param traj_a:  (N0, 3/4, 4) torch tensor
    :param traj_b:  (N0, 3/4, 4) torch tensor
    :param traj_c:  None or (N1, 3/4, 4) torch tensor
    :return:        (N1, 4,   4) torch tensor
    """
    device = traj_a.device
    if traj_c is None:
        traj_c = traj_a.clone()

    traj_a = traj_a.float().cpu().numpy()
    traj_b = traj_b.float().cpu().numpy()
    traj_c = traj_c.float().cpu().numpy()

    R_a = traj_a[:, :3, :3]  # (N0, 3, 3)
    t_a = traj_a[:, :3, 3]  # (N0, 3)
    quat_a = SO3_to_quat(R_a)  # (N0, 4)

    R_b = traj_b[:, :3, :3]  # (N0, 3, 3)
    t_b = traj_b[:, :3, 3]  # (N0, 3)
    quat_b = SO3_to_quat(R_b)  # (N0, 4)

    # This function works in quaternion.
    # scalar, (3, 3), (3, ) gt = R * s * est + t.
    s, R, t = alignTrajectory(t_a, t_b, quat_a, quat_b, method="se3")

    # reshape tensors
    R = R[None, :, :].astype(np.float32)  # (1, 3, 3)
    t = t[None, :, None].astype(np.float32)  # (1, 3, 1)
    s = float(s)

    R_c = traj_c[:, :3, :3]  # (N1, 3, 3)
    t_c = traj_c[:, :3, 3:4]  # (N1, 3, 1)

    R_c_aligned = R @ R_c  # (N1, 3, 3)
    t_c_aligned = s * (R @ t_c) + t  # (N1, 3, 1)
    traj_c_aligned = np.concatenate([R_c_aligned, t_c_aligned], axis=2)  # (N1, 3, 4)

    # append the last row
    traj_c_aligned = convert3x4_4x4(traj_c_aligned)  # (N1, 4, 4)

    traj_c_aligned = torch.from_numpy(traj_c_aligned).to(device)
    return traj_c_aligned, R, t  # (N1, 4, 4)


def SO3_to_quat(R):
    """
    :param R:  (N, 3, 3) or (3, 3) np
    :return:   (N, 4, ) or (4, ) np
    """
    x = RotLib.from_matrix(R)
    quat = x.as_quat()
    return quat


def alignTrajectory(p_es, p_gt, q_es, q_gt, method, n_aligned=-1):
    """
    calculate s, R, t so that:
        gt = R * s * est + t
    method can be: sim3, se3, posyaw, none;
    n_aligned: -1 means using all the frames
    """
    assert p_es.shape[1] == 3
    assert p_gt.shape[1] == 3
    assert q_es.shape[1] == 4
    assert q_gt.shape[1] == 4

    s = 1
    R = None
    t = None
    if method == "sim3":
        assert n_aligned >= 2 or n_aligned == -1, "sim3 uses at least 2 frames"
        s, R, t = alignSIM3(p_es, p_gt, q_es, q_gt, n_aligned)
    elif method == "se3":
        R, t = alignSE3(p_es, p_gt, q_es, q_gt, n_aligned)
    elif method == "posyaw":
        R, t = alignPositionYaw(p_es, p_gt, q_es, q_gt, n_aligned)
    elif method == "none":
        R = np.identity(3)
        t = np.zeros((3,))
    else:
        assert False, "unknown alignment method"

    return s, R, t


# align by a SE3 transformation
def alignSE3Single(p_es, p_gt, q_es, q_gt):
    """
    Calculate SE3 transformation R and t so that:
        gt = R * est + t
    Using only the first poses of est and gt
    """

    p_es_0, q_es_0 = p_es[0, :], q_es[0, :]
    p_gt_0, q_gt_0 = p_gt[0, :], q_gt[0, :]

    g_rot = tfs.quaternion_matrix(q_gt_0)
    g_rot = g_rot[0:3, 0:3]
    est_rot = tfs.quaternion_matrix(q_es_0)
    est_rot = est_rot[0:3, 0:3]

    R = np.dot(g_rot, np.transpose(est_rot))
    t = p_gt_0 - np.dot(R, p_es_0)

    return R, t


def alignSE3(p_es, p_gt, q_es, q_gt, n_aligned=-1):
    """
    Calculate SE3 transformation R and t so that:
        gt = R * est + t
    """
    if n_aligned == 1:
        R, t = alignSE3Single(p_es, p_gt, q_es, q_gt)
        return R, t
    else:
        idxs = _getIndices(n_aligned, p_es.shape[0])
        est_pos = p_es[idxs, 0:3]
        gt_pos = p_gt[idxs, 0:3]
        s, R, t = align.align_umeyama(gt_pos, est_pos, known_scale=True)  # note the order
        t = np.array(t)
        t = t.reshape((3,))
        R = np.array(R)
        return R, t


# align by similarity transformation
def alignSIM3(p_es, p_gt, q_es, q_gt, n_aligned=-1):
    """
    calculate s, R, t so that:
        gt = R * s * est + t
    """
    idxs = _getIndices(n_aligned, p_es.shape[0])
    est_pos = p_es[idxs, 0:3]
    gt_pos = p_gt[idxs, 0:3]
    s, R, t = align.align_umeyama(gt_pos, est_pos)  # note the order
    return s, R, t


def _getIndices(n_aligned, total_n):
    if n_aligned == -1:
        idxs = np.arange(0, total_n)
    else:
        assert n_aligned <= total_n and n_aligned >= 1
        idxs = np.arange(0, n_aligned)
    return idxs


def alignPositionYawSingle(p_es, p_gt, q_es, q_gt):
    """
    calcualte the 4DOF transformation: yaw R and translation t so that:
        gt = R * est + t
    """

    p_es_0, q_es_0 = p_es[0, :], q_es[0, :]
    p_gt_0, q_gt_0 = p_gt[0, :], q_gt[0, :]
    g_rot = tfs.quaternion_matrix(q_gt_0)
    g_rot = g_rot[0:3, 0:3]
    est_rot = tfs.quaternion_matrix(q_es_0)
    est_rot = est_rot[0:3, 0:3]

    C_R = np.dot(est_rot, g_rot.transpose())
    theta = align.get_best_yaw(C_R)
    R = align.rot_z(theta)
    t = p_gt_0 - np.dot(R, p_es_0)

    return R, t


def alignPositionYaw(p_es, p_gt, q_es, q_gt, n_aligned=1):
    if n_aligned == 1:
        R, t = alignPositionYawSingle(p_es, p_gt, q_es, q_gt)
        return R, t
    else:
        idxs = _getIndices(n_aligned, p_es.shape[0])
        est_pos = p_es[idxs, 0:3]
        gt_pos = p_gt[idxs, 0:3]
        _, R, t = align.align_umeyama(gt_pos, est_pos, known_scale=True, yaw_only=True)  # note the order
        t = np.array(t)
        t = t.reshape((3,))
        R = np.array(R)
        return R, t

# The below four functions are the functions to call when computing the pose-related metrics

def compute_ATE(gt, pred):
    """Compute RMSE of ATE
    Args:
        gt: ground-truth poses
        pred: predicted poses
    """
    errors = []

    for i in range(len(pred)):
        # cur_gt = np.linalg.inv(gt_0) @ gt[i]
        cur_gt = gt[i]
        gt_xyz = cur_gt[:3, 3]

        # cur_pred = np.linalg.inv(pred_0) @ pred[i]
        cur_pred = pred[i]
        pred_xyz = cur_pred[:3, 3]

        align_err = gt_xyz - pred_xyz
        errors.append(np.sqrt(np.sum(align_err**2)))

    ate = np.sqrt(np.mean(np.asarray(errors) ** 2))
    mean_ate = np.mean(np.asarray(errors))
    return ate, mean_ate


def compute_rpe(gt, pred):
    trans_errors = []
    rot_errors = []
    for i in range(len(gt) - 1):
        gt1 = gt[i]
        gt2 = gt[i + 1]
        gt_rel = np.linalg.inv(gt1) @ gt2

        pred1 = pred[i]
        pred2 = pred[i + 1]
        pred_rel = np.linalg.inv(pred1) @ pred2
        rel_err = np.linalg.inv(gt_rel) @ pred_rel

        trans_errors.append(translation_error(rel_err))
        rot_errors.append(rotation_error(rel_err))
    rpe_trans = np.mean(np.asarray(trans_errors))
    rpe_rot = np.mean(np.asarray(rot_errors))
    return rpe_trans, rpe_rot


def rotation_error(pose_error):
    """Compute rotation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        rot_error (float): rotation error
    """
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    return rot_error


def translation_error(pose_error):
    """Compute translation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        trans_error (float): translation error
    """
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    trans_error = np.sqrt(dx**2 + dy**2 + dz**2)
    return trans_error
