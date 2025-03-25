from pathlib import Path
import numpy as np
from typing import List
import open3d as o3d
import torch
from torch.utils.data import Dataset
from .ray_samplers import UniformRaySampler
from .dataset_getitem import obtain_getitem
from scipy.spatial import cKDTree
from model.PoseLearn import align_ate_c2b_use_a2b, compute_ATE, compute_rpe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset to load the data used in each trajectory
class PanoradarRadarDataset(Dataset):
    def __init__(self, cfg, model, train=True):
        # register the necessary parameters and the model
        self.train = train
        self.model = model
        if self.train:
            print("Training with PanoRadar Radar data")
        else:
            print("Evaluating with PanoRadar Radar data")

        self.outlier_range = cfg.dataset.outlier_range
        self.get_vel_pose = cfg.dataset.use_vel_pose
        self.ray_pose_th = cfg.train.rays_th
        self.cfg = cfg
        self.global_depth = None
        self.global_dir = None
        self.global_pos = None
        self.ray_range = np.array(cfg.train.ray_range, dtype=np.float32)
        self.model_name = cfg.dataset.model_name
        self.batch_training = cfg.dataset.batch_training
        self.ray_pts = cfg.train.ray_points

        if self.train:
            self.input_epsilon = 0.004
            print(f'epsilon is set to {self.input_epsilon}')

        self.use_uncertainty = cfg.dataset.use_uncertainty
        print(f"Using uncertainty {self.use_uncertainty}")

        # Load the data from the given directory in the config file
        file = self.cfg.dataset.lidar_pose_file
        # load ground truth poses
        self.lidar_pose = torch.tensor(np.load(file).astype(np.float32))
        # load the range images
        self.load_data()
        # load the initialized pose
        self.load_pose()
        # get the bounding box for the scene
        self.get_cube()

        print(
            f"global depth, pos, dir shape f{self.global_depth.shape}, {self.global_pos.shape}, {self.global_dir.shape}"
        )
        if cfg.model.optimize_pose:
            self.optimize_pose = True
        else: 
            self.optimize_pose = False

        # normalize the scene coordinates based on the scal_factor achieved from the self.get_cube() function
        self.render_grid_res = cfg.render.grid_res / self.scale_factor
        self.image_shape = self.global_depth.shape[:3]  # (N_images, N_elev, N_azi)
        self.scaled_ray_range = torch.tensor(self.ray_range / self.scale_factor)
        self.global_depth = torch.tensor(self.global_depth / self.scale_factor)
        self.global_pos = torch.tensor(self.global_pos)
        self.global_dir = torch.tensor(self.global_dir)
        self.global_depth_lidar = torch.tensor(self.global_depth_lidar / self.scale_factor)
        self.conf = torch.tensor(self.conf / self.scale_factor, dtype=torch.float32)
        self.raw_conf = torch.tensor(self.raw_conf, dtype=torch.float32).reshape(-1)
        self.pose_optimize_index = torch.where(self.raw_conf < self.ray_pose_th)[0]
        self.shift_tensor = torch.tensor(self.shift).to(device)

        # preparation for the online SLAM
        self.online_depth = self.global_depth.clone().squeeze()
        self.online_pose = torch.tensor(self.pose)
        self.online_lidar = self.global_depth_lidar
        self.online_conf = self.conf

        # for global random sampling all the rays
        self.chunk = cfg.train.chunk_size
        self.pose_chunk = cfg.train.pose_chunk_size
        total_iters = cfg.train.total_iters

        self.distances = None

        # prepare the data for training
        if self.train:
            if not self.batch_training:
                self.indices = torch.cat(
                    [
                        torch.randperm(self.image_shape[0] * self.image_shape[1] * self.image_shape[2])
                        for _ in range(
                            total_iters
                            * self.chunk
                            // (self.image_shape[0] * self.image_shape[1] * self.image_shape[2])
                            + 1
                        )
                    ]
                )
            else:
                self.indices = torch.cat(
                    [torch.randperm(self.image_shape[0]) for _ in range(total_iters // self.image_shape[0] + 1)]
                )

            self.pose_indices = torch.cat(
                [
                    torch.randperm(self.pose_optimize_index.shape[0])
                    for _ in range(total_iters * (self.pose_chunk) // self.pose_optimize_index.shape[0] + 1)
                ]
            )

        # register the sampling method for the render and the getitem function
        self.get_item = obtain_getitem(self.model_name)
        self.test_sampler = UniformRaySampler()

        print(f"cooridnates has been normalized into a unit cube. We have {self.global_depth.shape[0]} frames.")

    def load_data(self):
        """Load the range images and the glass masks"""
        file = self.cfg.dataset.image_dir

        npy_files = sorted(Path(file).glob("*.npy"))
        self.depth = np.asarray([np.load(file) * 10 for file in npy_files])
        self.raw_depth = np.asarray([np.load(file) * 10 for file in npy_files])
        print("Depth shape:", self.depth.shape)

        file = self.cfg.dataset.image_lidar_dir

        npy_files = sorted(Path(file).glob("*.npy"))
        self.global_depth_lidar = np.concatenate([np.load(file) * 10 for file in npy_files], 0)

        file = self.cfg.dataset.var_dir

        npy_files = sorted(Path(self.cfg.dataset.var_dir).glob("*.npy"))
        self.raw_conf = np.concatenate([np.load(file) * 10 for file in npy_files], axis=0)
        self.pre_raw_conf = np.copy(self.raw_conf)
        print(f"conf_shape {self.raw_conf.shape}")
        self.conf = self.raw_conf / np.sqrt(2.0)

    def get_cube(self):
        # find the bbox for the scene and normailze the scene coordinates
        padding = 0.3

        global_pos = self.pose @ np.linalg.inv(self.pose[0, :, :])
        max_depth = self.ray_range[1]
        lidar_view_corners = np.array(
            [
                [-max_depth, -max_depth, -max_depth, 1],
                [-max_depth, max_depth, -max_depth, 1],
                [max_depth, -max_depth, -max_depth, 1],
                [max_depth, max_depth, -max_depth, 1],
                [-max_depth, -max_depth, max_depth, 1],
                [-max_depth, max_depth, max_depth, 1],
                [max_depth, -max_depth, max_depth, 1],
                [max_depth, max_depth, max_depth, 1],
            ]
        )

        all_corners = np.matmul(global_pos[:, :3, :], lidar_view_corners.T).transpose(0, 2, 1).reshape(-1, 3)

        all_poses = global_pos[..., :3, 3]

        all_points = np.concatenate([all_corners, all_poses])

        min_coord = all_points.min(axis=0)
        max_coord = all_points.max(axis=0)
        self.coords = [min_coord, max_coord]

        origin = min_coord + (max_coord - min_coord) / 2

        scale_factor = (np.linalg.norm(max_coord - min_coord) / (2 * np.sqrt(np.array([3])))) * (1 + padding)

        self.scale_factor = scale_factor[0].astype(np.float32)
        self.shift = -origin.astype(np.float32)

    def load_pose(self):
        # load all necessary poses from the given directory
        file = self.cfg.dataset.pose_file
        if not self.get_vel_pose:
            self.pose = np.load(file)
        else:
            vel = np.load(self.cfg.dataset.vel_file)
            v_esti = vel["v_esti"]
            theta_v_esti = vel["theta_v_esti"]
            omega_imu = vel["omega_imu"]

            time_gap = 0.55
            self.pose = np.array(np.identity(4)[None, ...])
            for idx in range(1, v_esti.shape[0]):
                pre_v_esti = v_esti[idx - 1]
                pre_theta_v_esti = theta_v_esti[idx - 1]
                pre_omega_imu = omega_imu[idx - 1]

                delta = np.array(
                    [
                        [
                            np.cos(pre_omega_imu * time_gap),
                            -np.sin(pre_omega_imu * time_gap),
                            0,
                            np.cos(pre_theta_v_esti) * pre_v_esti * time_gap,
                        ],
                        [
                            np.sin(pre_omega_imu * time_gap),
                            np.cos(pre_omega_imu * time_gap),
                            0,
                            np.sin(pre_theta_v_esti) * pre_v_esti * time_gap,
                        ],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )

                prev_global_pose = self.pose[-1]
                cur_global_pose = prev_global_pose @ delta
                self.pose = np.concatenate((self.pose, cur_global_pose[None, ...]))
        self.pose = self.pose.astype(np.float32)

        print("pose shape", self.pose.shape)
        depth_shape = self.depth.shape
        elevation_angles = np.pi / 4 - np.pi / 2 / (depth_shape[2] - 1) * np.arange(depth_shape[2], dtype=np.float32)
        azimuth_angles = -2 * np.pi / depth_shape[3] * np.arange(depth_shape[3], dtype=np.float32)
        elevation_grid, azimuth_grid = np.meshgrid(elevation_angles, azimuth_angles, indexing="ij")
        self.local_dir = np.vstack(
            (
                (np.cos(azimuth_grid) * np.cos(elevation_grid))[None, ...],
                (np.sin(azimuth_grid) * np.cos(elevation_grid))[None, ...],
                np.sin(elevation_grid)[None, ...],
            )
        )

        self.global_pos = self.pose[:, :3, 3]

        self.global_pos = np.tile(self.global_pos[:, None, None, :], (1, depth_shape[2], depth_shape[3], 1))

        self.global_dir = (
            np.matmul(self.pose[:, :3, :3], self.local_dir.reshape(3, -1))
            .reshape(self.pose.shape[0], 3, self.local_dir.shape[1], self.local_dir.shape[2])
            .transpose(0, 2, 3, 1)
        )

        self.global_depth = self.depth.transpose(0, 2, 3, 1)

    def __getitem__(self, idx):
        """Global random ray sampling method"""

        if not self.batch_training:
            indices = self.indices[idx * self.chunk : (idx + 1) * self.chunk]
            pose_indices = self.pose_indices[idx * self.pose_chunk : (idx + 1) * self.pose_chunk]
            return idx, indices, idx, pose_indices
        else:
            frame_idx = self.indices[idx]
            indices = torch.randperm(self.image_shape[1] * self.image_shape[2])[: self.chunk]
            pose_indices = self.pose_indices[idx * self.pose_chunk : (idx + 1) * self.pose_chunk]
            return idx, indices, frame_idx, pose_indices

    def __len__(self):
        return self.global_pos.shape[0]


def train_step(
    batch,
    model,
    optimizer,
    lr_scheduler,
    dataset,
    pose_model=None,
    pose_optim=None,
    pose_scheduler=None,
    idx=None,
):
    # training step for the model, feed the data into the model and optimize the model
    optimizer.zero_grad()
    if dataset.optimize_pose:
        pose_optim.zero_grad()
    loss, _, logs = model.get_loss(batch, dataset, optimize_pose=False, pose_model=pose_model)
    loss.backward()

    optimizer.step()
    if dataset.optimize_pose:
        pose_optim.step()
        if pose_scheduler is not None:
            pose_scheduler.step()
            lr = pose_scheduler.get_last_lr()[0]
            logs["pose_lr"] = lr

    if lr_scheduler is not None:
        lr_scheduler.step()

        lr = lr_scheduler.get_last_lr()[0]
        logs["lr"] = lr

    return logs


def train_log(logs, writer, logger, i_step) -> None:
    # log function for training
    if writer is not None:
        for key, value in logs.items():
            writer.add_scalar(f"train/{key}", value, i_step)
    if logger is not None:
        logger.info(f'step {i_step}, lr: {logs["lr"]}, training Loss: {logs["loss"]:.5f}')


def val_step(model, val_dataset, config, pose_model=None, online_eval=False) -> List[torch.Tensor]:
    """Evaluate the l1 depth error for the whole scene."""
    # get necessary parameters
    other_logs = {}
    N_frame, N_elev, N_azi = val_dataset.image_shape
    N_rays = N_elev * N_azi
    depths = val_dataset.global_depth_lidar.view(-1, N_elev * N_azi, 1)
    scale_factor = val_dataset.scale_factor

    grid_res = config.render.grid_res / scale_factor
    every_n_frame = config.render.every_n_frame
    render_errs = []
    render_diff_errs = []
    grid_res = val_dataset.render_grid_res
    near, far = val_dataset.scaled_ray_range

    gt_poses = val_dataset.lidar_pose
    pose_tree = cKDTree(gt_poses[:, :3, 3])
    model.eval()
    ray_vis = dict()

    # start the rending for the whole scene, every_n_frame is the interval for the rendered frame
    with torch.no_grad():
        for idx in range(0, N_frame, every_n_frame):
            # get the pose and the direction for the current frame
            pose = pose_model.get_globalPose(idx)
            pose[:3, 3] = (pose[:3, 3] + val_dataset.shift_tensor) / val_dataset.scale_factor
            pos = pose[:3, 3:].T  # [1, 3]
            dir = (pose[:3, :3] @ pose_model.local_dir.T).T  # [N, 3]

            depth_lidar = depths[idx].to(device)
            pts_num = torch.round(far / grid_res).to(torch.int32)

            # ocmpose rays and sample points
            rays = torch.cat([torch.full((N_rays, 1), near), torch.full((N_rays, 1), far)], dim=1).to(device)
            z_vals = val_dataset.test_sampler.get_samples(rays, pts_num, False)
            pts = pos.unsqueeze(1) + dir.unsqueeze(1) * z_vals.unsqueeze(2)

            depth_render = []
            chunk_num = 16
            chunk_size = N_elev * N_azi // chunk_num * pts_num

            # render the depth result by chunk for gpu memory saving
            for chunk_id in range(chunk_num):
                ret = model.render(
                    pts=pts[int(chunk_id * chunk_size / pts_num) : int((chunk_id + 1) * chunk_size / pts_num)],
                    z_vals=z_vals[int(chunk_id * chunk_size / pts_num) : int((chunk_id + 1) * chunk_size / pts_num)],
                )
                depth_map = ret["depth_map"]
                depth_render.append(depth_map)  # (N_rays,)

            depth_render = torch.cat(depth_render)
            depth_lidar = depth_lidar[:, 0]
            cur_gt_pose = gt_poses[idx]
            cur_dir = (cur_gt_pose[:3, :3] @ val_dataset.local_dir.reshape(3, -1)).T
            cur_pts = cur_gt_pose[:3, 3:].T.to(depth_lidar.device) + cur_dir.to(depth_lidar.device) * (
                depth_lidar[..., None] * scale_factor
            )
            gt_pts_distances, _ = pose_tree.query(cur_pts.cpu())
            outlier_mask = gt_pts_distances <= val_dataset.outlier_range
            valid_region = (
                (depth_lidar > val_dataset.scaled_ray_range[0])
                & (depth_lidar < val_dataset.scaled_ray_range[1])
                & torch.tensor(outlier_mask, device=depth_lidar.device)
            )

            # get all depth information
            render_errs.append(torch.abs(depth_lidar - depth_render)[valid_region])
            render_diff_errs.append((depth_render - depth_lidar)[valid_region])

    # summarize the error
    render_errs = torch.cat(render_errs) * scale_factor
    render_diff_errs = torch.cat(render_diff_errs) * scale_factor
    gt_poses = val_dataset.lidar_pose
    pred_poses = torch.cat([(pose_model.get_globalPose(i))[None, ...] for i in range(pose_model.num_cams)]).detach()

    # align the pose for the ATE calculation
    c2ws_est_aligned, _, _ = align_ate_c2b_use_a2b(pred_poses, gt_poses)  # (N, 4, 4)

    # compute ate, rpe
    ate_rmse, ate_mean = compute_ATE(gt_poses.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
    rpe_trans, rpe_rot = compute_rpe(gt_poses.cpu().numpy(), c2ws_est_aligned.cpu().numpy())

    other_logs["RPE_TRANS(MEAN: cm)"] = rpe_trans * 100
    other_logs["RPE_ROT(MEAN: deg)"] = rpe_rot
    other_logs["ATE(RMSE: cm)"] = ate_rmse * 100
    other_logs["ATE(MEAN: cm)"] = ate_mean * 100

    # get the statistics for the depth error
    result = (
        torch.std(render_errs).cpu().numpy(),
        torch.mean(render_errs).cpu().numpy(),
        torch.median(render_errs).cpu().numpy(),
        torch.quantile(render_errs, 0.1).cpu().numpy(),
        torch.quantile(render_errs, 0.2).cpu().numpy(),
        torch.quantile(render_errs, 0.8).cpu().numpy(),
        torch.quantile(render_errs, 0.9).cpu().numpy(),
    )
    diff_result = (
        torch.std(render_diff_errs).cpu().numpy(),
        torch.mean(render_diff_errs).cpu().numpy(),
        torch.median(render_diff_errs).cpu().numpy(),
        torch.quantile(render_diff_errs, 0.1).cpu().numpy(),
        torch.quantile(render_diff_errs, 0.2).cpu().numpy(),
        torch.quantile(render_diff_errs, 0.8).cpu().numpy(),
        torch.quantile(render_diff_errs, 0.9).cpu().numpy(),
    )
    # set the model back to the training mode
    model.train()
    return result, diff_result, ray_vis, other_logs


def val_log(results, writer, logger, i_step) -> None:
    # log function for validation
    result, diff_result, ray_vis, other_logs = results

    for name in ray_vis.keys():
        writer.add_image(f"val image {name}", ray_vis[name], global_step=i_step)

    for name in other_logs.keys():
        writer.add_scalar("val_pose/" + name, other_logs[name], i_step)

    writer.add_scalar("val_diff/stddepth_error", diff_result[0], i_step)
    writer.add_scalar("val_diff/mean_depth_error", diff_result[1], i_step)
    writer.add_scalar("val_diff/median_depth_error", diff_result[2], i_step)
    writer.add_scalar("val_diff/10_depth_error", diff_result[3], i_step)
    writer.add_scalar("val_diff/20_depth_error", diff_result[4], i_step)
    writer.add_scalar("val_diff/80_depth_error", diff_result[5], i_step)
    writer.add_scalar("val_diff/90_depth_error", diff_result[6], i_step)

    writer.add_scalar("val_abs/stddepth_error", result[0], i_step)
    writer.add_scalar("val_abs/mean_depth_error", result[1], i_step)
    writer.add_scalar("val_abs/median_depth_error", result[2], i_step)
    writer.add_scalar("val_abs/10_depth_error", result[3], i_step)
    writer.add_scalar("val_abs/20_depth_error", result[4], i_step)
    writer.add_scalar("val_abs/80_depth_error", result[5], i_step)
    writer.add_scalar("val_abs/90_depth_error", result[6], i_step)
    if logger is not None:
        logger.info(
            f"step {i_step}, mean_depth_error: {result[1]:.4f}, median_depth_error: {result[2]:.4f}, "
            f"80_depth_error: {result[5]:.4f}, 90_depth_error: {result[6]:.4f}"
        )

def online_train_step(
    rays_sampled,
    model,
    optimizer,
    lr_scheduler,
    dataset,
    pose_model=None,
    pose_optim=None,
    pose_scheduler=None,
    idx=None,
    optimize_pose=False
):
    # online training step for the model
    optimizer.zero_grad()
    if dataset.optimize_pose and optimize_pose:
        pose_optim.zero_grad()

    loss, logs = model.online_get_loss(rays_sampled, dataset, idx, pose_model)

    loss.backward()
    optimizer.step()
    if dataset.optimize_pose and optimize_pose:
        pose_optim.step()
        if pose_scheduler is not None:
            lr = pose_scheduler.get_last_lr()[0]
            logs["pose_lr"] = lr

    if lr_scheduler is not None:
        lr = lr_scheduler.get_last_lr()[0]
        logs["lr"] = lr

    return logs


if __name__ == "__main__":
    import attridict, yaml

    config = "./config/config.yaml"
    with open(config, "r") as file:
        config = attridict(yaml.safe_load(file))

    dataset = PanoradarRadarDataset(config, train=False)
