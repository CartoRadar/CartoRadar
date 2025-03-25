# This file is used to define the RadarOccNerf model
import tinycudann as tcnn
import torch.nn as nn
import torch
import torch.nn.functional as F
from dataset.dataset_getitem import tensor_sample_pts, tensor_get_label_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RadarOccNerf(nn.Module):
    def __init__(self, cfg):
        super(RadarOccNerf, self).__init__()
        print("using RadarOccNerf Model")
        self.cfg = cfg
        self.name = "RadarOccNerf"
        pos_encoding_sigma = self.cfg["pos_encoding_sigma"]
        sigma_network = self.cfg["sigma_network"]

        # define the occupancy field network
        self._model_sigma = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=1,
            encoding_config=pos_encoding_sigma,
            network_config=sigma_network,
        )

        # set the flag for the pose optimization
        if self.cfg.optimize_pose:
            self.optimize_pose = True
        else:
            self.optimize_pose = False

        print("pos_encoding_sigma is ", pos_encoding_sigma)
        print("sigma_network is ", sigma_network)

    def get_loss(self, batch, dataset, optimize_pose=False, pose_model=None):
        """
        Args:
            batch: (idx, indices, frame_idx, pose_indices) idx represents the pre-allocated batch idx; frame_idx is valid in batch_training mode represengint the selceted frame idx
            indices and pose_indices are the indices of the sampled rays.
            dataset: used datset
            optimize_pose: whether to optimize the pose
            pose_model: pose model
        Returns:
            loss: loss
            pose_loss: pose loss
            logs: str log of the loss
        """
        idx, indices, frame_idx, pose_indices = batch
        loss = None
        pose_loss = None
        if not optimize_pose:
            if not dataset.batch_training:
                # query the ray
                indices = indices[0]
                frames_idx = indices // (64 * 512)
                init_pose = pose_model.get_initPose(frames_idx).to(dataset.shift_tensor.device)
                cur_pose = pose_model.get_frames(frames_idx).to(dataset.shift_tensor.device)

                global_pos = cur_pose[:, :3, 3]

                global_pos = (global_pos + dataset.shift_tensor) / dataset.scale_factor
                global_depth = dataset.global_depth.reshape(-1, 1)[indices].to(device)
                global_dir = dataset.global_dir.reshape(-1, 3)[indices].to(device)

                global_dir = torch.matmul(torch.linalg.inv(init_pose[:, :3, :3]), global_dir[..., None]).permute(
                    0, 2, 1
                )[:, 0, :3]
                global_dir = torch.matmul(cur_pose[:, :3, :3], global_dir[..., None]).permute(0, 2, 1)[:, 0, :3]

                global_conf = dataset.conf.reshape(-1, 1)[indices].to(device)
                indices = None

                # sample points on rays and get the weights for the probabilistic learning
                batch = dataset.get_item(
                    indices,
                    global_pos,
                    global_depth,
                    global_dir,
                    dataset,
                    idx,
                    global_conf,
                )

            else:
                raise ValueError("should not arrive here.")

            pts, labels, weights = batch["pts"], batch["labels"], batch["weights"]
            labels1, weights1 = batch["labels1"], batch["weights1"]

            # pass into the field network to get the occupancy value
            sigma = self.forward(pts)

            # loss by probabilistic learning, you can add more loss here
            bce_loss = F.binary_cross_entropy_with_logits(
                torch.cat((sigma, sigma)), torch.cat((labels, labels1)), torch.cat((weights, weights1))
            )
            loss = bce_loss

            # log the loss
            logs = {"loss": loss.item(), "bce_loss": bce_loss.item()}

        else:
            # query the ray
            pose_indices = pose_indices[0]
            frames_idx = pose_indices // (64 * 512)

            init_pose = pose_model.get_initPose(frames_idx).to(dataset.shift_tensor.device)
            cur_pose = pose_model.get_frames(frames_idx).to(dataset.shift_tensor.device)
            pose_pos = cur_pose[:, :3, 3]
            pose_pos = (pose_pos + dataset.shift_tensor) / dataset.scale_factor
            pose_depth = dataset.global_depth.reshape(-1, 1)[pose_indices].to(device)

            pose_dir = dataset.global_dir.reshape(-1, 3)[pose_indices].to(device)
            pose_dir = torch.matmul(torch.linalg.inv(init_pose[:, :3, :3]), pose_dir[..., None]).permute(0, 2, 1)[
                :, 0, :3
            ]
            pose_dir = torch.matmul(cur_pose[:, :3, :3], pose_dir[..., None]).permute(0, 2, 1)[:, 0, :3]

            pose_conf = dataset.conf.reshape(-1, 1)[pose_indices].to(device)
            indices = None

            # sample points on rays and get the weights for the probabilistic learning
            pose_batch = dataset.get_item(
                indices,
                pose_pos,
                pose_depth,
                pose_dir,
                dataset,
                idx,
                pose_conf,
            )

            pose_pts, pose_labels, pose_weights = (
                pose_batch["pts"],
                pose_batch["labels"],
                pose_batch["weights"],
            )
            labels1, weights1 = pose_batch["labels1"], pose_batch["weights1"]

            # pass into the field network to get the occupancy value
            pose_sigma = self.forward(pose_pts)

            # loss by probabilistic learning, you can add more loss here
            pose_bce_loss = F.binary_cross_entropy_with_logits(
                torch.cat((pose_sigma, pose_sigma)),
                torch.cat((pose_labels, labels1)),
                torch.cat((pose_weights, weights1)),
            )

            pose_loss = pose_bce_loss

            # log the loss
            logs = {
                "pose_loss": pose_loss.item(),
                "pose_bce_loss": pose_bce_loss.item(),
            }


        return loss, pose_loss, logs

    def online_get_loss(self, rays_sampled, dataset, idx=None, pose_model=None):
        """
        Args:
            rays_sampled: sampled rays [N, 6] frame_idx, elevation, azimuth, radar_d, lidar_d, conf
            dataset: dataset
            idx: idx
            pose_model: pose model
        Returns:
            loss: loss
            logs: str log of the loss"""
        if pose_model is None:
            raise ValueError()
        # query the ray information
        # rays_sampled [N, 6] frame_idx, elevation, azimuth, radar_d, lidar_d, conf
        frames_idx = rays_sampled[:, 0]
        ray_poses = pose_model.get_frames(frames_idx)
        ray_poses[:, :3, 3] = (ray_poses[:, :3, 3] + dataset.shift_tensor.cuda()) / dataset.scale_factor
        global_pos = ray_poses[:, :3, 3]
        local_dir = pose_model.get_localDir(rays_sampled[:, 1:2], rays_sampled[:, 2:3])
        global_dir = (ray_poses[:, :3, :3] @ local_dir.unsqueeze(-1)).squeeze(-1)
        global_depth = rays_sampled[:, 3:4]
        global_conf = rays_sampled[:, 5:6]
        valid_mask = (global_depth > dataset.scaled_ray_range[0])[:, 0]

        # sample pts and assign the weights for the probabilistic learning
        pts, z_vals = tensor_sample_pts(
            global_pos[valid_mask],
            global_dir[valid_mask],
            global_depth[valid_mask],
            dataset.scaled_ray_range[0],
            dataset.scaled_ray_range[1],
            dataset.ray_pts,
        )
        labels0, weights0, labels1, weights1= tensor_get_label_weight(
            global_depth[valid_mask],
            global_conf[valid_mask],
            z_vals,
            idx,
            dataset.ray_pts,
            dataset.scaled_ray_range[0],
            dataset.scaled_ray_range[1],
            dataset.scale_factor,
            dataset.use_uncertainty,
            dataset.input_epsilon,
        )

        # pass into the field network to get the occupancy value and get the loss
        sigma = self.forward(pts.view(-1, 3))
        labels0, weights0, labels1, weights1 = labels0.view(-1, 1), weights0.view(-1, 1), labels1.view(-1, 1), weights1.view(-1, 1)
        bce_loss = F.binary_cross_entropy_with_logits(
            torch.cat((sigma, sigma)),
            torch.cat((labels0, labels1)),
            torch.cat((weights0, weights1)),
        )
        loss = bce_loss

        # log the loss
        logs = {"loss": loss.item(), "bce_loss": bce_loss.item()}

        return loss, logs

    def forward(self, pts):
        """
        Args:
            pts: points [N, 3]
        Returns:
            sigma: occupancy value [N, 1]
        """

        pts = (pts + 1) / 2
        sigma = self._model_sigma(pts)
        return sigma

    def render(self, pts=None, z_vals=None):
        """
        Args:
            pts: points [ray_number, point_number_per_ray, 3(xyz)]
            z_vals: z values [ray_number, point_number_per_ray]
        Returns:
            ret: dictionary of the rendered results, including the depth map, alpha, weights, z_vals, alpha_scaled
        """
        pts_num = z_vals.shape[-1]
        # get occupancy
        sigma = self(pts.view(-1, 3))

        # render the range value
        sigma = torch.sigmoid(sigma).view(-1, pts_num)

        z_vals = z_vals.view(-1, pts_num)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat(
            [
                dists,
                torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape),
            ],
            -1,
        )
        alpha = sigma**10
        weights = (alpha) * torch.cumprod(
            torch.cat(
                [
                    torch.ones((alpha.shape[0], 1), device=alpha.device),
                    (1.0 - alpha) + 1e-10,
                ],
                -1,
            ),
            -1,
        )[:, :-1]
        arg_idx = torch.argmax(weights, dim=1)
        depth_map = z_vals[list(range(z_vals.shape[0])), arg_idx]

        ret = {
            "depth_map": depth_map,
            "alpha": sigma,
            "weights": weights,
            "z_vals": z_vals,
            "alpha_scaled": alpha,
        }
        return ret