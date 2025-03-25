import numpy as np
import torch

# OnlineSlamDataset is the mentioned shared queue in paper

class OnlineSlamDataset:
    def __init__(self, number_frames):
        ### everything is in the normailized coordinate
        self.global_rays = (
            torch.empty([number_frames * 512 * 64, 6]).cuda().share_memory_()
        )  # [N, 6] frame_idx, elevation, azimuth, radar_d, lidar_d, conf
        self.number_frames = torch.tensor(0).share_memory_()

        elevation_angles = np.pi / 4 - np.pi / 2 / (64 - 1) * np.arange(
            64, dtype=np.float32
        )
        azimuth_angles = -2 * np.pi / 512 * np.arange(512, dtype=np.float32)
        elevation_grid, azimuth_grid = np.meshgrid(
            elevation_angles, azimuth_angles, indexing="ij"
        )

        # initialize the local direction to save computation overhead
        self.local_dir = np.vstack(
            (
                (np.cos(azimuth_grid) * np.cos(elevation_grid))[None, ...],
                (np.sin(azimuth_grid) * np.cos(elevation_grid))[None, ...],
                np.sin(elevation_grid)[None, ...],
            )
        )
        self.local_dir = torch.tensor(self.local_dir).cuda()
        self.local_dir_rays = self.local_dir.reshape(3, -1)

        rows, cols = torch.meshgrid(torch.arange(64), torch.arange(512), indexing="ij")
        self.flat_rows = rows.flatten()
        self.flat_cols = cols.flatten()

        self.window_size = 5
        self.num_temporal_frames = 1

        self.random_idx = 0

    def generate_sample_indice(self, chunk):
        # generate indices after all the data is loaded
        self.indices = torch.cat(
            [
                torch.randperm(self.number_frames * 64 * 512)
                for _ in range(20000 * chunk // (self.number_frames * 64 * 512) + 1)
            ]
        )

    def add_data(self, item):
        # add frame information into the shared queue
        [cur_depth, _, cur_lidar_depth, cur_conf] = item
        cur_rays = torch.stack(
            (
                torch.ones_like(self.flat_cols) * self.number_frames,
                self.flat_rows,
                self.flat_cols,
                cur_depth.flatten(),
                cur_lidar_depth.flatten(),
                cur_conf.flatten(),
            ),
            dim=-1,
        )

        self.global_rays[
            self.number_frames * 64 * 512 : (self.number_frames + 1) * 64 * 512
        ] = cur_rays.cuda()
        self.number_frames += 1

    def sample_data(self, number_rays):
        # sample rays from teh shared queue
        if number_rays == -1:
            return self.global_rays
        num_temporal_frames = min(
            self.num_temporal_frames, self.number_frames, self.window_size
        )
        indices = torch.randperm(self.number_frames - num_temporal_frames)[
            : self.window_size - num_temporal_frames
        ].tolist()

        indices += list(
            i + int(self.number_frames.item()) for i in range(-num_temporal_frames, 0)
        )

        candidate_ray_idx = (
            torch.arange(64 * 512)[None, ...] + (torch.tensor([indices]) * 64 * 512).T
        )
        candidate_ray_idx = candidate_ray_idx.flatten()
        shuffle = torch.randperm(candidate_ray_idx.shape[0])[:number_rays]
        out = self.global_rays[candidate_ray_idx[shuffle]]

        return out

    def sample_all_data(self, number_rays):
        # sample if we have loaded all the rays
        if number_rays == -1:
            return self.global_rays
        ray_index = self.indices[
            self.random_idx * number_rays : (self.random_idx + 1) * number_rays
        ]
        self.random_idx += 1

        out = self.global_rays[ray_index]

        return out

    def fetch_frame_rays(self, frame_idx):
        # sample rays for a specific frame
        # [N, 6] frame_idx, elevation, azimuth, radar_d, lidar_d, conf
        rays_depth = self.global_rays[
            frame_idx * 64 * 512 : (frame_idx + 1) * 64 * 512
        ][:, 3:4]
        rays_conf = self.global_rays[frame_idx * 64 * 512 : (frame_idx + 1) * 64 * 512][
            :, -1:
        ]

        return rays_depth, rays_conf
