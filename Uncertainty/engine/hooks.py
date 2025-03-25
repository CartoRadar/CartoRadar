import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter

from detectron2.data import MetadataCatalog
from detectron2.engine import HookBase


def project_polar_to_cartesian(
    heatmap: np.ndarray,
    r_rings: np.ndarray,
    max_range: float,
    grid_size: float,
    beam_angles: np.ndarray,
    default_value: float = 0.0,
    rotation_offset: float = 0,
) -> np.ndarray:
    """Project the polar system imaging result to Cartesian system.
    Args:
        heatmap: the polar system imaging result, (N_rings, N_beams)
        r_rings: radius of each ring, in meter, shape (N_rings, )
        max_range: maximum projection range, m x m image
        grid_size: the actual size of each grid/pixel
        beam_angles: the facing angle of each beam
    Return:
        proj_heatmap: the Cartesian system imaging result
    """
    PROJ_MAP_SZ = int(2 * max_range / grid_size)  # size of the projected heatmap
    proj_heatmap = np.full((PROJ_MAP_SZ, PROJ_MAP_SZ), default_value, dtype=np.float32)
    N_rings, N_beams = heatmap.shape

    cos_phi = np.cos(beam_angles + rotation_offset)
    sin_phi = np.sin(beam_angles + rotation_offset)

    # project polar to Cartesian
    for ring_id in range(0, N_rings):
        x_grid_id = (r_rings[ring_id] * cos_phi + max_range) / grid_size
        y_grid_id = (r_rings[ring_id] * sin_phi + max_range) / grid_size
        x_grid_id = np.round(x_grid_id).astype(np.int32)
        y_grid_id = np.round(y_grid_id).astype(np.int32)

        # bound to PROJ_MAP_SZ
        valid = np.logical_and(
            np.logical_and(x_grid_id >= 0, x_grid_id < PROJ_MAP_SZ),
            np.logical_and(y_grid_id >= 0, y_grid_id < PROJ_MAP_SZ),
        )

        proj_heatmap[y_grid_id[valid], x_grid_id[valid]] = heatmap[ring_id][valid]

    return proj_heatmap


class ImageVisHook(HookBase):
    def __init__(self, cfg):
        """The hook for visualizing validation set images and log them to tensorboard.
        Args:
            cfg: the config object
        """
        super().__init__()
        self.cfg = cfg
        self.period = cfg.VIS_PERIOD
        self.writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)  # tensorboard
        self.dataset_name = cfg.DATASETS.TEST[0]
        self.dataset_metadata = MetadataCatalog.get(self.dataset_name)
        self.vis_indices = self.dataset_metadata.vis_ind

    def after_step(self):
        if (self.trainer.iter + 1) % self.period != 0:
            return

        dataloader = self.trainer.build_test_loader(self.cfg, self.dataset_name)
        self.trainer.model.eval()

        with torch.no_grad():
            vis_title = 0  # in the tensorboard image title

            for idx, input_dict in enumerate(dataloader):
                if idx not in self.vis_indices:
                    continue

                # get model prediction, draw image, and log it to the tensorboard
                outputs = self.trainer.model(input_dict)[0]
                whole_img = draw_vis_image(
                    input_dict[0],
                    outputs,
                    self.dataset_metadata,
                    True,
                    return_rgb=True,
                )
                self.writer.add_image(f"val image {vis_title}", whole_img, global_step=self.trainer.iter)
                vis_title += 1

        self.trainer.model.train()


def draw_vis_image(
    input_dict,
    outputs,
    dataset_metadata,
    need_depth: bool,
    return_rgb: bool = True,
):
    """draw the image for visualization.
    Args:
        input_dict: the input dictionary, not a list
        preds: the output predictions dictionary from the model, not a list
        dataset_metadata: the metadata from `MetadataCatalog.get
        need_seg_obj: whether semantic segmentation and object are there
        return_rgb: If true, return a rgb image for tensorboard, else return fig and ax.
    """
    # prepare the lidar Magma colormap
    gt_depth = input_dict['depth'].cpu().squeeze().numpy()  # (64, 512)
    pred_depth = outputs[0].cpu().squeeze().numpy()  # (64, 512)
    pred_var = outputs[1].cpu().squeeze().numpy()  # (64, 512)

    # draw depth estimation
    rows = 5
    fig, ax = plt.subplots(
        rows,
        2,
        gridspec_kw={'width_ratios': [4, 1]},
        figsize=(8, 2 * rows),
    )

    if need_depth:
        _plot_depth(
            input_dict['image'].cpu().permute(1, 0, 2).numpy(),
            pred_depth,
            pred_var,
            gt_depth,
            ax[:5],
        )

    fig.tight_layout()

    if return_rgb:
        fig.canvas.draw()
        whole_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        whole_img = whole_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        whole_img = whole_img.transpose(2, 0, 1)  # (3, H, W)
        plt.close()
        return whole_img
    else:
        return fig, ax


def _plot_depth(
    rf_raw: np.ndarray, pred_depth: np.ndarray, pred_depth_var: np.ndarray, y_depth: np.ndarray, ax: np.ndarray
):
    """Private function for plotting depth result.
    Args:
        rf_raw: the raw RF heatmap, in (H,C,W)
        pred_depth: the predicted depth image, in (H,W)
        pred_depth_var: the predicted variance, in (H,W)
        y_depth: the ground truth depth image, in (H,W)
        ax: the matplotlib ax objects
    """
    N_rings = rf_raw.shape[1]
    N_beams = rf_raw.shape[2]
    N_upsample_beams = N_beams * 6
    r_rings = 0.037474 * np.arange(N_rings)
    mask = y_depth > 0

    # transform to 2D pc floor plan
    thetas = np.linspace(0, -2 * np.pi, N_beams)
    valid_inds = (mask == 1)[31]
    lidar_slice = y_depth[31].copy()
    pred_slice = pred_depth[31].copy()
    lidar_slice[lidar_slice > 0.96] = 2.0  # out of range
    pred_slice[pred_slice > 0.96] = 2.0  # out of range
    #
    pc_x = lidar_slice * np.cos(thetas) * 10
    pc_y = lidar_slice * np.sin(thetas) * 10
    pc_x, pc_y = pc_y[valid_inds], -pc_x[valid_inds]
    out_x = pred_slice * np.cos(thetas) * 10
    out_y = pred_slice * np.sin(thetas) * 10
    out_x, out_y = out_y[valid_inds], -out_x[valid_inds]

    # transform 2D polar to Cartesian
    img_x = cv2.resize(rf_raw[31], (N_upsample_beams, N_rings))  # width, height
    img_x = project_polar_to_cartesian(
        img_x,
        r_rings,
        0.96 * 10,
        0.04,
        np.linspace(0, -2 * np.pi, N_upsample_beams),
        default_value=np.nan,
        rotation_offset=-np.pi / 2,
    )

    y_depth = y_depth.copy()
    y_depth[mask == 0] = np.nan

    # Draw image visualization
    ax[0, 0].imshow(rf_raw[31], cmap='jet', origin='lower', vmin=0.1, vmax=0.5, aspect='auto')
    ax[0, 0].axis('off')
    ax[0, 1].imshow(img_x, origin='lower', vmin=-0.1, vmax=0.5, cmap='jet')
    ax[0, 1].axis('off')
    ax[1, 0].imshow(-y_depth, aspect='auto', cmap='magma', vmin=-0.8, vmax=-0.05)
    ax[1, 0].axis('off')
    ax[1, 1].scatter(pc_x, pc_y, s=1)
    ax[1, 1].scatter(0, 0, c='red')
    ax[1, 1].set_xlim([-10, 10])
    ax[1, 1].set_ylim([-10, 10])
    ax[1, 1].set_aspect('equal')
    ax[1, 1].set_xticklabels([])
    ax[1, 1].set_yticklabels([])
    ax[1, 1].grid(True, which='both')
    ax[2, 0].imshow(-pred_depth, aspect='auto', cmap='magma', vmin=-0.8, vmax=-0.05)
    ax[2, 0].axis('off')
    ax[2, 1].scatter(out_x, out_y, s=1)
    ax[2, 1].scatter(0, 0, c='red')
    ax[2, 1].set_xlim([-10, 10])
    ax[2, 1].set_ylim([-10, 10])
    ax[2, 1].set_aspect('equal')
    ax[2, 1].set_xticklabels([])
    ax[2, 1].set_yticklabels([])
    ax[2, 1].grid(True, which='both')
    #
    l1_error = np.abs(pred_depth - y_depth)
    l1_error[mask == 0] = np.nan
    ax[3, 0].imshow(l1_error, aspect='auto', cmap='jet', vmax=0.3)
    ax[3, 0].axis('off')
    ax[3, 1].axis('off')
    ax[4, 0].imshow(pred_depth_var, aspect='auto', cmap='jet')
    ax[4, 0].axis('off')
    ax[4, 1].axis('off')
