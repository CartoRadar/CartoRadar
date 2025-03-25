"""A mapper change the dataset dict to a batch dict that is ready for the model.
Data Augmentation also happens here.

Dataset Dict: {'file_name', 'image_id', 'height', 'width', 
    'annotations': {'bbox', 'bbox_mode', 'segmentation', 'category_id'}
}

Batch Dict: {# Can keep all the keys in the dataset dict, but must have
    'image': torch.tensor(C,H,W), 
    'instances': The detectron2.structures.Instances object
    'sem_seg': torch.tensor(C,H,W), 
}
"""

import numpy as np
import copy, cv2
from typing import List, Dict, Tuple
import torch


# ==========================================================
# =====================  Augmentation  =====================
# ==========================================================
def crop_and_resize(
    image: np.ndarray,
    glass_seg: np.ndarray,
    max_crop_length=(16, 16),
    crop_and_resize_p=0.5,
) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
    """Crop and resize augmentation.
    Args:
        image: RGB or Lidar npy image, shape (C, H, W)
        glass_seg: glass segmentation npy image, shape (H, W)
        max_crop_length: the length of the crop, (half height, half width)
        crop_and_resize_p: the probability for crop and resize
    Returns:
        image_aug, annos_aug, sem_seg_aug: the augmented image, annotation and segmentation
    """
    if np.random.rand() < crop_and_resize_p:
        H, W = image.shape[1:]
        crop_offset_h0 = np.random.randint(0, max_crop_length[0] + 1)
        crop_offset_w0 = np.random.randint(0, max_crop_length[1] + 1)
        crop_offset_h1 = H - crop_offset_h0
        crop_offset_w1 = W - crop_offset_w0

        # image cropping and resize
        image = image[:, crop_offset_h0:crop_offset_h1, crop_offset_w0:crop_offset_w1]
        image = cv2.resize(image.transpose(1, 2, 0), (W, H), interpolation=cv2.INTER_LINEAR)
        image = image.transpose(2, 0, 1)  # back to (C, H, W)

        glass_seg = glass_seg[crop_offset_h0:crop_offset_h1, crop_offset_w0:crop_offset_w1]
        glass_seg = cv2.resize(glass_seg, (W, H), interpolation=cv2.INTER_NEAREST)

    return image, glass_seg


def jitter_image(image: np.ndarray, mean=0.0, std=0.003, jitter_p=0.5) -> np.ndarray:
    """Jitter the image (or depth lidar npy) with Gaussian noise.
    Args:
        image: RGB or Lidar npy image, shape (C, H, W)
        mean, std: the mean and standard deviation for the Guassian distribution
        jitter_p: the probability for jittering
    Returns:
        image_aug: the augmented image
    """
    if np.random.rand() < jitter_p:
        jitter = np.random.randn(*image.shape) * std + mean
        image = image + jitter
    return image


def scaling_transform(image: np.ndarray, scale_range=(0.8, 1.2), scaling_p=0.5) -> np.ndarray:
    """Scale transform for the image (or depth lidar npy).
    Args:
        image: RGB or Lidar npy image, shape (C, H, W)
        scale_range: the (min, max) ratio for scaling
        scaling_p: the probability for the scaling
    Returns:
        image_aug: the augmented image
    """
    if np.random.rand() < scaling_p:
        scale = np.random.uniform(*scale_range)
        image = image * scale
    return image


def mix_after_first_reflection(
    image0: np.ndarray, depth0: np.ndarray, image1: np.ndarray, jitter_p: float = 0.5
) -> np.ndarray:
    """Find the range bin of the first reflection from lidar.
    Then jitter the values in range bins after it.
    Args:
        image0: rf data to be augmented, (256, 64, 512)
        depth0: the lidar depth data, (1, 64, 512)
        image1: rf data used to augment image0, (256, 64, 512)
        jitter_p: the probability to do the jitter augmentation
    Return:
        refl_data: The data for mixing after the first reflection
    """
    y_per_bin = 0.003747
    guard_bin = 3

    if np.random.rand() < jitter_p:
        C, H, W = image0.shape
        start_bin = depth0 / y_per_bin + guard_bin + np.random.randint(-2, 3)  # (1,H,W)
        start_bin[start_bin < 0] = C  # don't jitter failure and glass region
        mask = np.arange(0, C, dtype=np.float32).reshape(-1, 1, 1)
        mask = np.tile(mask, (1, H, W)) > start_bin
        image0 = image0 + (image1 - image0) * mask * 0.5

    return image0


# ==========================================================
# ========================  Mapper  ========================
# ==========================================================
class RfMapper:
    def __init__(self, cfg, is_train: bool):
        """Load and map the rf npy files.
        The four tasks are: depth, surface normal, semantic seg, obj detection

        The callable currently does the following:
        1. Read the image from "file_name"
        2. Applies augmentation to the image and annotations
        3. Prepare data and annotations to Tensor and :class:`Instances`

        Args:
            cfg: the config object, CfgNode
            is_train: whether it's for training, control augmentation
        """
        self.cfg = cfg
        self.instance_mask_format = cfg.INPUT.MASK_FORMAT

        self.is_train = is_train
        self.need_rotate_aug = cfg.INPUT.ROTATE.ENABLED
        self.rotate_p = cfg.INPUT.ROTATE.ROTATE_P
        self.hflip_p = cfg.INPUT.ROTATE.HFLIP_P

        self.need_jitter_aug = cfg.INPUT.JITTER.ENABLED
        self.jitter_mean = cfg.INPUT.JITTER.MEAN
        self.jitter_std = cfg.INPUT.JITTER.STD
        self.jitter_p = cfg.INPUT.JITTER.JITTER_P

        self.need_first_refl_aug = cfg.INPUT.FIRST_REFL.ENABLED
        self.first_refl_p = cfg.INPUT.FIRST_REFL.JITTER_P

        self.prev_data = None  # for mix_after_first_reflection

    def __call__(self, dataset_dict):
        """Do the mapping for the dataset_dict
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # load data, image of float32, sem_seg of uint8
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = np.load(dataset_dict['file_name'])  # rf (256,H=64,W=512)
        depth = np.load(dataset_dict['depth_file_name'])  # shape (1,H=64,W=512)
        glass = np.load(dataset_dict['glass_file_name'])  # shape (1,H=64,W=512)

        # mask glass region out
        depth[glass] = -1e3

        # deal with augmentation
        if self.need_rotate_aug and self.is_train:
            image, depth = self.rotate_and_hflip(image, depth, self.rotate_p, self.hflip_p)

        if self.need_first_refl_aug and self.is_train:
            if self.prev_data is None:
                self.prev_data = image  # store it initially
            else:
                image_ = mix_after_first_reflection(image, depth, self.prev_data, self.first_refl_p)
                self.prev_data = image
                image = image_

        if self.need_jitter_aug and self.is_train:
            image = jitter_image(image, self.jitter_mean, self.jitter_std, self.jitter_p)

        # save to the dataset dict
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image, np.float32))
        dataset_dict["depth"] = torch.as_tensor(np.ascontiguousarray(depth, np.float32))

        return dataset_dict

    @staticmethod
    def rotate_and_hflip(
        image: np.ndarray,
        depth: np.ndarray,
        rotate_p=1.0,
        hflip_p=0.5,
    ) -> Tuple[np.ndarray, List[Dict], np.ndarray, np.ndarray, np.ndarray]:
        """Rotation and horizontal flip augmentation.
        Args:
            image: RGB or Lidar npy image, shape (C, H, W)
            annos: the annotation dict in detectron2 format
            sem_seg: semantic segmentation npy image, shape (H, W)
            depth: depth map, shape (1, H, W)
            normal: surface normal, shape (3, H, W)
            rotate_p, flip_p: the probabilities for rotation and hflip
        Returns:
            image_aug, annos_aug: the augmented image and annotation
        """
        WIDTH = image.shape[-1]

        if np.random.rand() < rotate_p:
            rot_ind = int(WIDTH * np.random.rand())
            image = np.concatenate((image[:, :, rot_ind:], image[:, :, :rot_ind]), axis=-1)
            depth = np.concatenate((depth[:, :, rot_ind:], depth[:, :, :rot_ind]), axis=-1)

        if np.random.rand() < hflip_p:
            image = np.flip(image, axis=-1)
            depth = np.flip(depth, axis=-1)

        return image, depth
