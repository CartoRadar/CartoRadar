"""This is the dataset registration file for SLAM trajectories.
All of them are collected when the robot is moving.
"""

from pathlib import Path
import numpy as np
from functools import partial
from typing import List, Dict

from detectron2.data import MetadataCatalog, DatasetCatalog


def register_dataset(cfg):
    """Register all the custom datasets that are used."""
    base_path = Path(cfg.DATASETS.BASE_PATH)

    # define trajectories
    lobo_building3_test = sorted(base_path.glob('building3'))
    lobo_building4_test = sorted(base_path.glob('building4'))
    lobo_building1_test = sorted(base_path.glob('building1'))
    lobo_building2_test = sorted(base_path.glob('building2'))
    lobo_building5_test = sorted(base_path.glob('building5'))

    lobo_building3_train = sorted(lobo_building4_test + lobo_building1_test + lobo_building2_test + lobo_building5_test)
    lobo_building4_train = sorted(lobo_building3_test + lobo_building1_test + lobo_building2_test + lobo_building5_test)
    lobo_building1_train = sorted(lobo_building3_test + lobo_building4_test + lobo_building2_test + lobo_building5_test)
    lobo_building2_train = sorted(lobo_building3_test + lobo_building4_test + lobo_building1_test + lobo_building5_test)
    lobo_building5_train = sorted(lobo_building3_test + lobo_building4_test + lobo_building1_test + lobo_building2_test)

    # *********************  LOBO (Leave one building out)  *********************
    DatasetCatalog.register('lobo_building3_train', partial(get_dataset_dicts, lobo_building3_train))
    DatasetCatalog.register('lobo_building3_test', partial(get_dataset_dicts, lobo_building3_test))
    MetadataCatalog.get('lobo_building3_test').set(vis_ind=get_vis_indices(lobo_building3_test))

    DatasetCatalog.register('lobo_building4_train', partial(get_dataset_dicts, lobo_building4_train))
    DatasetCatalog.register('lobo_building4_test', partial(get_dataset_dicts, lobo_building4_test))
    MetadataCatalog.get('lobo_building4_test').set(vis_ind=get_vis_indices(lobo_building4_test))

    DatasetCatalog.register('lobo_building1_train', partial(get_dataset_dicts, lobo_building1_train))
    DatasetCatalog.register('lobo_building1_test', partial(get_dataset_dicts, lobo_building1_test))
    MetadataCatalog.get('lobo_building1_test').set(vis_ind=get_vis_indices(lobo_building1_test))

    DatasetCatalog.register('lobo_building2_train', partial(get_dataset_dicts, lobo_building2_train))
    DatasetCatalog.register('lobo_building2_test', partial(get_dataset_dicts, lobo_building2_test))
    MetadataCatalog.get('lobo_building2_test').set(vis_ind=get_vis_indices(lobo_building2_test))

    DatasetCatalog.register('lobo_building5_train', partial(get_dataset_dicts, lobo_building5_train))
    DatasetCatalog.register('lobo_building5_test', partial(get_dataset_dicts, lobo_building5_test))
    MetadataCatalog.get('lobo_building5_test').set(vis_ind=get_vis_indices(lobo_building5_test))


def get_dataset_dicts(traj_paths: List[Path]) -> List[Dict]:
    """Get the dataset dict from disk.

    NOTE: It only sets the file names. The dataset mapper in `mapper.py`
    will load the actual content and add them to the dict.

    Args:
        traj_paths: list of trajectory path base/building/trajectory
    Returns:
        Dataset Dict: [
           {'file_name', 'image_id', 'height', 'width',
            'depth_file_name', 'glass_file_name'},
        ...]
    """
    dataset_dicts = []
    image_id = 0

    for traj_path in traj_paths:
        rf_npy_names = sorted((traj_path / Path('rf_npy')).iterdir())
        lidar_npy_names = sorted((traj_path / Path('lidar_npy')).iterdir())
        glass_npy_names = sorted((traj_path / Path('glass_npy')).iterdir())

        for rf_npy_name, lidar_npy_name, glass_npy_name in zip(rf_npy_names, lidar_npy_names, glass_npy_names):

            record = {
                'file_name': str(rf_npy_name),
                'depth_file_name': str(lidar_npy_name),
                'glass_file_name': str(glass_npy_name),
                'image_id': image_id,
                'height': 64,
                'width': 512,
            }
            dataset_dicts.append(record)
            image_id += 1

    return dataset_dicts


def get_vis_indices(val_trajs: List[str]) -> List[int]:
    """Get the validation indices for logging images.
    Select the first and the middle one for each trajectory.

    Args:
        val_trajs: the validation trajectories
    Returns:
        vis_indices: the visualization indices for logging images
    """
    num_traj_files = [len(list((traj_path / Path('rf_npy')).iterdir())) for traj_path in val_trajs]
    num_traj_files.insert(0, 0)
    traj_start_ind = np.cumsum(num_traj_files)  # [0, num_traj1, num_traj1+num_traj2, ...]

    # select the first one and the middle one
    vis_indices = []
    for i in range(1, len(traj_start_ind)):
        vis_indices.append(traj_start_ind[i - 1])
        vis_indices.append(traj_start_ind[i - 1] + num_traj_files[i] // 2)

    return vis_indices
