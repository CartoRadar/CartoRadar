import pickle
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d
import time
import configargparse
from scipy.spatial import KDTree
from model.PoseLearn import compute_ATE, compute_rpe
import json, copy
from pathlib import Path
from typing import Union, Dict
from tqdm import tqdm
import natsort, os


def parse_arguments():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True, default=None, help="target folder.")
    parser.add_argument("--iter_num", type=int, default=None, help="assign the model for evaluation.")
    return parser.parse_args()


def compute_3D_distance(
    point_set1: Union[np.ndarray, KDTree], point_set2: np.ndarray, truncation_dist=np.inf, ignore_outlier=True
) -> np.ndarray:
    """For each kd-point in point_set2, find the nearest distance to point_set1.
    Args:
        point_set1: k-dimensional point sets, shape (N1, K), or scipy KDTree
        point_set2: k-dimensional point sets, shape (N2, K)
        truncation_dist: a threshold for avoiding outliers
        ignore_outlier: if False, will set d>truncation_dist to be truncation_dist;
            if True, will ignore those d that d>truncation_dist.
    Returns:
        distances: shape (<=N2, K)
    """
    ps1_tree = KDTree(point_set1) if isinstance(point_set1, np.ndarray) else point_set1
    distances, _ = ps1_tree.query(point_set2, workers=-1)

    if ignore_outlier:
        select = distances < truncation_dist
        distances = distances[select]
    else:
        distances = np.clip(distances, None, truncation_dist)
    return distances


def eval_mesh_wtraj(
    render_pcd,
    gt_pcd,
    render_align,
    down_sample_res=0.02,
    threshold=0.1,
    truncation_acc=0.50,
    truncation_com=0.50,
    gt_bbx_mask_on=True,
    mesh_sample_point=2000000,
    possion_sample_init_factor=5,
):
    """Compute Mesh metrics between prediction and target.
    Opens the Meshs and runs the metrics
    Args:
        data_dict: from where we get aligned pred point cloud
        file_trgt: file path of target (shoud be mesh)
        down_sample_res: use voxel_downsample to uniformly sample mesh points
        threshold: distance threshold used to compute precision/recall
        truncation_acc: points whose nearest neighbor is farther than the distance would not be taken into account (take pred as reference)
        truncation_com: points whose nearest neighbor is farther than the distance would not be taken into account (take trgt as reference)
        gt_bbx_mask_on: use the bounding box of the trgt as a mask of the pred mesh
        mesh_sample_point: number of the sampling points from the mesh
        possion_sample_init_factor: used for possion uniform sampling, check open3d for more details (deprecated)
    Returns:

    Returns:
        Dict of mesh metrics (chamfer distance, precision, recall, f1 score, etc.)
    """
    print("calculate the mapping error w/ the trajectory")

    render_aligned_pcd = copy.deepcopy(render_pcd)
    render_aligned_pcd.transform(render_align)

    if down_sample_res > 0:
        gt_pcd = gt_pcd.voxel_down_sample(down_sample_res)
        render_aligned_pcd = render_aligned_pcd.voxel_down_sample(down_sample_res)

    render_points = np.asarray(render_aligned_pcd.points)
    gt_points = np.asarray(gt_pcd.points)
    dist_p = compute_3D_distance(gt_points, render_points, truncation_acc)
    dist_r = compute_3D_distance(render_points, gt_points, truncation_com)

    dist_p_s = np.square(dist_p)
    dist_r_s = np.square(dist_r)

    dist_p_mean = np.mean(dist_p)
    dist_r_mean = np.mean(dist_r)

    dist_p_s_mean = np.mean(dist_p_s)
    dist_r_s_mean = np.mean(dist_r_s)

    chamfer_l1 = 0.5 * (dist_p_mean + dist_r_mean)
    chamfer_l2 = np.sqrt(0.5 * (dist_p_s_mean + dist_r_s_mean))

    precision = np.mean((dist_p < threshold).astype("float"))
    recall = np.mean((dist_r < threshold).astype("float"))
    fscore = 2 * precision * recall / (precision + recall)

    metrics = {
        "mean accuracy (m)": dist_p_mean,
        "80% accuracy (m)": np.quantile(dist_p, 0.8),
        "90% accuracy (m)": np.quantile(dist_p, 0.9),
        "95% accuracy (m)": np.quantile(dist_p, 0.95),
        "mean completeness (m)": dist_r_mean,
        "80% completeness (m)": np.quantile(dist_r, 0.8),
        "90% completeness (m)": np.quantile(dist_r, 0.9),
        "95% completeness (m)": np.quantile(dist_r, 0.95),
        "Spacing (m)": down_sample_res,  # evlaution setup
        "Outlier_truncation_acc (m)": truncation_acc,  # evlaution setup
        "Outlier_truncation_com (m)": truncation_com,  # evlaution setup
        "Chamfer_L1 (m)": chamfer_l1,
        "Chamfer_L2 (m)": chamfer_l2,
        "Precision": precision,
        "Recall": recall,
        "F-score": fscore,
        "Precision Recall Threshold (m)": threshold,
    }
    return {"global_alignment": metrics}



def umeyama_4dof(source: np.ndarray, target: np.ndarray):
    """Compute the transformation such that T(Source) -> target.
    In this function we limit T to only have 4 DoF, i.e. {x,y,z,yaw}, so it has
    the form like
          [R_0  R_1  0  x]             [R_0  R_1]
        T=[R_2  R_3  0  y] ,  where R'=[R_2  R_3] \in SO(2)
          [ 0    0   1  z]
          [ 0    0   0  1]
    Args:
        source: the source point cloud, P, shape (N, 3)
        target: the target point cloud, Q, shape (N, 3)
    Returns:
        trans: 4*4 np.ndarray transformation matrix
    """
    P = source.T  # 3*N
    Q = target.T  # 3*N

    # 1. compute centroids of both point sets
    p_mean = np.mean(P, axis=1, keepdims=True)  # 3*1
    q_mean = np.mean(Q, axis=1, keepdims=True)  # 3*1

    # 2. compute centered vectors and 3*3 covariance matrix
    P_centered = P - p_mean
    Q_centered = Q - q_mean
    S_cov = P_centered @ Q_centered.T

    # 3. compute singular value decomposition S=U Sigma VT
    # Note: Since R only relates to yaw, a sub-matrix of S is used
    U, Sigma, VT = np.linalg.svd(S_cov[:2, :2])
    Sign = np.diag([1, np.linalg.det(U @ VT)])
    R_2x2 = VT.T @ Sign @ U.T  # 2*2

    # 4. compute the optimal translation
    trans = np.identity(4)
    trans[:2, :2] = R_2x2
    trans[:3, 3:] = q_mean - trans[:3, :3] @ p_mean

    return trans


def map_acc_comp_align_segment(
    render_poses: np.ndarray,
    gt_poses: np.ndarray,
    render_points: np.ndarray,
    gt_points: np.ndarray,
    render_pcd: o3d.geometry.PointCloud,
    gt_pcd: o3d.geometry.PointCloud,
    down_sample_res=0.01,
    truncation_acc=10.0,
    truncation_comp=10.0,
    segment_len: int = 15,
    precision_recall_thres=0.1,
) -> Dict:
    """Compute the mapping accuracy & completion using the aligning local segment method.
    Args:
        render_poses: the network render poses, shape (N, 4, 4)
        gt_poses: the ground truth poses, shape (N, 4, 4)
        render_points: the raw render points, shape (N_frame, M, 3)
        gt_points: the raw gt points, shape (N_frame, M, 3)
        render_pcd: the render point cloud map. Can be the one after masking
        gt_pcd: the ground truth point cloud map. Can be the one after noise removal
        down_sample_res: voxel downsampling size. -1 means no downsampling
        truncation_acc, truncation_comp: the distance for not considering the point (outlier or noise)
        segment_len: the length of the local segment
        precision_recall_thres: the threshold for the precision and recall
    Returns:
        results: a dictionary containing the accuracy and completion
    """
    print('constructing kd-tree ...')
    render_pcd_tree = KDTree(np.asarray(render_pcd.points))
    gt_pcd_tree = KDTree(np.asarray(gt_pcd.points))
    dist_acc_s = []
    dist_comp_s = []

    if down_sample_res > 0:
        print('downsampling for gt_pcd and render_pcd ...')
        gt_pcd = gt_pcd.voxel_down_sample(down_sample_res)
        render_pcd = render_pcd.voxel_down_sample(down_sample_res)

    render_ds_pcd_tree = KDTree(np.asarray(render_pcd.points))
    gt_ds_pcd_tree = KDTree(np.asarray(gt_pcd.points))

    # compute accuracy
    for ind in tqdm(range(0, len(render_points), segment_len), 'compute acc.'):
        # alignment transform
        render_pose_segment = render_poses[ind : ind + segment_len, :3, 3]
        gt_pose_segment = gt_poses[ind : ind + segment_len, :3, 3]
        align_T = umeyama_4dof(render_pose_segment, gt_pose_segment)

        # prepare render pcd segment (only keep points after masks, visual outlier, etc)
        render_points_segment = render_points[ind : ind + segment_len].reshape(-1, 3)
        point_num = render_pcd_tree.query_ball_point(render_points_segment, 0.01, workers=-1, return_length=True)
        render_pcd_segment = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(render_points_segment[point_num > 0]))
        render_pcd_segment.transform(align_T)

        if down_sample_res > 0:
            render_pcd_segment = render_pcd_segment.voxel_down_sample(down_sample_res)
        dist_acc_s.append(compute_3D_distance(gt_ds_pcd_tree, np.asarray(render_pcd_segment.points), truncation_acc))

    # compute completion
    for ind in tqdm(range(0, len(render_points), segment_len), 'compute comp.'):
        # alignment transform
        render_pose_segment = render_poses[ind : ind + segment_len, :3, 3]
        gt_pose_segment = gt_poses[ind : ind + segment_len, :3, 3]
        align_T = umeyama_4dof(gt_pose_segment, render_pose_segment)

        # prepare gt pcd segment (only keep points after masks, SOR noise removal, etc)
        gt_points_segment = gt_points[ind : ind + segment_len].reshape(-1, 3)
        point_num = gt_pcd_tree.query_ball_point(gt_points_segment, 0.01, workers=-1, return_length=True)
        gt_pcd_segment = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_points_segment[point_num > 0]))
        gt_pcd_segment.transform(align_T)

        if down_sample_res > 0:
            gt_pcd_segment = gt_pcd_segment.voxel_down_sample(down_sample_res)
        dist_comp_s.append(compute_3D_distance(render_ds_pcd_tree, np.asarray(gt_pcd_segment.points), truncation_comp))

    dist_acc_s = np.concatenate(dist_acc_s)
    dist_comp_s = np.concatenate(dist_comp_s)

    dist_acc_mean = np.mean(dist_acc_s)
    dist_comp_mean = np.mean(dist_comp_s)

    chamfer_l1 = 0.5 * (dist_acc_mean + dist_comp_mean)
    precision = np.mean(dist_acc_s < precision_recall_thres)
    recall = np.mean(dist_comp_s < precision_recall_thres)
    fscore = 2 * precision * recall / (precision + recall)

    results = {
        "mean accuracy (m)": dist_acc_mean,
        "80% accuracy (m)": np.quantile(dist_acc_s, 0.8),
        "90% accuracy (m)": np.quantile(dist_acc_s, 0.9),
        "95% accuracy (m)": np.quantile(dist_acc_s, 0.95),
        "mean completeness (m)": dist_comp_mean,
        "80% completeness (m)": np.quantile(dist_comp_s, 0.8),
        "90% completeness (m)": np.quantile(dist_comp_s, 0.9),
        "95% completeness (m)": np.quantile(dist_comp_s, 0.95),
        "segment_length": segment_len,
        "downsample_size (m)": down_sample_res,
        "outlier_truncation_acc (m)": truncation_acc,
        "outlier_truncation_comp (m)": truncation_comp,
        "Chamfer_L1 (m)": chamfer_l1,
        "Precision": precision,
        "Recall": recall,
        "F-score": fscore,
        "Precision Recall Threshold (m)": precision_recall_thres,
    }
    return {"local_alignment": results}


if __name__ == "__main__":
    # This file is used to get the complete evaluation reported in our paper.
    # You need to first run mobicom_generate_pkl.py to generate the pkl file.
    # You must assign the target experiment folder to the argument --target.
    # You can also assign the iteration number to the argument --iter_num to evaluate the model at that iteration.

    # Get arguments. 
    args = parse_arguments()
    folder = args.target
    iter_num = args.iter_num

    # Find the needed files.
    folder = Path('./output/' + folder)
    model_folder = folder / "ckpt"
    gt_mesh_sample = folder / "gt_map.pcd"
    cfg_file = next(folder.glob('*.yaml'))

    if iter_num is None:
        all_files = [f for f in os.listdir(model_folder) if os.path.isfile(os.path.join(model_folder, f))]
        model_files = [f for f in all_files if f.startswith("model_step_")]
        pose_model_files = [f for f in all_files if f.startswith("pose_model_step_")]
        render_aligned_pcd_files = folder.glob('model_step_*_aligned_map.pcd')
        render_pcd_files = folder.glob('model_step_*[0-9]_map.pcd')

        sorted_model_files = natsort.natsorted(model_files)
        sorted_pose_model_files = natsort.natsorted(pose_model_files)
        sorted_render_aligned_pcd_files = natsort.natsorted(render_aligned_pcd_files)
        sorted_render_pcd_files = natsort.natsorted(render_pcd_files)

        latest_model_file = sorted_model_files[-1]
        latest_pose_model_file = sorted_pose_model_files[-1]
        latest_render_aligned_pcd_file = sorted_render_aligned_pcd_files[-1]
        latest_render_pcd_file = sorted_render_pcd_files[-1]
    else:
        latest_model_file = "model_step_" + str(iter_num) + ".pth"
        latest_pose_model_file = "pose_model_step_" + str(iter_num) + ".pth"
        latest_render_aligned_pcd_file = "model_step_" + str(iter_num) + "_aligned_map.pcd"
        latest_render_pcd_file = folder / ("model_step_" + str(iter_num) + "_map.pcd")

    # create the output file
    pkl_add = folder / (latest_model_file[:-4] + "_result" + ".pkl")
    adds = str(pkl_add).split("/")
    adds[-1] = "analysis_th30_th10_Normalth10_debug" + adds[-1][:-4] + ".json"
    write_add = "/".join(adds)

    with open(pkl_add, "rb") as f:
        data_dict = pickle.load(f)

    logs = {}

    # Load the data
    gt_mesh_sample = o3d.io.read_point_cloud(str(gt_mesh_sample))
    _, ind = gt_mesh_sample.remove_statistical_outlier(nb_neighbors=10, std_ratio=3.0)
    gt_mesh_sample = gt_mesh_sample.select_by_index(ind)
    render_pcd = o3d.io.read_point_cloud(str(latest_render_pcd_file))

    # evaluate the global mapping performance
    wtraj_err = eval_mesh_wtraj(
        render_pcd,
        gt_mesh_sample,
        data_dict["render_align"],
        down_sample_res=0.01,
        truncation_acc=10.0,
        truncation_com=10.0,
        threshold=0.10,
    )
    logs.update(wtraj_err)
    logs.update(data_dict["L1_err"])
    
    # evaluate the local mapping performance
    local_map_err = map_acc_comp_align_segment(
        data_dict['render_poses'],
        data_dict['gt_poses'],
        np.load(folder / "render_points.npy"),
        np.load(folder / "gt_points.npy"),
        render_pcd,
        gt_mesh_sample,
        down_sample_res=0.01,
        truncation_acc=10.0,
        truncation_comp=10.0,
        segment_len=15,
        precision_recall_thres=0.1,
    )
    logs.update(local_map_err)

    # write the log into a file
    with open(write_add, "w") as f:
        json.dump(logs, f, indent=4)
