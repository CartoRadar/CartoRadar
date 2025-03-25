import yaml, attridict, configargparse
from pathlib import Path
import pickle
import torch, os
from model import get_model
from model.PoseLearn import LearnPose
from dataset import get_dataset
from model.PoseLearn import align_ate_c2b_use_a2b, compute_ATE, compute_rpe
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import natsort, copy, json, glob
from scipy.spatial import cKDTree


def parse_arguments():
    # Parse the arguments
    parser = configargparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True, default=None, help="target experiment for the preparation of result analysis.")
    parser.add_argument("--vis_traj", action="store_true", default=False, help="flag for trajectory visualization.")
    parser.add_argument("--vis_pc", action="store_true", default=False, help="flag for point cloud visualization.")
    parser.add_argument(
        "--save_pkl",
        action="store_true",
        default=False,
        help="store all results into a pickle for the following evaluations",
    )
    parser.add_argument("--iter_num", type=int, default=None, help="assign the iteration of the model for evaluation.")

    return parser.parse_args()


def main():
    # load the arguments
    args = parse_arguments()
    folder = args.target
    vis_traj = args.vis_traj
    vis_pc = args.vis_pc
    save_pkl = args.save_pkl
    iter_num = args.iter_num
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # find the target ckpt file and load the configuration from the target folder
    folder = Path('./output/' + folder)
    model_folder = folder / "ckpt"
    cfg_files = os.path.join(folder, "RadarOccNerf*.yaml")
    cfg_files = glob.glob(cfg_files)
    cfg_file = cfg_files[0]

    if iter_num is None:
        all_files = [f for f in os.listdir(model_folder) if os.path.isfile(os.path.join(model_folder, f))]
        model_files = [f for f in all_files if f.startswith("model_step_")]
        pose_model_files = [f for f in all_files if f.startswith("pose_model_step_")]

        sorted_model_files = natsort.natsorted(model_files)
        sorted_pose_model_files = natsort.natsorted(pose_model_files)

        latest_model_file = sorted_model_files[-1]
        latest_pose_model_file = sorted_pose_model_files[-1]

    else:
        latest_model_file = "model_step_" + str(iter_num) + ".pth"
        latest_pose_model_file = "pose_model_step_" + str(iter_num) + ".pth"

    scene_model_add = model_folder / latest_model_file
    pose_model_add = model_folder / latest_pose_model_file

    print(f"loading pose model {pose_model_add}, map model {scene_model_add}")
    with open(cfg_file, "r") as file:
        config = attridict(yaml.safe_load(file))

    outlier_range = config.dataset.outlier_range
    use_uncertainty = config.dataset.use_uncertainty
    every_n_frame = 1

    # initialize a model according to the config
    model = get_model(config).to(device)
    # initialize a model according to the dataset
    train_dataset, val_dataset = get_dataset(config, model, train=False)

    # load the checkpoint into the model
    model.load_state_dict(torch.load(scene_model_add))
    if config.model.optimize_pose:
        pose_model = LearnPose(
            train_dataset.pose.shape[0],
            True,
            True,
            config,
            torch.tensor(train_dataset.pose).to(device),
        ).to(device)
    else:
        pose_model = LearnPose(
            train_dataset.pose.shape[0],
            False,
            False,
            config,
            torch.tensor(train_dataset.pose).to(device),
        ).to(device)

    pose_model.load_state_dict(torch.load(pose_model_add))

    # set the model to evaluation mode
    pose_model.eval()
    model.eval()

    # prepare the local direction for the rays to generate the point clouds
    ele_idx = 64
    azi_idx = 512

    elevation_angles = np.pi / 4 - np.pi / 2 / (ele_idx - 1) * np.arange(ele_idx, dtype=np.float32)
    azimuth_angles = -2 * np.pi / azi_idx * np.arange(azi_idx, dtype=np.float32)
    elevation_grid, azimuth_grid = np.meshgrid(elevation_angles, azimuth_angles, indexing="ij")
    local_dir = np.vstack(
        (
            (np.cos(azimuth_grid) * np.cos(elevation_grid))[None, ...],
            (np.sin(azimuth_grid) * np.cos(elevation_grid))[None, ...],
            np.sin(elevation_grid)[None, ...],
        )
    )

    # project the depth infomratino back to the world coordinate (unnormalized)
    other_logs = {}
    N_frame, N_elev, N_azi = val_dataset.image_shape
    N_rays = N_elev * N_azi
    scale_factor = val_dataset.scale_factor

    depths = (val_dataset.global_depth_lidar.view(-1, N_elev * N_azi, 1) * scale_factor).cpu().numpy()
    radar_depths = (val_dataset.global_depth.view(-1, N_elev * N_azi, 1) * scale_factor).cpu().numpy()

    render_errs = []
    render_diff_errs = []
    grid_res = val_dataset.render_grid_res
    near, far = val_dataset.scaled_ray_range


    # log the raw pose
    pred_poses = torch.cat([(pose_model.get_globalPose(i))[None, ...] for i in range(pose_model.num_cams)]).detach()
    radar_poses = torch.cat([(pose_model.get_initPose(i))[None, ...] for i in range(pose_model.num_cams)]).detach()
    gt_poses = val_dataset.lidar_pose

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred_poses.cpu().numpy()[:, :3, 3]))
    o3d.io.write_point_cloud(str(folder / "pred_poses.pcd"), pcd)
    print("pose saved")
    np.save(folder / ("pred_pose.npy"), pred_poses.cpu().numpy())


    # align the pose with the ground truth and plot the trajectory for comparison
    pred_aligned_poses, pred_R, pred_t = align_ate_c2b_use_a2b(pred_poses, gt_poses)
    pred_trans = np.concatenate((pred_R, pred_t), axis=-1)[0]
    pred_trans = np.concatenate((pred_trans, [[0, 0, 0, 1]]))

    radar_aligned_poses, radar_R, radar_t = align_ate_c2b_use_a2b(radar_poses, gt_poses)
    radar_trans = np.concatenate((radar_R, radar_t), axis=-1)[0]
    radar_trans = np.concatenate((radar_trans, [[0, 0, 0, 1]]))

    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 2)
    plt.scatter(
        radar_aligned_poses[:, 0, 3].cpu(),
        radar_aligned_poses[:, 1, 3].cpu(),
        label="input_pose_aligned",
    )
    plt.scatter(
        pred_aligned_poses[:, 0, 3].cpu(),
        pred_aligned_poses[:, 1, 3].cpu(),
        label="pred_aligned",
    )
    plt.scatter(gt_poses[:, 0, 3], gt_poses[:, 1, 3], label="gt")
    plt.title("gt_poses vs pred_aligned_poses vs radar_aligned_poses")
    plt.legend()

    plt.subplot(1, 2, 1)
    plt.scatter(radar_poses[:, 0, 3].cpu(), radar_poses[:, 1, 3].cpu(), label="input_pose")
    plt.scatter(pred_poses[:, 0, 3].cpu(), pred_poses[:, 1, 3].cpu(), label="pred")
    plt.scatter(gt_poses[:, 0, 3].cpu(), gt_poses[:, 1, 3].cpu(), label="gt")
    plt.title("gt_poses vs pred_poses vs radar_poses")
    plt.legend()

    plt.tight_layout()
    plt.savefig(str(folder / (latest_model_file[:-4] + "_traj" + ".png")))

    # begin the range rendering
    render_depth = []
    render_depth_mask = []
    test_idx = []
    with torch.no_grad():
        for idx in range(0, N_frame, every_n_frame):
            test_idx.append(idx)
            # prepare the origins of the rays and transform the local directions to the world coordinate.
            pose = pose_model.get_globalPose(idx)
            pose[:3, 3] = (pose[:3, 3] + val_dataset.shift_tensor) / val_dataset.scale_factor
            pos = pose[:3, 3:].T  # [1, 3]
            dir = (pose[:3, :3] @ pose_model.local_dir.T).T

            # sample points on the rays
            pts_num = torch.round(far / grid_res).to(torch.int32)
            rays = torch.cat([torch.full((N_rays, 1), near), torch.full((N_rays, 1), far)], dim=1).to(device)
            z_vals = val_dataset.test_sampler.get_samples(rays, pts_num, False)
            pts = pos.unsqueeze(1) + dir.unsqueeze(1) * z_vals.unsqueeze(2)

            # render the depth result; chumk_num can be adjusted to fit the GPU memory
            depth_render = []
            depth_render_mask = []
            chunk_num = 32
            chunk_size = N_elev * N_azi // chunk_num * pts_num

            for chunk_id in range(chunk_num):
                ret = model.render(
                    pts=pts[int(chunk_id * chunk_size / pts_num) : int((chunk_id + 1) * chunk_size / pts_num)],
                    z_vals=z_vals[int(chunk_id * chunk_size / pts_num) : int((chunk_id + 1) * chunk_size / pts_num)],
                )
                depth_map = ret["depth_map"].clone()
                alpha = ret["alpha_scaled"].clone()
                T_rays_left = torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((alpha.shape[0], 1), device=alpha.device),
                            1.0 - alpha + 1e-10,
                        ],
                        -1,
                    ),
                    -1,
                )[:, :-1][:, -1:]
                depth_map_mask = T_rays_left <= (0.9 if use_uncertainty else 0.1)
                # z_val = ret['z_vals']
                depth_render.append(depth_map)  # (N_rays,)
                depth_render_mask.append(depth_map_mask)
            depth_render = torch.cat(depth_render).reshape(1, N_elev * N_azi, 1)
            depth_render_mask = torch.cat(depth_render_mask).reshape(1, N_elev * N_azi, 1)
            render_depth.append((depth_render * scale_factor).cpu().numpy())
            render_depth_mask.append(depth_render_mask.cpu().numpy())
        render_depth = np.concatenate(render_depth, axis=0)
        render_depth_mask = np.concatenate(render_depth_mask, axis=0)

        pred_poses = pred_poses[test_idx].cpu().numpy()
        radar_poses = radar_poses[test_idx].cpu().numpy()
        gt_poses = gt_poses[test_idx].cpu().numpy()
        radar_aligned_poses = radar_aligned_poses[test_idx].cpu().numpy()
        pred_aligned_poses = pred_aligned_poses[test_idx].cpu().numpy()

    depths = depths[test_idx]
    radar_depths = radar_depths[test_idx]

    # all values we have now are in the world coordinate
    # here the radar_depths are the raw radar inputs, rays with nothing in 9.6m are set to 1000
    # here the render_depth are the raw rendered depths
    # here the render_depth_mask is used as the filter for the rays further than 9.6m
    # here the depths are the raw lidar outputs, non-reflected rays and mirrows are set as -1000

    if vis_traj:
        plt.scatter(gt_poses[:, 0, 3], gt_poses[:, 1, 3], label="gt")
        plt.scatter(pred_aligned_poses[:, 0, 3], pred_aligned_poses[:, 1, 3], label="render")
        plt.legend()
        plt.show()

    # project the range information into point clouds and save them for the following evaluations
    global_pos = pred_poses[:, :3, 3]
    global_pos = np.tile(global_pos[:, None, None, :], (1, 64, 512, 1))

    global_dir = (
        np.matmul(pred_poses[:, :3, :3], local_dir.reshape(3, -1))
        .reshape(pred_poses.shape[0], 3, local_dir.shape[1], local_dir.shape[2])
        .transpose(0, 2, 3, 1)
    )
    render_pts = global_pos.reshape(-1, 32768, 3) + render_depth * global_dir.reshape(-1, 32768, 3)
    np.save(folder / "render_points.npy", render_pts)

    global_pos = gt_poses[:, :3, 3]
    global_pos = np.tile(global_pos[:, None, None, :], (1, 64, 512, 1))

    global_dir = (
        np.matmul(gt_poses[:, :3, :3], local_dir.reshape(3, -1))
        .reshape(gt_poses.shape[0], 3, local_dir.shape[1], local_dir.shape[2])
        .transpose(0, 2, 3, 1)
    )
    gt_pts = global_pos.reshape(-1, 32768, 3) + depths * global_dir.reshape(-1, 32768, 3)
    np.save(folder / "gt_points.npy", gt_pts)


    # calcualte the ray-level range errors
    glass_reflect_gt_mask = depths > (val_dataset.scaled_ray_range[0].item() * scale_factor)
    within_range_gt_mask = depths < (val_dataset.scaled_ray_range[1].item() * scale_factor)
    gt_tree = cKDTree(gt_poses[:, :3, 3])
    gt_pts_distances, _ = gt_tree.query(gt_pts.reshape(-1, 3))
    gt_outlier_mask = (gt_pts_distances <= outlier_range).reshape(-1, 32768, 1)
    gt_depth_mask = glass_reflect_gt_mask & gt_outlier_mask & within_range_gt_mask
    gt_depth_candidate = depths[gt_depth_mask[..., 0] & render_depth_mask[..., 0]]
    render_depth_candidate = render_depth[gt_depth_mask[..., 0] & render_depth_mask[..., 0]]
    render_errs = gt_depth_candidate - render_depth_candidate
    render_diff_errs = copy.deepcopy(render_errs)
    render_errs = np.abs(render_errs)

    render_tree = cKDTree(pred_poses[:, :3, 3])
    render_pts_distances, _ = render_tree.query(render_pts.reshape(-1, 3))
    render_outlier_mask = (render_pts_distances <= outlier_range).reshape(-1, 32768, 1)
    vis_mask = render_outlier_mask & render_depth_mask

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector((render_pts[vis_mask[..., 0]].reshape(-1, 3)))
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(
        (gt_pts[(glass_reflect_gt_mask & gt_outlier_mask)[..., 0]].reshape(-1, 3))
    )

    if vis_pc:
        o3d.visualization.draw_geometries([pcd])

    # save the point cloud into the files
    o3d.io.write_point_cloud(str(folder / (latest_model_file[:-4] + "_map" + ".pcd")), pcd)
    pcd.transform(pred_trans)
    o3d.io.write_point_cloud(str(folder / (latest_model_file[:-4] + "_aligned_map" + ".pcd")), pcd) # the map here is aligned with the gt trajectory

    # save all the intermediate results into the pickle file if needed
    if save_pkl:
        data_dict = {}
        data_dict["input_depths"] = radar_depths
        data_dict["render_depths"] = render_depth
        data_dict["gt_depths"] = depths

        data_dict["input_poses"] = radar_poses
        data_dict["render_poses"] = pred_poses
        data_dict["gt_poses"] = gt_poses

        data_dict["input_aligned_poses"] = radar_aligned_poses
        data_dict["render_aligned_poses"] = pred_aligned_poses
        data_dict["input_align"] = radar_trans
        data_dict["render_align"] = pred_trans

        data_dict["gt_outlier_mask"] = gt_outlier_mask
        data_dict["pred_outlier_mask"] = render_outlier_mask
        data_dict["render_depth_mask"] = render_depth_mask  # No hit points

        data_dict["outlier_range"] = outlier_range

    # calculate the ATE and RPE errors
    ate_rmse, ate_mean = compute_ATE(gt_poses, pred_aligned_poses)
    rpe_trans, rpe_rot = compute_rpe(gt_poses, pred_aligned_poses)

    # log all the above metrics into a dictionary
    other_logs = {}
    other_logs["RPE_TRANS(MEAN: cm)"] = rpe_trans.item() * 100
    other_logs["RPE_ROT(MEAN: deg)"] = np.rad2deg(rpe_rot.item())
    other_logs["ATE(RMSE: cm)"] = ate_rmse.item() * 100
    other_logs["ATE(MEAN: cm)"] = ate_mean.item() * 100

    result = (
        np.std(render_errs),
        np.mean(render_errs),
        np.median(render_errs),
        np.quantile(render_errs, 0.1),
        np.quantile(render_errs, 0.2),
        np.quantile(render_errs, 0.8),
        np.quantile(render_errs, 0.9),
    )
    diff_result = (
        np.std(render_diff_errs),
        np.mean(render_diff_errs),
        np.median(render_diff_errs),
        np.quantile(render_diff_errs, 0.1),
        np.quantile(render_diff_errs, 0.2),
        np.quantile(render_diff_errs, 0.8),
        np.quantile(render_diff_errs, 0.9),
    )

    abs_error = {}
    diff_error = {}

    abs_error["mean"] = result[1].item()
    abs_error["std"] = result[0].item()
    abs_error["err_0.1"] = result[3].item()
    abs_error["err_0.2"] = result[4].item()
    abs_error["err_0.5"] = result[2].item()
    abs_error["err_0.8"] = result[5].item()
    abs_error["err_0.9"] = result[6].item()

    diff_error["mean"] = diff_result[1].item()
    diff_error["std"] = diff_result[0].item()
    diff_error["err_0.1"] = diff_result[3].item()
    diff_error["err_0.2"] = diff_result[4].item()
    diff_error["err_0.5"] = diff_result[2].item()
    diff_error["err_0.8"] = diff_result[5].item()
    diff_error["err_0.9"] = diff_result[6].item()

    other_logs["L1_abs_error"] = abs_error
    other_logs["L1_diff_error"] = diff_error

    # write into a json file
    with open(folder / (latest_model_file[:-4] + ".json"), "w") as f:
        json.dump(other_logs, f, indent=4)

    # write into a pickle file
    if save_pkl:
        data_dict["L1_err"] = other_logs
        with open(folder / (latest_model_file[:-4] + "_result" + ".pkl"), "wb") as f:
            pickle.dump(data_dict, f)


if __name__ == "__main__":
    # This file is used to render the scene and save intermediate results for the latter complete evaluations.
    # Some evaluations are conducted for early estimation of the performance of the model.
    # Visualizations are also saved for the qualitative analysis.
    # You must assign the target experiment folder to the argument --target.
    # You must set the flag --save_pkl to True if you want to run mobicom_analyze_pkl.py for full evaluations.
    # You can choose the saved model by setting the argument --iter_num. By default, the latest model is selected.
    # If you want to visualize the trajectory in the middle, set the flag --vis_traj to True.
    # If you want to visualize the point cloud in the middle, set the flag --vis_pc to True.
    main()
