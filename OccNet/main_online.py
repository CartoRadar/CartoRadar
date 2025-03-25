import configargparse
import yaml, attridict, shutil
from pathlib import Path
import logging, os
import torch
from torch.utils.tensorboard import SummaryWriter
from model import get_model
from model.PoseLearn import LearnPose
from dataset import (
    get_dataset,
    get_train_val_functions,
)
import time
import torch.multiprocessing as mp
from dataset.online_dataset import OnlineSlamDataset
from model.PoseLearn import construct_odom_graph
import open3d as o3d
import numpy as np
import copy
from scipy.spatial.transform import Rotation


def parse_arguments():
    parser = configargparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="data config file path"
    )
    parser.add_argument(
        "--expname", type=str, default="develop", help="experiment name"
    )
    parser.add_argument("--no_log", action="store_false", help="choose to ignore log")
    return parser.parse_args()


def frame_system(
    queue,
    train_dataset
):
    """
    This function simulates the frame system in the online SLAM system.
    Especially, it simulates the time delay between the production of frames, which is 0.55s.
    """
    
    for i in range(
        train_dataset.online_depth.shape[0]
    ):
        cur_depth = train_dataset.online_depth[i : i + 1]
        cur_pose = train_dataset.online_pose[i : i + 1]
        cur_lidar_depth = train_dataset.online_lidar[i : i + 1]
        cur_conf = train_dataset.online_conf[i : i + 1]
        item = [cur_depth, cur_pose, cur_lidar_depth, cur_conf, i]
        queue.put(item)
        time.sleep(0.55)  # Simulating time delay between productions. 0.55
    queue.put(None)  # Signal that all frames have been provided.
    print('exiting frame system')


def loop_closure_check_process(
    lock, raw_for_loop, rcv_from_loop, pose_model, online_dataset, scale_factor
):
    """
    This function is used for the loop detection process in the online SLAM system.
    Loop closure would be triggered when a new frame is added to the system.
    """
    history_constraints = []
    while True:
        if not raw_for_loop.empty():
            pose_idx = raw_for_loop.get()
            if pose_idx is None:
                print('exiting loop closure process')
                break
            else:
                # print(f"Process LoopDetection: handling frame{pose_idx}")
                if pose_idx >= 1:
                    raw_poses = (
                        pose_model.get_frames(torch.arange(pose_idx))
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    consec_shift = (
                        (
                            torch.linalg.inv(pose_model.original_init_c2w[pose_idx - 1])
                            @ (pose_model.original_init_c2w[pose_idx])
                        )
                        .cpu()
                        .numpy()
                    )
                    raw_poses = np.concatenate(
                        (raw_poses, (raw_poses[-1] @ consec_shift)[None, ...])
                    )
                else:
                    raw_poses = (
                        pose_model.get_frames(torch.arange(1)).detach().cpu().numpy()
                    )

                flag, constraint, new_poses = loop_closure_check(
                    online_dataset,
                    raw_poses,
                    scale_factor,
                    history_constraints,
                )
                if not flag:
                    N_poses = raw_poses.shape[0]
                    assert N_poses == (pose_idx + 1)
                    with lock:
                        pose_model.init_c2w.data[:N_poses] = torch.tensor(
                            raw_poses
                        ).float()
                        pose_model.r.data.zero_()
                        pose_model.t.data.zero_()
                else:
                    history_constraints.append(constraint)
                    N_poses = new_poses.shape[0]
                    assert N_poses == (pose_idx + 1)

                    # update the pose
                    with lock:
                        pose_model.init_c2w.data[:N_poses] = torch.tensor(
                            new_poses
                        ).float()
                        pose_model.r.data.zero_()
                        pose_model.t.data.zero_()
                        # print(f"Process LoopDetection: Overwriting the pose.")

                    rcv_from_loop.put(N_poses)


def loop_closure_check(online_dataset, raw_poses, scale_factor, history_constraints):
    # This function is doing loop closure detection.
    # hyperparameters for the loop detection
    N_POSE_GAP = 80
    POSE_DIST_THRES = 5.0
    ICP_MEAN_ERR_THRES = 0.18  # 0.18
    ICP_FITNESS_THRES = 0.95  # 0.95
    range_max = 9.6

    # check if the number of poses is enough
    assert raw_poses.shape[0] <= online_dataset.number_frames
    frames_number = raw_poses.shape[0]
    if frames_number <= N_POSE_GAP:
        return False, None, None

    poses0 = copy.deepcopy(raw_poses)

    N_poses = poses0.shape[0]
    i = N_poses - 1
    if N_poses <= N_POSE_GAP:
        return False, None, None

    # find closest pose that is within the threshold
    dists = np.linalg.norm(
        poses0[i, :2, 3] - poses0[: (i - N_POSE_GAP + 1), :2, 3], ord=2, axis=1
    )
    min_ind = np.argmin(dists)
    if dists[min_ind] >= POSE_DIST_THRES:
        return False, None, None

    # found the loop candidate
    # print(f'find close inds: {i} -> {min_ind}')

    # construct the o3d PointCloud of the input point cloud and the matched point cloud for registration
    src_pose = poses0[min_ind]
    dist_pose = poses0[i]

    local_dirs = ((online_dataset.local_dir.cpu().numpy().reshape(3, -1))).T

    src_rays, _ = online_dataset.fetch_frame_rays(min_ind)
    src_rays = src_rays.cpu().numpy() * scale_factor
    src_mask = (src_rays < range_max) & (src_rays > 0)
    src_mask = src_mask[:, 0]
    src_pc = (src_rays[..., None] * local_dirs[:, None, :]).reshape(-1, 3)[src_mask]
    src_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_pc))
    src_pcd.estimate_normals()

    dst_rays, _ = online_dataset.fetch_frame_rays(i)
    dst_rays = dst_rays.cpu().numpy() * scale_factor
    dst_mask = (dst_rays < range_max) & (dst_rays > 0)
    dst_mask = dst_mask[:, 0]
    dst_pc = (dst_rays[..., None] * local_dirs[:, None, :]).reshape(-1, 3)[dst_mask]
    dst_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst_pc))
    dst_pcd.estimate_normals()

    # do ICP for twice to find transform and distance
    T = np.linalg.inv(src_pose) @ dist_pose
    reg_p2p = o3d.pipelines.registration.registration_icp(
        dst_pcd,
        src_pcd,
        5.0,  # max_dist
        T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-10, relative_rmse=1e-10, max_iteration=100
        ),
    )

    reg_p2p = o3d.pipelines.registration.registration_icp(
        dst_pcd,
        src_pcd,
        1.0,  # max_dist
        reg_p2p.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-10, relative_rmse=1e-10, max_iteration=100
        ),
    )

    # T = reg_p2p.transformation
    T = np.copy(reg_p2p.transformation)
    distances = reg_p2p.inlier_rmse

    if np.mean(distances) > ICP_MEAN_ERR_THRES or reg_p2p.fitness < ICP_FITNESS_THRES:
        return False, None, None

    # found loop! Add to graph, optimize, and update poses
    # print("loop detected. Add to graph, optimizing\n")

    # construct the graph and optimize
    graph = construct_odom_graph(poses0)
    for constraint in history_constraints:
        graph.add_edge([constraint[0], constraint[1]], constraint[2])
    graph.add_edge([min_ind.item(), i], T)
    graph.optimize()
    poses0 = np.stack([graph.get_pose(j) for j in range(N_poses)])

    return True, (min_ind.item(), i, T), poses0


def mapping_localizing(
    queue,
    lock,
    raw_for_loop,
    rcv_from_loop,
    online_dataset,
    online_train_step,
    model,
    optimizer,
    lr_scheduler,
    train_dataset,
    pose_model,
    pose_optimizer,
    pose_lr_scheduler,
    train_log=None,
    log_folder=None,
    log_per=10,
    val_step=None,
    val_dataset=None,
    config=None,
    val_log=None,
    ckpt_folder=None,
    pose_folder=None,
    sample_number=4096,
):
    """
    This function is the mapping process of the online SLAM system.
    It samples the frames from the queue and trains the model with the frames.
    """
    # count is the iteracion counter
    count = 0
    # create the tensorbaord writer if needed
    if log_folder is not None:
        writer = SummaryWriter(log_folder)  # tensorboard
    else:
        writer = None

    # start the model training
    while True:
        # rcv_from_loop is used to receive the information about whcih frame's pose has beed overwritten.
        # It can be removed if you don't want to know. We keep it here for others to better understand the code and also for debugging.
        if not rcv_from_loop.empty():
            frame_number = rcv_from_loop.get()
            # print(
            #     f"Process Mapping: PoseModel has been overwritten by pose graph for frame {frame_number}."
            # )

        # If there are items in the queue,
        if not queue.empty():
            # get the item from the passed queue
            item = queue.get()
            # Other process would put None in the queue to signal that all frames have been provided.
            # Thus, if the received item is None, all frames have been provided by 'frame_system' and we continue the mapping process
            if item is None:
                # print("Process Mapping: No more frames to add. Continue the training.")
                raw_for_loop.put(None)
                # Generate the sample indice for the online dataset
                online_dataset.generate_sample_indice(sample_number)
                start_time = time.time()
                tmp_count = 0
                while True:
                    if not rcv_from_loop.empty():
                        frame_number = rcv_from_loop.get()
                        # print(
                        #     f"Process Mapping: PoseModel has been overwritten by pose graph for frame {frame_number}."
                        # )

                    # sample rays from the online dataset
                    rays_sampled = online_dataset.sample_all_data(sample_number)
                    # Use lock to avoid the conflict when changing the value in the online dataset like the poses.
                    with lock:
                        logs = online_train_step(
                            rays_sampled,
                            model,
                            optimizer,
                            lr_scheduler,  # lr_scheduler
                            train_dataset,
                            pose_model,
                            pose_optimizer,
                            pose_lr_scheduler,  # pose_lr_scheduler
                            idx=count + 1,
                            optimize_pose = True
                        )
                    # Log the training information
                    if writer is not None and count % 10 == 0:
                        train_log(logs, writer, None, count)
                    count += 1

                    # You can manually change the False to True if you want to save the intermediate model. Here we set it False to reduce the time overhead.
                    if count % 2000 == 0 and False:
                        if ckpt_folder and pose_folder:
                            # print(ckpt_folder,  Path(f'model_step_{count}.pth'))
                            torch.save(
                                model.state_dict(),
                                ckpt_folder / Path(f"model_step_{count}.pth"),
                            )
                            if pose_model is not None:
                                torch.save(
                                    pose_model.state_dict(),
                                    ckpt_folder / Path(f"pose_model_step_{count}.pth"),
                                )
                                with open(
                                    pose_folder
                                    / Path(f"pred_trajectory_step_{count}.txt"),
                                    "w",
                                ) as file:
                                    for pose in pose_model.return_pose():
                                        pose = pose[:3, :]
                                        text = " ".join(
                                            map(str, pose.view(-1).tolist())
                                        )
                                        file.write(text + "\n")
                    # stop the online training
                    # if time.time() - start_time > 120:
                    if tmp_count == 20000:
                        if ckpt_folder and pose_folder:
                            torch.save(
                                model.state_dict(),
                                ckpt_folder / Path(f"model_step_{count}.pth"),
                            )
                            if pose_model is not None:
                                torch.save(
                                    pose_model.state_dict(),
                                    ckpt_folder / Path(f"pose_model_step_{count}.pth"),
                                )
                                with open(
                                    pose_folder
                                    / Path(f"pred_trajectory_step_{count}.txt"),
                                    "w",
                                ) as file:
                                    for pose in pose_model.return_pose():
                                        pose = pose[:3, :]
                                        text = " ".join(
                                            map(str, pose.view(-1).tolist())
                                        )
                                        file.write(text + "\n")
                        break

                    tmp_count += 1

                print('exiting mapping process')
                break

            
            # If it is not None, it means we have new observed frames.
            # we add them to the online dataset.
            online_dataset.add_data(item[:-1])
            raw_for_loop.put(item[-1])

        # If the online dataset is empty, we wait for the first frame to be added.
        if online_dataset.number_frames == 0:
            # print("waiting for the init of online_dataset.")
            continue

        # sample rays from the online dataset
        rays_sampled = online_dataset.sample_data(sample_number)
        # Use lock to avoid the conflict when changing the value.
        with lock:
            # print("Process Mapping: doing the mapping")
            logs = online_train_step(
                rays_sampled,
                model,
                optimizer,
                lr_scheduler,
                train_dataset,
                pose_model,
                pose_optimizer,
                pose_lr_scheduler,
                idx=count + 1,
            )
        # Log the training information
        if writer is not None and count % 10 == 0:
            train_log(logs, writer, None, count)
        count += 1

def main():
    # parse the arguments
    args = parse_arguments()
    with open(args.config, "r") as file:
        config = attridict(yaml.safe_load(file))

    require_log = args.no_log
    log_folder = None
    ckpt_folder = None
    pose_folder = None

    # create the output folders 
    if require_log:
        # make output experiment folders
        exp_folder = Path("./output") / Path(args.expname)
        exp_folder.mkdir(parents=True, exist_ok=True)

        p = Path(config.dataset.pose_file)
        target = p.resolve().parent.parent / "maps" / "gt_map_clean.pcd"
        link_name = exp_folder / ("gt_map.pcd")
        os.symlink(target, link_name)
        
        ckpt_folder = exp_folder / Path("ckpt")
        ckpt_folder.mkdir(parents=True, exist_ok=True)
        vis_folder = exp_folder / Path("vis")
        vis_folder.mkdir(parents=True, exist_ok=True)
        log_folder = exp_folder / Path("log")
        log_folder.mkdir(parents=True, exist_ok=True)
        pose_folder = exp_folder / Path("pose")
        pose_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, exp_folder)  # copy config file

        # prepare logging
        logging.basicConfig(
            filename=log_folder / Path("train.log"),
            filemode="w",
            level=logging.INFO,
            format="[%(levelname)s %(asctime)s] %(message)s",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build datasets and models
    model = get_model(config).to(device)
    train_dataset, val_dataset = get_dataset(config, model)

    if config.model.optimize_pose:
        pose_model = LearnPose(
            train_dataset.online_pose.shape[0],
            True,
            True,
            config,
            train_dataset.online_pose.to(device),
        ).to(device)
    else:
        pose_model = LearnPose(
            train_dataset.online_pose.shape[0],
            False,
            False,
            config,
            train_dataset.online_pose.to(device),
        ).to(device)

    pose_model.provide_gt(train_dataset.lidar_pose)

    # create the optimizer and lr_scheduler
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.train.lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=config.train.lr_milestones,
        gamma=config.train.lr_gamma,
    )

    pose_optimizer, pose_lr_scheduler = None, None
    if config.model.optimize_pose is True:
        pose_optimizer = torch.optim.Adam(
            params=pose_model.parameters(), lr=config.train.pose_lr
        )

        pose_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=pose_optimizer,
            milestones=config.train.lr_milestones,
            gamma=config.train.lr_gamma,
        )

    # save the ground truth poses
    gt_poses = val_dataset.lidar_pose
    if require_log:
        with open(pose_folder / Path(f"gt_trajectory.txt"), "w") as file:
            for pose in gt_poses:
                pose = pose[:3, :]
                text = " ".join(map(str, pose.view(-1).tolist()))
                file.write(text + "\n")

    # get the training and validation functions
    train_step, online_train_step, train_log, val_step, val_log = (
        get_train_val_functions(config)
    )

    # online_dataset refers to the shared data queue in our paper. It is initialized with empty frames.
    online_dataset = OnlineSlamDataset(train_dataset.image_shape[0])

    # begin the multi-process setting
    # queue is used to pass the frames from the frame system to the mapping process
    # lock is used to avoid the conflict when changing the value in the online dataset like the poses.
    # raw_for_loop is used to pass the frame index to the loop detection process
    # rcv_from_loop is used to receive the information about which frame's pose has been overwritten.
    
    mp.set_start_method("spawn")
    lock = mp.Lock()
    queue = mp.Queue()
    raw_for_loop = mp.Queue()
    rcv_from_loop = mp.Queue()

    # create the processes
    m = mp.Process(
        target=mapping_localizing,
        args=(
            queue,
            lock,
            raw_for_loop,
            rcv_from_loop,
            online_dataset,
            online_train_step,
            model,
            optimizer,
            lr_scheduler,
            train_dataset,
            pose_model,
            pose_optimizer,
            pose_lr_scheduler,
            train_log,
            log_folder,
            config.train.train_log_per,
            val_step,
            val_dataset,
            config,
            val_log,
            ckpt_folder,
            pose_folder,
        ),
    )

    f = mp.Process(
        target=frame_system,
        args=(
            queue,
            train_dataset,
        ),
    )

    l = mp.Process(
        target=loop_closure_check_process,
        args=(
            lock,
            raw_for_loop,
            rcv_from_loop,
            pose_model,
            online_dataset,
            train_dataset.scale_factor,
        ),
    )

    # start the processes
    l.start()
    m.start()
    f.start()
    print("processes all started")

    # join the processes
    l.join()
    m.join()
    f.join()

    print("Online SLAM has completed.")


if __name__ == "__main__":
    # This is a multi-process python script for the online SLAM system.
    # It would be much more complicated than main_offline.py
    # In this file, we intentionally comment out some print statements to reduce the unnecessary time overhead in an online system.
    # If you want to see the detailed information, you can uncomment them, especially if you want to debug this multi-process system.

    # set random seed
    torch.random.manual_seed(0)
    main()
