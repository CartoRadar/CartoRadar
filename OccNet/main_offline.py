import configargparse
import yaml, attridict, shutil
from tqdm import tqdm
from pathlib import Path
import logging, os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import get_model
from model.PoseLearn import LearnPose
from dataset import (
    InfiniteSampler,
    get_dataset,
    get_train_val_functions,
)
from torch.optim.lr_scheduler import _LRScheduler


class WarmupMultiStepPoseLR(_LRScheduler):
    # customized warmup scheduler
    def __init__(self, optimizer, init_iters, warm_up, milestones, gamma=0.1, last_epoch=-1):
        self.init_iters = init_iters
        self.warmup_iters = warm_up
        self.milestones = set(milestones)
        self.gamma = gamma
        super(WarmupMultiStepPoseLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.init_iters:
            lr_scale = 0
            return [base_lr * lr_scale for base_lr in self.base_lrs]

        if self.last_epoch >= self.init_iters and (self.last_epoch - self.init_iters) < self.warmup_iters:
            lr_scale = ((self.last_epoch - self.init_iters) + 1) / self.warmup_iters
            return [base_lr * lr_scale for base_lr in self.base_lrs]

        if self.last_epoch in self.milestones:
            return [group["lr"] * self.gamma for group in self.optimizer.param_groups]
        return [group["lr"] for group in self.optimizer.param_groups]


def parse_arguments():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml", help="data config file path")
    parser.add_argument("--expname", type=str, default="develop", help="experiment name")
    parser.add_argument("--no_log", action="store_false", help="choose to ignore log")
    return parser.parse_args()


def main():
    # load arguments
    args = parse_arguments()
    with open(args.config, "r") as file:
        config = attridict(yaml.safe_load(file))
    require_log = args.no_log

    # create necessary folders and prepare the logger and tensorboard writer
    writer = None
    log_folder = None
    ckpt_folder = None
    pose_folder = None
    if require_log:
        exp_folder = Path('./output/' + args.expname)
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
        writer = SummaryWriter(log_folder)  # tensorboard
        logging.basicConfig(
            filename=log_folder / Path("train.log"),
            filemode="w",
            level=logging.INFO,
            format="[%(levelname)s %(asctime)s] %(message)s",
        )
        logger = logging.getLogger(__name__)  # log file

    # load model and build datasets
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config).to(device)

    train_dataset, val_dataset = get_dataset(config, model)
    infinite_sampler = InfiniteSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=infinite_sampler, num_workers=6, pin_memory=True)

    # load pose model if needed
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

    # create optimizer and lr scheduler
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.train.lr)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=config.train.lr_milestones,
        gamma=config.train.lr_gamma,
    )

    # create pose optimizer and lr scheduler if needed
    pose_optimizer, pose_lr_scheduler = None, None
    if config.model.optimize_pose is True:
        pose_optimizer = torch.optim.Adam(params=pose_model.parameters(), lr=config.train.pose_lr)

        pose_lr_scheduler = WarmupMultiStepPoseLR(
            pose_optimizer,
            config.train.zero_init,
            config.train.warmup,
            config.train.lr_milestones,
        )

    # log the input trajectory and gt trajectory into txt files if needed
    gt_poses = val_dataset.lidar_pose
    if require_log:
        with open(pose_folder / Path(f"gt_trajectory.txt"), "w") as file:
            for pose in gt_poses:
                pose = pose[:3, :]
                text = " ".join(map(str, pose.view(-1).tolist()))
                file.write(text + "\n")

    raw_poses = torch.tensor(val_dataset.pose)
    if require_log:
        with open(pose_folder / Path(f"raw_trajectory.txt"), "w") as file:
            for pose in raw_poses:
                pose = pose[:3, :]
                text = " ".join(map(str, pose.view(-1).tolist()))
                file.write(text + "\n")

    # get training and validation functions
    train_step, online_train_step, train_log, val_step, val_log = get_train_val_functions(config)

    # start the training loop
    for i, batch in enumerate(tqdm(train_loader, total=config.train.total_iters, desc="training")):
        # train the model, the returned logs include loss information, current lr, etc.
        logs = train_step(
            batch,
            model,
            optimizer,
            lr_scheduler,
            train_dataset,
            pose_model,
            pose_optimizer,
            pose_lr_scheduler,
            idx=i + 1,
        )

        i = i + 1
        # log the loss, lr if needed
        if require_log and i % config.train.train_log_per == 0:
            train_log(logs, writer, logger, i)

        # validation
        if i % (config.train.val_per[0] if i < config.train.val_per[1] else config.train.val_per[1]) == 0:
            results = val_step(model, val_dataset, config, pose_model)
            if require_log:
                val_log(results, writer, logger, i)
            torch.cuda.empty_cache()

        # save the model and pose model checkpoint
        if require_log and i % config.train.save_per == 0:
            torch.save(model.state_dict(), ckpt_folder / Path(f"model_step_{i}.pth"))
            if pose_model is not None:
                torch.save(
                    pose_model.state_dict(),
                    ckpt_folder / Path(f"pose_model_step_{i}.pth"),
                )
                with open(pose_folder / Path(f"pred_trajectory_step_{i}.txt"), "w") as file:
                    for pose in pose_model.return_pose():
                        pose = pose[:3, :]
                        text = " ".join(map(str, pose.view(-1).tolist()))
                        file.write(text + "\n")

        # stop the training
        if i == config.train.total_iters:
            print("training finished. Exit..")
            break


if __name__ == "__main__":
    # set random seed
    torch.random.manual_seed(0)
    main()
