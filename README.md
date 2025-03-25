<p align="center">
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/implemented%20in-PyTorch-1f929f" alt="pytorch" height="20"></a>
  <a href="https://www.python.org/downloads/release/"><img src="https://img.shields.io/badge/python-&ge;3.9-blue.svg" alt="Python" height="20"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="black" height="20"></a>
  <a href=""><img src="https://img.shields.io/badge/paper-open%20access-1fa25f.svg" alt="paper" height="20"></a>
</p>

# CartoRadar [MobiCom'25]

This is an official repo of CartoRadar for the following paper:  

> **[RF-Based 3D SLAM Rivaling Vision Approaches]()**  
> Anonymous Authors  
> *ACM International Conference on Mobile Computing and Networking (**MobiCom**), 2025* 
>

[[`Paper`]()] [[`Website`]()] [[`Demo Video`]()] [[`Dataset`]()] [[`BibTeX`](#CitingCartoRadar)]

---

## Installation

### Requirements
- GPU: Nvidia RTX 3090 to achieve a similar online SLAM performance (suggested), or GPUs whose memory >= 16 GB, driver version >= 520.61.05, CUDA version >= 11.8
- RAM: Memory >= 32 GB
- Disk: Available space >= 550 GB
- OS: Ubuntu >= 22.04 with Python ≥ 3.9

### Preparation
- Clone the codebase to your local machine
- Change the working directory to the root folder of the repo.

### Environment
We provide two ways to get the environment to run the code: `Conda` and `Docker`. For people who have already had `Docker` on their local machine, we suggest using `Docker`. If you have a problem with one method, you can switch to the other one.

<details>

<summary>Using Docker Image</summary>

Please follow the instructions below to build the image. If you haven't added your user into the docker group (see [here](https://docs.docker.com/engine/install/linux-postinstall)), you need `sudo` access to run the following **docker-related** commands.
```bash
# Detect GPU compute capability
export CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk '{print $1 * 10}')

# build the image (might need sudo)
docker build --build-arg USERNAME=$(whoami) --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) --build-arg CUDA_ARCH=${CUDA_ARCH} -t cartoradar -f docker/Dockerfile .  # don't forget the ending dot

# Start a new container (might need sudo)
docker run -it --rm --gpus all -v .:/home/$(whoami)/CartoRadar -w ~/CartoRadar --shm-size=4096M cartoradar /bin/bash
```

After you start a new container, you can run our code in it. The environment setup is done.

**Note**: Make sure you have **Nvidia driver version >= 520.61.05 and CUDA version >= 11.8** in your local machine, otherwise you won't be able to start the container.

**Note (For advanced users)**: The machine that uses the docker image needs to have the same GPU compute capability as the machine that builds the image. If you want to build the image locally and use it in another machine (e.g., a remote server), you need to specify the compute capability of the target machine by running `export CUDA_ARCH=<GPU compute capability of the target machine>`, then re-build the image.


</details>

<details>

<summary>Using Conda Environment</summary>

Please follow the below instructions to create the conda environment. If you have any problem, feel free to communicate with us as suggested by the conference or try `Docker`.
```bash
# Create a new conda environment
conda env create -f environment.yml
conda activate cartoradar
pip install setuptools==58.2.0 cmake==3.25.0
sudo apt-get install libsuitesparse-dev

# Install g2opy
git clone https://github.com/uoip/g2opy.git \
    && mkdir g2opy/build \
    && cd g2opy/build \
    && cmake .. \
    && make -j8 \
    && cd .. \
    && python setup.py install \
    && cd ..

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install tinycudann
export LDFLAGS="$LDFLAGS -L$CONDA_PREFIX/lib/stubs"
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install other packages
pip install -r requirements.txt
```


</details>


## Dataset
Our dataset includes two parts: the uncertainty quantification dataset and the SLAM evaluation dataset. Our models are also using the [PanoRadar dataset](https://upenn.box.com/v/panoradar-dataset) for pretrain weights.

- **Uncertainty quantification dataset**: This dataset is organized corresponding to different buildings. Within each building, it has three folders, i.e., glass_npy, lidar_npy, and rf_npy. There are synchronized glass masks, lidar range images, and RF heatmaps in those folders. The dataset can be downloaded [here](https://xxxxxx). Please download the data, unzip it, and put it in the folder `cartoradar/Uncertainty/data/uncertainty/`.

- **Pretain weights from PanoRadar dataset**: The [PanoRadar dataset](https://upenn.box.com/v/panoradar-dataset) is a large-scale RF imaging dataset collected from diverse environments. We use the depth-only pretrain weights for our uncertainty quantification models. They can be downloaded [here](https://xxxxxx). Please download the data, unzip it, and put it in the folder `cartoradar/Uncertainty/data/pretrain/`.

- **SLAM evaluation dataset**:
This dataset is organized as different robot moving trajectories. Within each trajectory folder, there are ground truth range images from LiDAR (images-gt), our predicted range images from RF (images-pred), the predicted uncertainty from OursH-16 method (uncertainty-mixed), the robot poses for offline SLAM (poses-npy), and the ground truth point cloud map of the environment (maps). The dataset can be downloaded [here](https://xxxxxx). Please download the data, unzip it, and put it in the folder `cartoradar/OccNet/data/`.

Please make sure you put all the data in the right path. It should have a file structure like this:
```
cartoradar
├── Uncertainty
│   ├── data
│   │   ├── uncertainty
│   │   │   ├── building1
│   │   │   ├── building2
│   │   │   ├── ...
│   │   │   └── building5
│   │   └── pretrain
│   │       ├── depth_building1_lobo_x2
│   │       ├── ...
│   │       └── laplace_building5_lobo_x2
│   ├── ...
│   
├── OccNet
│   ├── data
│   │   ├── building1-exp000
│   │   ├── building1-exp001
|   |   ├──  ...
│   │   ├── building5-exp000
│   │   └── building5-exp001
│   ├── ...
├── ...
```

## Uncertainty Quantification
Our perturbation and sampling based uncertainty quantification methods are applied to trained ML models to estimate the uncertainty of the predictions. We have proposed two types of approaches, denoted as `Ours-#` and `OursH-#` in the paper.

 To start with, change the working directory to `Uncertainty` and follow the instructions below.

### Model Training
To train a model on a specific building, run the following commands. Make sure to change the working directory to `Uncertainty`.
```bash
python train_net.py --config-file configs/<config file>
```

Since we use the leave-one-building-out training strategy to ensure generalization, the above command only trains a model for **one** building. Please substitute `<config file>` to the following files. There are **10** models that need to be trained in total.
|  Ours-# config files | OursH-# config files |
| -------- | ------- |
| pure_depth_building1.yaml  | laplace_net_building1.yaml |
| pure_depth_building2.yaml  | laplace_net_building2.yaml |
| pure_depth_building3.yaml    | laplace_net_building3.yaml |
| pure_depth_building4.yaml    | laplace_net_building4.yaml |
| pure_depth_building5.yaml    | laplace_net_building5.yaml|


### Evaluate Uncertainty Performance
After finishing training the above **10** models, we can evaluate the performance of our uncertainty quantification approaches.

- To evaluate `Ours-#`, run the following command:
  ```bash
  python scripts/eval_sampling_uncertainty.py
  ```
  After the script finishes, you can see the evaluation results printed in the terminal.

- To evaluate `OursH-#`, run the following command:
  ```bash
  python scripts/eval_laplacian_uncertainty.py
  ```
  After the script finishes, you can see the evaluation results printed in the terminal.


## SLAM
Our SLAM system has two versions: offline and online. The following section provides detailed instructions on how to run our offline and online uncertainty-aware RF-based SLAM.

### Evaluate All Trajectories
To evaluate the performance of our SLAM system, first ensure you are in the folder `OccNet`, then you can evaluate our offline performance by simply running the following command in the terminal:
```
bash offline_run_all.sh
```
The above shell script would run our algorithm on all trajectories and output the average performance. You can find it at the end of the terminal output.  Checkpoints can be found in the folder `OccNet/output/`

Similarly, for the online performance, simply run:
```
bash online_run_all.sh
```

**Note**: While running online SLAM, please **avoid** performing other jobs at the same time. Otherwise, they may compete for resources and affect the online SLAM performance.

If you miss the result summary output in the terminal, you can regenerate them simply by running the following code:

```
# For offline result summary
python result_summary.py

# For online result summary
python result_summary.py --online
```

### (Optional) Evaluate a Single Trajectory
If you want to perform offline/online SLAM for one trajectory, first train the network as the following:
```
# For offline slam
python ./main_offline.py --config $config --expname $expname

# For online slam
python ./main_online.py --config $config --expname $expname
```
`$config` represents the path to the config like `./config/building2-exp001/RadarOccNerf_xxx.yaml`, and `$expname` is the name of the experiment.

To evaluate the performance of the SLAM result:
```
# Render and save the scene
python ./mobicom_generate_pkl.py --target "$expname" --save_pkl

# Do the comprehensive evaluation
python ./mobicom_analyze_pkl.py --target "$expname"
```
The metric is written into a JSON file located in the output folder. You can also find other visualizations including the rendered point cloud, the path comparison, etc.

We also provide shell scripts to simplify the above process:
```
# For offline slam
bash offline_run.sh $config $expname

# For online slam
bash online_run.sh $config $expname
```

## License
CartoRadar is licensed under a [CC BY 4.0 License](LICENSE).

## <a name="CitingCartoRadar"></a>Citing CartoRadar
If you use CartoRadar in your research, find the code useful, or would like to acknowledge our work, please consider citing our paper:

```BibTeX
@inproceedings{CartoRadar,
  author    = {Anonymous Authors},
  title     = {RF-Based 3D SLAM Rivaling Vision Approaches},
  booktitle = {ACM International Conference on Mobile Computing and Networking (MobiCom)},
  year      = {2025},
  doi       = {Anonymous DOI},
}
```
