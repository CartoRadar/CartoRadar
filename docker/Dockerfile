FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set default build arguments (override during build)
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000
ARG CUDA_ARCH=86

# Create a user with the passed UID and GID
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID --create-home --shell /bin/bash $USERNAME

# Set timezone
ENV TZ=America/New_York

# Install some system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    python3.10 \
    python3-pip \
    vim \
    tmux \
    htop \
    rsync \
    libsuitesparse-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Install python dependencies
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 \
    attridict==0.0.8 \
    fvcore==0.1.5.post20221221 \
    pyiqa==0.1.6.4 \
    matplotlib==3.6.0 \
    numpy==1.23.3 \
    open3d==0.16.0 \
    scipy==1.9.1 \
    torch==2.0.0 \
    torchaudio==2.0.1 \
    torchvision==0.15.1 \
    tqdm==4.64.1 \
    pycocotools==2.0.6 \
    cloudpickle==2.2.1 \
    omegaconf==2.3.0 \
    opencv-python==4.6.0.66 \
    tensorboard==2.12.2 \
    natsort==8.4.0

# Install detectron2
RUN git clone https://github.com/facebookresearch/detectron2.git \
    && cp -r detectron2/detectron2 /usr/lib/python3/dist-packages \
    && rm -r detectron2

# Install tinycudann
ENV TCNN_CUDA_ARCHITECTURES=$CUDA_ARCH
RUN pip install --no-cache-dir git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install Eigen3.3.4 (need for g2opy)
RUN curl -O https://gitlab.com/libeigen/eigen/-/archive/3.3.4/eigen-3.3.4.tar \
    && tar -xf eigen-3.3.4.tar \
    && mkdir eigen-3.3.4/build \
    && cd eigen-3.3.4/build \
    && cmake .. \
    && make install

# Install g2opy
RUN git clone https://github.com/uoip/g2opy.git \
    && mkdir g2opy/build \
    && cd g2opy/build \
    && cmake .. -DBUILD_WITH_MARCH_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release \
    && make -j8 \
    && cd .. \
    && python setup.py install

# Cleaning
RUN cd .. && rm -rf g2opy eigen-3.3.4 eigen-3.3.4.tar

# Switch to the new user
USER $USERNAME
WORKDIR /home/$USERNAME
