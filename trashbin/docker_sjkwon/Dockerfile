FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

ARG USERNAME=sjkwon
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    locales curl gnupg2 lsb-release build-essential \
    python3-pip sudo git wget vim && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8

# Miniconda 설치
RUN curl -sSLo miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    chmod +x miniforge.sh && ./miniforge.sh -b -p $CONDA_DIR && \
    rm miniforge.sh && conda clean -afy

RUN conda install -y python=3.10
# COPY requirements.txt /tmp/requirements.txt
# RUN pip install -r /tmp/requirements.txt

# PyTorch 및 기본 라이브러리 설치 (CUDA 12.1에 맞게)
# RUN conda install -y python=3.10 pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 numpy==1.26.4 scipy pandas ipython jupyter matplotlib -c pytorch -c nvidia

# RUN conda install -y python=3.10 && \
#     conda install -y pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia && \
#     conda install -y numpy==1.26.4 scipy pandas ipython jupyter matplotlib

# RUN conda install -y python=3.10 && \
#     conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia && \
#     conda install -y numpy scipy pandas ipython jupyter matplotlib -c conda-forge

# sjkwon 유저 생성
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# bash 설정 (conda only)
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/${USERNAME}/.bashrc && \
    echo "conda activate base" >> /home/${USERNAME}/.bashrc

USER ${USERNAME}
WORKDIR /home/sjkwon/workspace
