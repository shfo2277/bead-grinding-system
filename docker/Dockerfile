# =============================================================================
# BEADtrain Docker Image: bead-humble-torch
# 비드 그라인딩 자동화 시스템 — ROS2 Humble + PyTorch + CUDA
# =============================================================================
# 빌드: docker build -t bead-humble-torch ./docker/
# =============================================================================

FROM ros:humble-desktop

# ---------- 시스템 패키지 ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    git \
    vim \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------- ROS2 추가 패키지 ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-image-transport-plugins \
    ros-humble-tf2-ros \
    ros-humble-tf-transformations \
    ros-humble-cyclonedds \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-librealsense2 \
    ros-humble-realsense2-camera \
    ros-humble-realsense2-camera-msgs \
    ros-humble-control-msgs \
    && rm -rf /var/lib/apt/lists/*

# ---------- pip 업그레이드 ----------
RUN python3 -m pip install --upgrade pip

# ---------- sympy (torch 의존성 충돌 방지) ----------
RUN pip install "sympy>=1.13.3" --ignore-installed

# ---------- PyTorch + CUDA ----------
# ※ CUDA 버전에 맞게 cu128 부분을 변경할 것 (cu118, cu121, cu124 등)
# ※ --pre는 nightly 버전. 안정 버전 원하면 --pre 제거하고 URL에서 nightly→test 또는 제거
RUN pip install --pre \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# ---------- Python 패키지 ----------
RUN pip install \
    opencv-python-headless \
    numpy \
    pillow \
    matplotlib \
    tqdm \
    segmentation-models-pytorch \
    albumentations \
    einops

# ---------- 작업 디렉토리 ----------
RUN mkdir -p /workspace/BEADtrain
WORKDIR /workspace/BEADtrain

CMD ["/bin/bash"]
