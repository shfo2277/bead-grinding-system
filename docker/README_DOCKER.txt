================================================================================
  bead_gpu 컨테이너 구축 가이드
  처음부터 끝까지 순서대로
================================================================================


================================================================================
  0. 호스트 PC 사전 준비
================================================================================

  아래 3개가 설치되어 있어야 한다.

  (1) NVIDIA 드라이버
      확인: nvidia-smi 명령어가 동작하면 OK
      안 되면: sudo apt install nvidia-driver-570  (버전은 GPU에 맞게)

  (2) Docker Engine
      확인: docker --version
      설치: https://docs.docker.com/engine/install/ubuntu/

  (3) NVIDIA Container Toolkit (nvidia-docker2)
      확인: docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
      설치:
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
          sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
          sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
          sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker


================================================================================
  1. 이미지 빌드 (방법 A — Dockerfile로 새로 빌드)
================================================================================

  BEADtrain 폴더가 있는 위치에서:

    cd /home/ho/BEADtrain
    docker build -t bead-humble-torch ./docker/

  시간: 인터넷 속도에 따라 10~30분 소요 (PyTorch 다운로드가 큼)
  결과: bead-humble-torch 이미지가 로컬에 생성됨

  ※ CUDA 버전 확인:
     nvidia-smi 오른쪽 상단에 "CUDA Version: 12.x" 확인
     Dockerfile 안의 cu128 부분을 맞게 수정:
       CUDA 11.8 → cu118
       CUDA 12.1 → cu121
       CUDA 12.4 → cu124
       CUDA 12.8 → cu128


================================================================================
  2. 이미지 저장/불러오기 (방법 B — 빌드 없이 파일로 복사)
================================================================================

  이미 빌드된 이미지를 .tar 파일로 저장해서 다른 PC로 옮길 수 있다.
  Dockerfile 빌드 없이 바로 사용 가능.

  --- 저장 (현재 PC에서) ---

    docker save bead-humble-torch -o bead-humble-torch.tar

    파일 크기: 약 13GB
    압축하면 절반 정도로 줄일 수 있다:

    docker save bead-humble-torch | gzip > bead-humble-torch.tar.gz

  --- 불러오기 (새 PC에서) ---

    docker load -i bead-humble-torch.tar

    또는 압축 버전:

    gunzip -c bead-humble-torch.tar.gz | docker load

    확인:

    docker images bead-humble-torch

  --- USB로 옮기기 ---

    USB에 .tar 또는 .tar.gz 파일을 복사해서 새 PC로 옮기면 된다.
    빌드 필요 없이 docker load만 하면 바로 사용 가능.


================================================================================
  3. 컨테이너 생성 (docker run)
================================================================================

  이미지가 준비되면 아래 명령어로 컨테이너를 만든다.
  이 명령어가 현재 bead_gpu 컨테이너와 동일한 조건이다:

    docker run -it \
      --name bead_gpu \
      --runtime=nvidia \
      --gpus all \
      --privileged \
      --network host \
      -v /dev:/dev \
      -v /home/ho/BEADtrain:/workspace/BEADtrain \
      -w /workspace/BEADtrain \
      -e NVIDIA_VISIBLE_DEVICES=all \
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
      bead-humble-torch

  실행하면 바로 컨테이너 안의 bash 셸로 진입된다.

  ※ 다른 PC에서는 -v 경로를 해당 PC에 맞게 변경:
     -v /home/새유저/BEADtrain:/workspace/BEADtrain

  --- 각 옵션 설명 ---

    --name bead_gpu        컨테이너 이름
    --runtime=nvidia       NVIDIA GPU 런타임 사용
    --gpus all             모든 GPU를 컨테이너에 할당
    --privileged           USB 장치 접근 허용 (RealSense 카메라 필수)
    --network host         호스트 네트워크 그대로 사용 (ROS2 DDS 통신)
    -v /dev:/dev           USB 장치 파일 마운트 (카메라 접근)
    -v ...BEADtrain:...    작업 폴더를 컨테이너 안에 마운트
    -w /workspace/...      시작 작업 디렉토리
    -e NVIDIA_...          GPU 환경변수


================================================================================
  4. 컨테이너 진입 / 시작 / 정지
================================================================================

  # 컨테이너가 정지 상태일 때 시작
  docker start bead_gpu

  # 실행 중인 컨테이너에 진입 (터미널 여러 개 열 수 있음)
  docker exec -it bead_gpu bash

  # 컨테이너 정지
  docker stop bead_gpu

  # 컨테이너 삭제 (다시 만들어야 함)
  docker rm bead_gpu

  # 컨테이너 상태 확인
  docker ps -a | grep bead_gpu


================================================================================
  5. 컨테이너 안에서 환경 설정
================================================================================

  컨테이너에 진입하면 매번 아래를 실행해야 한다:

    source /opt/ros/humble/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

  ※ 매번 치기 귀찮으면 .bashrc에 추가:

    echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
    echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> ~/.bashrc


================================================================================
  6. 카메라 + 노드 실행 (동작 확인)
================================================================================

  --- 터미널 1: 카메라 ---

    docker exec -it bead_gpu bash
    source /opt/ros/humble/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

    ros2 launch realsense2_camera rs_launch.py \
      depth_module.depth_profile:=848x480x15 \
      depth_module.color_profile:=1280x720x15 \
      publish_tf:=false

  --- 터미널 2: 메인 노드 ---

    docker exec -it bead_gpu bash
    source /opt/ros/humble/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

    cd /workspace/BEADtrain/nodes
    python3 -u tcp_test_client_V3.py

  --- GPU 동작 확인 ---

    docker exec -it bead_gpu bash
    python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    # True NVIDIA GeForce RTX XXXX 가 나오면 OK


================================================================================
  7. 현재 설치된 주요 패키지 버전 (참고)
================================================================================

  Python:
    torch                       2.10.0.dev20251122+cu128
    torchvision                 0.25.0.dev20251122+cu128
    segmentation_models_pytorch 0.5.0
    albumentations              2.0.8
    opencv-python-headless      4.12.0.88

  ROS2:
    ros-humble-desktop          0.10.0
    ros-humble-cv-bridge
    ros-humble-realsense2-camera 4.56.4
    ros-humble-cyclonedds       0.10.5
    ros-humble-rmw-cyclonedds-cpp 1.3.4
    ros-humble-control-msgs     4.8.0

  GPU (현재 시스템):
    NVIDIA GeForce RTX 5070 (12GB)
    Driver: 570.172.08


================================================================================
  8. rx160_noetic_clone 컨테이너 (ROS1 Noetic — 로봇 드라이버)
================================================================================

  bead_gpu와 별도로, 로봇 드라이버용 ROS1 Noetic 컨테이너가 있다.
  두 컨테이너를 ros1_bridge로 연결해서 ROS1↔ROS2 통신한다.

  --- 구조 ---

    bead_gpu (ROS2 Humble)
        ↕  ros1_bridge (bead_gpu 안에서 실행)
    rx160_noetic_clone (ROS1 Noetic, MoveIt, RViz, 로봇 드라이버)

    둘 다 --network host 이므로 같은 네트워크에서 통신한다.

  --- rx160_noetic_clone 컨테이너 생성 조건 ---

    이미지:   rx160_noetic_full:latest
    네트워크: host
    마운트:   /tmp/.X11-unix:/tmp/.X11-unix (GUI 표시용)
              /home/ho/urdf_file:/root/urdf_file (URDF 파일)

    docker run -it \
      --name rx160_noetic_clone \
      --network host \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      -v /home/ho/urdf_file:/root/urdf_file \
      -e DISPLAY=$DISPLAY \
      rx160_noetic_full:latest

    ※ GUI(RViz) 사용하려면 호스트에서 먼저: xhost +local:docker

  --- rx160_noetic_full 이미지도 저장/옮기기 ---

    docker save rx160_noetic_full:latest | gzip > ~/rx160-noetic-full.tar.gz

    새 PC에서:
    gunzip -c rx160-noetic-full.tar.gz | docker load


================================================================================
  9. ros1_bridge — 실체와 동작 원리
================================================================================

  --- 이게 뭐냐 ---

    ros1_bridge는 별도 도커가 아니다.
    bead_gpu 컨테이너 안에 설치된 ROS2 패키지(프로그램)이다.

    호스트에서 hbridge를 치면 이런 일이 일어난다:

      [호스트 터미널]
        hbridge  (= alias)
          ↓
      [실제로 실행되는 명령어]
        docker exec -it bead_gpu bash /opt/ros-humble-ros1-bridge/start_bridge.sh
          ↓
      [bead_gpu 컨테이너 안에서]
        ros2 run ros1_bridge dynamic_bridge 프로그램이 실행됨
          ↓
      [bridge가 하는 일]
        ROS2 토픽 ←→ ROS1 토픽을 자동 변환/중계

    즉, "호스트에서 명령어를 치지만 실제로는 bead_gpu 안에서 돌아가는 것"이다.
    docker exec는 이미 실행 중인 컨테이너 안에서 명령어를 실행하는 도커 기능이다.

  --- 왜 bead_gpu 안에 설치했나 ---

    ros1_bridge는 ROS1과 ROS2 양쪽 라이브러리가 동시에 필요하다.
    bead_gpu(Humble) 안에 ROS1 Noetic 라이브러리를 넣고 소스 빌드한 것이다.
    /opt/ros-humble-ros1-bridge/ 에 빌드 결과물이 들어 있다.

    이 빌드물은 bead-humble-torch 이미지가 아니라 컨테이너 안에서 추가로
    빌드한 것이라, docker save로 저장한 이미지에는 포함되지 않는다.

    ※ 중요: bead-humble-torch.tar.gz 이미지에는 bridge가 없다.
    컨테이너를 통째로 저장하거나, 새 PC에서 다시 빌드해야 한다.

  --- 컨테이너 통째로 저장하는 방법 (bridge 포함) ---

    # 현재 bead_gpu 컨테이너 상태를 통째로 이미지로 만들기
    docker commit bead_gpu bead-gpu-full:latest

    # 저장
    docker save bead-gpu-full:latest | gzip > ~/bead-gpu-full.tar.gz

    # 새 PC에서 불러오기
    gunzip -c bead-gpu-full.tar.gz | docker load

    # 컨테이너 생성 (이미지 이름만 변경)
    docker run -it \
      --name bead_gpu \
      --runtime=nvidia \
      --gpus all \
      --privileged \
      --network host \
      -v /dev:/dev \
      -v /home/새유저/BEADtrain:/workspace/BEADtrain \
      -w /workspace/BEADtrain \
      -e NVIDIA_VISIBLE_DEVICES=all \
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
      bead-gpu-full:latest

    이렇게 하면 bridge, .bashrc alias, 추가 설치한 패키지 전부 포함된다.

  --- 통신 구조 ---

    ┌─────────────────────────────────────────────┐
    │  bead_gpu 컨테이너 (--network host)          │
    │                                             │
    │  ROS2 노드들 (카메라, UNet, pose)             │
    │       ↕                                     │
    │  ros1_bridge (dynamic_bridge)               │
    │       │  ROS_MASTER_URI=localhost:11311      │
    └───────┼─────────────────────────────────────┘
            │  같은 localhost (둘 다 --network host)
    ┌───────┼─────────────────────────────────────┐
    │       ↓                                     │
    │  roscore (포트 11311)                        │
    │  MoveIt + 로봇 드라이버                       │
    │                                             │
    │  rx160_noetic_clone 컨테이너 (--network host) │
    └─────────────────────────────────────────────┘

    --network host 덕분에 두 컨테이너가 같은 localhost를 공유한다.
    bridge가 localhost:11311의 roscore에 연결되어 토픽을 양방향 변환한다.

  --- 실행 방법 ---

    # 호스트에서 (단축어)
    hbridge

    # 또는 직접
    docker exec -it bead_gpu bash /opt/ros-humble-ros1-bridge/start_bridge.sh

  --- start_bridge.sh 내용 ---

    export LD_LIBRARY_PATH=/opt/ros-humble-ros1-bridge/install/ros1_bridge/lib:$LD_LIBRARY_PATH
    source /opt/ros/humble/setup.bash
    source /opt/ros-humble-ros1-bridge/install/local_setup.bash
    export ROS_MASTER_URI=http://localhost:11311
    ros2 run ros1_bridge dynamic_bridge

  --- 호스트 alias (새 PC에서도 등록해야 함) ---

    echo "alias hbridge='docker exec -it bead_gpu bash /opt/ros-humble-ros1-bridge/start_bridge.sh'" >> ~/.bashrc
    source ~/.bashrc

  --- 주의 ---

    - rx160_noetic_clone의 roscore가 먼저 떠 있어야 bridge가 연결됨
    - 둘 다 --network host여야 localhost:11311로 통신 가능
    - ros-humble-control-msgs 패키지가 bead_gpu에 설치되어 있어야 함 (이미 설치됨)


================================================================================
  10. bead_gpu 컨테이너 안의 alias / 환경 설정 (.bashrc)
================================================================================

  컨테이너 안 /root/.bashrc에 아래 설정이 들어 있다:

    source /opt/ros/humble/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI=file:///workspace/BEADtrain/cyclonedds.xml
    umask 000

  alias:
    real  = 카메라 실행 (1280x720x30, align_depth 포함)
    tf    = 고정 TF 발행 (tool0→camera_color_optical_frame)
    sc    = source /opt/ros/humble/setup.bash

  ※ 새 컨테이너를 만들면 이 alias가 없으니 직접 추가해야 한다:

    cat >> /root/.bashrc << 'EOF'
    source /opt/ros/humble/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI=file:///workspace/BEADtrain/cyclonedds.xml
    umask 000

    alias real='ros2 launch realsense2_camera rs_launch.py depth_module.depth_profile:=1280x720x30 depth_module.color_profile:=1280x720x30 align_depth.enable:=true'
    alias sc='source /opt/ros/humble/setup.bash'
    EOF


================================================================================
  11. 전체 실행 순서 요약
================================================================================

  1. docker start rx160_noetic_clone
     → roscore + 로봇 드라이버 실행

  2. docker start bead_gpu

  3. bead_gpu 터미널 1: 카메라
     docker exec -it bead_gpu bash
     ros2 launch realsense2_camera rs_launch.py \
       depth_module.depth_profile:=848x480x15 \
       depth_module.color_profile:=1280x720x15 \
       publish_tf:=false

  4. 호스트 터미널: ros1_bridge
     hbridge

  5. bead_gpu 터미널 3: 메인 노드
     docker exec -it bead_gpu bash
     cd /workspace/BEADtrain/nodes
     python3 -u tcp_test_client_V3.py


================================================================================
  12. 문제 해결
================================================================================

  Q: docker run 할 때 "nvidia runtime not found" 에러
  A: NVIDIA Container Toolkit이 안 설치됨. 0번 (3)항 참고.

  Q: 카메라가 안 잡힘
  A: --privileged -v /dev:/dev 옵션 확인. USB 케이블 다시 연결.
     호스트에서 lsusb | grep Intel 로 카메라 보이는지 확인.

  Q: ROS2 토픽이 호스트에서 안 보임
  A: --network host 옵션 확인.
     컨테이너 안에서 export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp 했는지 확인.
     호스트에서도 동일한 DDS 구현 사용하는지 확인.

  Q: torch.cuda.is_available() 이 False
  A: --runtime=nvidia --gpus all 옵션 확인.
     nvidia-smi가 호스트에서 동작하는지 확인.

  Q: 이미 bead_gpu 이름의 컨테이너가 있다고 에러
  A: docker rm bead_gpu 후 다시 docker run.

================================================================================
