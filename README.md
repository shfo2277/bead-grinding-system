# 용접 비드 자동 그라인딩 시스템

카메라로 용접 비드를 인식하고, 산업용 로봇(Staubli RX160)이 비드를 그라인딩한 뒤 **다시 스캔해서 연삭이 끝났는지(종료점)를 스스로 판정**하는 자동화 시스템입니다.
판정이 OK가 나올 때까지 `인식 → 경로 생성 → 그라인딩 → 재스캔 → 판정` 사이클을 반복합니다.

```
 RealSense D405 ─▶ bead_unet_node ─▶ bead_pose_node ──▶ tcp_test_client_V3 ──TCP──▶ Staubli RX160 (VAL3)
   (컬러+뎁스)       비드 인식(UNet)    3D 경로 생성          오케스트레이터                 그라인딩
                                                               ▲
                                                               │ OK → 정지 / NOK → 반복
                                         bead_unet_node_after ─┘
                                         재스캔 · 종료점 판정(UNet)
```

## 주요 기능

**1. 비드 인식** – `nodes/bead_unet_node.py`
- UNet(ResNet34 인코더)으로 컬러 영상에서 용접 비드 세그멘테이션
- 연속 3프레임이 안정적일 때 결과를 확정하고, 비드 외곽선(contour)과 폭을 CSV로 저장 → 이후 판정의 **기준**이 됨

**2. 3D 그라인딩 경로 생성** – `nodes/bead_pose_node.py`
- 비드 마스크 + aligned depth → 카메라 3D 좌표 → 핸드-아이 변환 행렬로 로봇 world 좌표 변환
- depth 이상값 제거(median ± 3·MAD) 후 XY 평면 **PCA로 비드 방향**을 구해 웨이포인트 30개 생성

**3. 로봇 제어 / 전체 루프** – `nodes/tcp_test_client_V3.py`
- Staubli 컨트롤러와 TCP 소켓 4개로 통신 (Motion 11000 / Command 11001 / Feedback 11002 / Joint 11003)
- 인식·경로·판정 노드를 subprocess로 실행/종료하며 사이클 관리
- 2회차부터는 첫 경로를 재사용하고 **Z축을 0.5 mm씩 내려가며** 반복 연삭

**4. 종료점 판정** – `nodes/bead_unet_node_after.py`
- 그라인딩 후 재스캔 영상을 별도 학습한 UNet(1280 입력)으로 세그멘테이션
- 기준 외곽선 대비 연삭된 영역 비율(**contour coverage ≥ 93%**)이면 OK, 기준 밖까지 깎인 경우(overcut)도 정지
- 결과를 색으로 표시: 🟩 연삭 완료 · 🟧 잔여 비드 · 🟥 과삭

**5. 핸드-아이 캘리브레이션** – `calibration/`
- ArUco 보드로 eye-in-hand 캘리브레이션, 35개 로봇 자세에서 샘플 수집
- 잔차가 큰 샘플을 걸러낸 뒤 OpenCV 5가지 방법(TSAI, PARK, HORAUD, DANIILIDIS, ANDREFF)을 비교해 median 잔차가 가장 작은 **TSAI(0.55 mm)** 선택 (`cal.py`)
- 로봇을 여러 자세로 움직여 보드 좌표 일관성으로 결과 검증 (`verify_node.py`)

**6. 로봇 프로그램** – `VAL3_PA_grd_fin/`
- Staubli CS9 컨트롤러에서 실행되는 VAL3 프로그램 (명령 수신, 웨이포인트 수신, 스캔 위치 이동, 그라인딩 패스, 그라인더 IO)

## 모델 학습

`train/bead_train_end.py`

- `segmentation_models_pytorch` UNet (ResNet34, ImageNet 사전학습)
- 입력 1280, 비율 유지 리사이즈 + 패딩 (`LongestMaxSize` + `PadIfNeeded`)
- 손실: BCE + Dice, pos_weight 자동 계산, EarlyStopping, ReduceLROnPlateau
- 데이터: 직접 촬영한 비드 이미지를 Labelme로 라벨링 → `convert_json_to_mask_png.py`로 마스크 변환
- 평가: `evaluation.py` – IoU / Dice / Precision / Recall, GPU 추론 속도

## 폴더 구조

```
├── nodes/             실시간 실행 ROS 2 노드 (인식 · 경로 · 판정 · 오케스트레이터)
├── train/             UNet 학습 · 평가 · 라벨 변환
├── tools/             오프라인 판정 · 시각화 유틸리티
├── calibration/       핸드-아이 캘리브레이션 및 검증
├── test/              카메라 · 뎁스 정확도 테스트
├── VAL3_PA_grd_fin/   Staubli 로봇 VAL3 프로그램
├── staubli_configs/   Staubli 컨트롤러 설정 백업
├── docker/            GPU 컨테이너 Dockerfile
└── archive/           이전 버전 코드
```

파일별 자세한 설명은 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md), 실행 환경과 명령어는 [PROJECT_SUMMARY.txt](PROJECT_SUMMARY.txt)에 정리되어 있습니다.

## 개발 환경

| 구분 | 내용 |
| --- | --- |
| 로봇 | Staubli RX160, CS9 컨트롤러 (VAL3) |
| 카메라 | Intel RealSense D405 (컬러 1280×720, 뎁스 848×480) |
| 인식 · 제어 | Docker 컨테이너 – Ubuntu 22.04, ROS 2 Humble, CUDA GPU, CycloneDDS |
| 호스트 | Ubuntu 24.04, ROS 2 Jazzy |
| 딥러닝 | PyTorch, segmentation_models_pytorch, albumentations, OpenCV |

> 학습 데이터(`REAL/`)와 모델 가중치(`modelresult/`, `*.pth`)는 저장소에 포함되어 있지 않습니다.
