# BEADtrain 프로젝트 구조

비드 그라인딩 자동화 시스템 — UNet 세그멘테이션 기반 비드 인식 + Staubli RX160 로봇 제어

---

## 시스템 개요

```
카메라(D405) → UNet 비드 인식 → 3D 경로 생성 → TCP 통신 → Staubli RX160 그라인딩
                                                    ↓
                                        그라인딩 후 재스캔 → 판정(OK/NOK) → 반복 or 정지
```

- **bead_gpu 컨테이너**: ROS2 Humble, Ubuntu 22.04, GPU(CUDA), 카메라 + 비드 노드 실행
- **rx160_noetic_clone 컨테이너**: ROS1 Noetic, MoveIt + RViz, 로봇 드라이버
- **호스트**: ROS2 Jazzy, Ubuntu 24.04
- **로봇**: Staubli RX160, TCP 소켓 통신 (172.31.0.1, 포트 11000~11003)

---

## 폴더 구조 요약

```
BEADtrain/
├── nodes/              # 핵심 ROS2 노드 (실시간 실행 코드)
├── train/              # UNet 모델 학습 및 평가
├── tools/              # 오프라인 분석/시각화 유틸리티
├── test/               # 카메라, 뎁스 센서 테스트
├── calibration/        # 핸드-아이 캘리브레이션
├── VAL3_PA_grd_fin/    # Staubli 로봇 컨트롤러 VAL3 프로그램
├── staubli_configs/    # Staubli 컨트롤러 설정 백업 (.cfx)
├── modelresult/        # 학습된 UNet 모델 가중치 + 결과
├── REAL/               # 학습용 이미지/마스크 데이터셋
├── scan_image/         # 실험별 그라인딩 스캔 이미지
├── archive/            # 이전 버전 코드 백업
└── (루트 파일들)        # CSV, 설정, 로그
```

---

## 1. nodes/ — 핵심 ROS2 노드

실시간으로 bead_gpu 컨테이너 안에서 실행되는 메인 코드들.

| 파일 | 역할 |
|------|------|
| **tcp_test_client_V3.py** | 전체 그라인딩 루프 오케스트레이터. TCP 소켓 4개(Command/Motion/Feedback/Joint)로 Staubli 로봇과 통신하며, 인식→경로생성→그라인딩→재스캔→판정 사이클을 반복 제어 |
| **bead_unet_node.py** | 그라인딩 전 비드 인식 노드. 카메라 영상에서 UNet으로 비드를 세그멘테이션하고, 외곽선(contour)·폭(width) CSV와 마스크를 저장. 모델: `1125unet.pth` (1024 입력) |
| **bead_pose_node.py** | 3D 경로 생성 노드. 비드 마스크 + 뎁스 영상으로 3D 좌표를 계산하고, PCA 기반 웨이포인트를 생성하여 /bead/grind_path_base 로 퍼블리시. 핸드-아이 변환은 직접 4x4 행렬 계산 |
| **bead_unet_node_after.py** | 그라인딩 후 판정 노드. 재스캔 영상에서 UNet으로 잔여 비드를 인식하고, 기준 contour 대비 shape match ratio로 OK/NOK 판정. 오버컷(과삭) 감지 시에도 OK 처리. 모델: `2026-03-07_23-23-27/unet_best.pth` (1280 입력) |

### 실행 순서 (tcp_test_client_V3가 제어)
1. `bead_unet_node.py` — 비드 인식 + CSV 저장
2. `bead_pose_node.py` — 3D 웨이포인트 생성
3. 로봇 그라인딩 실행
4. `bead_unet_node_after.py` — 재스캔 + 판정 → OK면 정지, NOK면 3번 반복

---

## 2. train/ — UNet 모델 학습 및 평가

| 파일 | 역할 |
|------|------|
| **bead_train.py** | UNet 학습 초기 버전. A.Resize 정사각 리사이즈(1024), BCE+Dice 손실, EarlyStopping. IMAGE_DIR/MASK_DIR 미정의 상태 — bead_train_end.py가 개선 버전 |
| **bead_train_end.py** | UNet 학습 개선 버전 (현재 사용). LongestMaxSize+PadIfNeeded 비율 유지(1280), pos_weight 자동 계산, gradient clipping. 결과: unet_best.pth, loss_curve.png, loss_history.csv, train_result.txt |
| **bead_train_end_test.py** | 기존 모델 평가 전용 (학습 없이). 동일 데이터 분할(SEED=42) 재현하여 Train/Val/Test 전체 Loss·IoU·Dice·Precision·Recall 산출 |
| **evaluation.py** | Test set 평가 + GPU 추론 속도(ms) 측정. evaluation_result.txt 저장 |
| **convert_json_to_mask_png.py** | Labelme JSON 라벨링 파일 → 바이너리 마스크 PNG 변환. "end" 라벨 polygon을 흰색(255)으로 채움. 학습 데이터 준비용 |
| **unet_diagram.py** | UNet(ResNet34) 아키텍처 블록 다이어그램을 matplotlib으로 생성. unet_architecture.png 저장 |

---

## 3. tools/ — 오프라인 분석·시각화 유틸리티

| 파일 | 역할 |
|------|------|
| **bead_judge_offline.py** | bead_unet_node_after.py의 오프라인 버전. 이미지 파일 경로를 입력받아 UNet 추론 → 기준 contour 대비 판정 → 마스크·오버레이(녹색/주황/빨강) 이미지 저장. 여러 스캔을 일괄 처리 가능 |
| **test_overlay.py** | 단일 이미지에 UNet 추론 → 오버레이·외곽선·마스크 이미지 저장. 모델 인식 결과 빠르게 확인할 때 사용 |
| **visualize_segmentation.py** | 세그멘테이션 결과 시각화 (원본 / 마스크 / 오버레이 3열). torchvision 전처리 사용 |
| **calc_tcp_waypoints.py** | 캘리브레이션 결과(T_tool0_cam)와 tTool을 이용해 TCP(그라인더) 기준 웨이포인트를 계산 |

---

## 4. test/ — 카메라·뎁스 센서 테스트

| 파일 | 역할 |
|------|------|
| **camera.py** | RealSense D405 컬러 영상 녹화 + 프레임 캡처 ('s' 키로 PNG 저장) |
| **cameravideo.py** | D405 컬러 스트림 뷰어 + 이미지 캡처 (원본 해상도 1280x720 PNG 저장) |
| **camera_test_node.py** | 카메라 TF 검증용 ROS2 노드. 카메라 앞 0.5m 지점을 base_link 좌표로 변환하여 퍼블리시 |
| **depth_accuracy_test.py** | D405 뎁스 정확도 테스트. ROI 중심영역의 depth 통계(평균/표준편차/min/max) 실시간 표시. 여러 거리에서 측정값 기록 |
| **depth_bead_height_test.py** | 비드 높이(Z 오차) 측정. 기준 평면 depth와 비드 영역 depth의 차이로 비드 높이(mm) 산출 |
| **depth_check.py** | ROS2 depth 토픽 1회 수신하여 중앙 ROI의 depth 값(mm/cm) 확인하는 간단 테스트 |

---

## 5. calibration/ — 핸드-아이 캘리브레이션

ArUco 보드를 이용한 eye-on-hand 캘리브레이션 도구들.

| 파일 | 역할 |
|------|------|
| **handeye_calib_node.py** | 메인 캘리브레이션 ROS2 노드. ArUco 보드 검출 + 로봇 관절값 수신 → 여러 자세에서 샘플 수집 → OpenCV hand-eye 캘리브레이션 수행 |
| **handeye_calib_node copy.py** | 위 노드의 백업 복사본 |
| **cal.py** | 캘리브레이션 잔차(residual) 분석. 수집된 샘플의 T_base_target 일관성 검증 |
| **find_transform.py** | 플랜지 좌표 기반 T_world_base 변환 탐색. 티치펜던트 좌표와 비교하여 변환 행렬 검증 |
| **tf_test_aruco.py** | ArUco 보드 인식 + TF 체인(base_link→tool0→camera→aruco) 좌표 출력 테스트 |
| **verify_node.py** | 캘리브레이션 현장 검증 노드. 로봇을 여러 자세로 움직이면서 보드의 base 좌표 일관성 확인 (std < 3mm이면 양호) |
| **view_samples.py** | 수집된 캘리브레이션 샘플 데이터 뷰어. reproj 오차, 회전/이동 행렬 등 확인 |

### 캘리브레이션 데이터
- `handeye_samples_latest/` — R/t 행렬, reproj 오차 등 (.npy 파일)
- `handeye_samples_latest.npz` — 위 데이터의 압축 아카이브
- `result.txt` — 캘리브레이션 결과 요약

---

## 6. VAL3_PA_grd_fin/ — Staubli 로봇 VAL3 프로그램

Staubli CS9 컨트롤러에서 실행되는 VAL3 프로그램. tcp_test_client_V3.py의 TCP 통신 상대편.

| 파일 | 역할 |
|------|------|
| `_PA_grd_fin.pjx` | VAL3 프로젝트 파일 |
| `_PA_grd_fin.dtx` | 데이터 파일 (변수, 좌표 등) |
| `start.pgx` | 프로그램 시작 |
| `stop.pgx` | 프로그램 정지 |
| `_mainLoop.pgx` | 메인 루프 (TCP 명령 수신 → 분기) |
| `_TmainLoop.pgx` | 스레드용 메인 루프 |
| `_recvWaypoints.pgx` | TCP로 웨이포인트 수신 (Motion 포트 11001) |
| `_recvFloat.pgx` | TCP로 float 데이터 수신 |
| `_sendJoint.pgx` | 관절값 전송 (Joint 포트 11003) |
| `_sendStatus.pgx` | 상태 전송 (Feedback 포트 11002) |
| `_startPass.pgx` | 그라인딩 패스 시작 실행 |
| `_moveToScan.pgx` | 스캔 위치로 이동 |
| `_finish.pgx` | 작업 종료 처리 |
| `Alter.pgx` | Alter 모션 (실시간 경로 보정) |
| `_alter_stauts.pgx` | Alter 상태 관리 |
| `_Io.pgx` | IO 제어 (그라인더 ON/OFF 등) |

---

## 7. staubli_configs/ — Staubli 컨트롤러 설정 백업

| 파일 | 역할 |
|------|------|
| `arm.cfx` | 로봇 팔 기구학 설정 (관절 범위, DH 파라미터) |
| `cell.cfx` | 작업 셀 환경 설정 (안전 영역, 주변 장치) |
| `controller.cfx` | 컨트롤러 통신/시스템 설정 (IP, 포트, IO) |

※ Python 코드에서 참조하지 않음. CS9 컨트롤러에 직접 업로드/복원하는 용도의 백업 파일.

---

## 8. modelresult/ — 학습된 UNet 모델

| 경로 | 설명 |
|------|------|
| `1125unet.pth` | bead_unet_node.py(인식)에서 사용하는 모델 (1024 입력, torchvision 전처리) |
| `2026-02-12_19-28-57/unet_best.pth` | 이전 판정 모델 |
| `2026-03-07_23-23-27/unet_best.pth` | bead_unet_node_after.py(판정)에서 사용하는 현재 모델 (1280 입력, albumentations 전처리) |
| `unet_architecture2.png` | UNet 아키텍처 다이어그램 |

---

## 9. REAL/ — 학습용 데이터셋

| 경로 | 설명 |
|------|------|
| `end/endimg/` | 그라인딩 후 이미지 (학습용) |
| `end/endmask/` | 그라인딩 후 마스크 (학습용) |
| `Bead학습이미지/fitimage/` | 비드 이미지 (학습용) |
| `Bead학습이미지/fitmask/` | 비드 마스크 (학습용) |
| `*.mp4` | 카메라 녹화 영상들 |

---

## 10. scan_image/ — 실험별 그라인딩 스캔 이미지

| 경로 | 설명 |
|------|------|
| `2026-03-04_08-24-56실험2/` | 실험 2 (recognition/ + scan_1~scan_9/) |
| `2026-03-05_05-49-02실험3/` | 실험 3 |
| `망한거/` | 실패한 실험 데이터 |

각 실험 폴더 구조:
```
실험N/
├── recognition/          # 그라인딩 전 인식 결과 (raw, overlay, contour, CSV)
├── scan_1/               # 1회차 재스캔 (raw.png, overlay_color.png, overlay_judge.png)
├── scan_2/
└── ...
```

---

## 11. archive/ — 이전 버전 코드 백업

| 파일 | 역할 |
|------|------|
| `bead_pose_node실시간 조인트 수신.py` | bead_pose_node 이전 버전 (TF lookup 방식, 현재는 직접 행렬 계산) |
| `bead_train copy.py` | bead_train.py 복사본 (endgrinding 데이터용) |
| `deletefile.py` | 이미지/마스크 1:1 매칭 안 되는 파일 삭제 유틸 |

---

## 12. 루트 파일

| 파일 | 역할 |
|------|------|
| `bead_contour.csv` | 비드 외곽선 좌표 (이전 저장분) |
| `bead_width.csv` | 비드 폭 데이터 (이전 저장분) |
| `bead_waypoints.csv` | 비드 웨이포인트 좌표 |
| `scan_result_latest.txt` | 최신 스캔 판정 결과 (bead_unet_node_after → tcp_test_client_V3가 읽음) |
| `grinding_log.txt` | 그라인딩 로그 |
| `cyclonedds.xml` | CycloneDDS 설정 (ROS2 DDS 통신) |
