#!/usr/bin/env python3
#그냥 이미지 1개, 오버레이 이미지, 외곽선 이미지, 중심 경로 뽑은 이미지를 저장하는 방식으로 
# png 파일로 저장 
# 

import os
from datetime import datetime

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import segmentation_models_pytorch as smp

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ==============================
# 0) 모델 / 전처리 설정
# ==============================
IMAGE_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

MODEL_PATH = "/workspace/BEADtrain/modelresult/1125unet.pth"

# ===== contour / width 저장 =====
CONTOUR_CSV = "bead_contour.csv"
WIDTH_CSV   = "bead_width.csv"
NUM_WIDTH_SLICES = 5

# ===== 이미지 저장 경로 =====
IMAGE_SAVE_DIR = "/workspace/BEADtrain/image"

# ==============================
# ✅ 안정 프레임 선정 파라미터 (튜닝 포인트)
# ==============================
WARMUP_SEC = 1.0           # 시작 후 이 시간 동안은 무시(카메라 AE/AWB 안정화)
TIMEOUT_SEC = 8.0          # 이 시간 안에 조건 만족 못하면 best 후보로 강제 확정
NEEDED_GOOD = 3            # 연속으로 조건 만족해야 하는 프레임 수

# "비드가 제대로 잡혔다"를 판정할 조건(환경마다 조정 필요)
MIN_MASK_AREA_PX = 150        # mask의 흰 픽셀 수
MIN_CONTOUR_AREA_PX = 120     # largest contour 면적


# ==============================
# UNet Node
# ==============================
class BeadUNetNode(Node):
    def __init__(self):
        super().__init__("bead_unet_node")

        self.bridge = CvBridge()
        self.get_logger().info("✅ BeadUNetNode started")

        # ===== 기존 상태 변수 =====
        self.contour_extracted = False
        self.saved_contour = None
        self.width_slices_y = []

        # ===== ✅ 안정 프레임 선택용 상태 =====
        self.start_time = self.get_clock().now()
        self.good_streak = 0
        self.published_once = False

        # "연속 good 구간" 중 베스트 프레임(가장 큰 contour area)을 저장
        self.best_msg = None
        self.best_frame_bgr = None
        self.best_mask_resized = None
        self.best_contour = None
        self.best_score = -1  # 여기서는 largest contour area로 score 사용

        # 확정된 스냅샷 메시지 (재발행용)
        self.final_mask_msg = None
        self.final_overlay_msg = None

        # -----------------------------
        # 1) Model load
        # -----------------------------
        self.get_logger().info("✅ Loading Unet(resnet34) model...")

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        ).to(DEVICE)

        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.get_logger().info(f"✅ Model loaded from {MODEL_PATH}")
        self.get_logger().info(f"✅ Device: {DEVICE}")

        # -----------------------------
        # 2) QoS
        # -----------------------------
        # 카메라는 RELIABLE로 publish
        qos_camera = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        # mask/overlay는 BEST_EFFORT
        self.qos_sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # -----------------------------
        # 3) Subscriber / Publisher
        # -----------------------------
        CAMERA_TOPIC = "/camera/camera/color/image_rect_raw"
        self.get_logger().info(f"📡 Subscribing to: {CAMERA_TOPIC}")

        self.sub_image = self.create_subscription(
            Image,
            CAMERA_TOPIC,
            self.image_callback,
            qos_camera,
        )

        self.pub_mask = self.create_publisher(Image, "/bead/mask", self.qos_sensor)
        self.pub_overlay = self.create_publisher(Image, "/bead/overlay", self.qos_sensor)

    # ==============================
    # 내부 유틸: contour/width 저장 + overlay 생성
    # (기존 기능 유지)
    # ==============================
    def compute_and_save_contour_width_once(self, mask_resized: np.ndarray):
        """
        - 최초 1회 contour + width 계산하고 CSV 저장
        - self.saved_contour / self.width_slices_y 갱신
        """
        if self.contour_extracted:
            return

        contours, _ = cv2.findContours(
            mask_resized,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            return

        bead_contour = max(contours, key=cv2.contourArea)
        self.saved_contour = bead_contour
        self.contour_extracted = True

        # contour 저장 (기존 유지)
        np.savetxt(
            CONTOUR_CSV,
            bead_contour.reshape(-1, 2),
            delimiter=",",
            header="x,y",
            comments=""
        )

        # width 계산 + CSV 저장 (기존 유지)
        x, y, bw, bh = cv2.boundingRect(bead_contour)
        ys = np.linspace(y, y + bh, NUM_WIDTH_SLICES + 2)[1:-1]

        self.width_slices_y = []

        with open(WIDTH_CSV, "w") as f:
            f.write("slice_y,x_start,width_px\n")
            for y_line in ys:
                xs = [
                    p[0][0] for p in bead_contour
                    if abs(p[0][1] - y_line) < 2
                ]
                if len(xs) >= 2:
                    x_min = min(xs)
                    x_max = max(xs)
                    width = x_max - x_min
                    f.write(f"{int(y_line)},{int(x_min)},{int(width)}\n")
                    self.width_slices_y.append(int(y_line))

        self.get_logger().info("📐 Bead contour & width saved (once)")

    def make_overlay(self, frame_bgr: np.ndarray, mask_resized: np.ndarray) -> np.ndarray:
        """
        - 기존 overlay(빨간 반투명) + contour(초록) + 폭선(노랑) 시각화 유지
        """
        overlay = frame_bgr.copy()

        # 기존 비드 overlay
        color = np.array([255, 0, 0], dtype=np.uint8)
        alpha = 0.5
        mask_bool = mask_resized > 0
        overlay[mask_bool] = (
            alpha * color + (1 - alpha) * overlay[mask_bool]
        ).astype(np.uint8)

        # contour 시각화 (초록색)
        if self.saved_contour is not None:
            cv2.drawContours(
                overlay,
                [self.saved_contour],
                -1,
                (0, 255, 0),
                2
            )

            # 폭 선 시각화 (노란색)
            for y_line in self.width_slices_y:
                xs = [
                    p[0][0] for p in self.saved_contour
                    if abs(p[0][1] - y_line) < 2
                ]
                if len(xs) >= 2:
                    x_min, x_max = min(xs), max(xs)
                    cv2.line(
                        overlay,
                        (x_min, y_line),
                        (x_max, y_line),
                        (0, 255, 255),
                        2
                    )

        return overlay

    def save_snapshot_images(self, frame_bgr: np.ndarray, mask_resized: np.ndarray):
        """스냅샷 확정 시 이미지 저장"""
        # 세션 폴더가 있으면 그 안의 recognition/ 에 저장, 없으면 기존 방식
        session_dir = os.environ.get('GRIND_SESSION_DIR')
        if session_dir:
            save_dir = os.path.join(session_dir, "recognition")
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_dir = os.path.join(IMAGE_SAVE_DIR, timestamp)
        os.makedirs(save_dir, exist_ok=True)

        # 0) 흑백 마스크
        cv2.imwrite(os.path.join(save_dir, "0_mask_bw.png"), mask_resized)

        # 1) 원본 카메라 이미지
        cv2.imwrite(os.path.join(save_dir, "1_raw_camera.png"), frame_bgr)

        # 2) 비드 오버레이만 (빨간 반투명 마스크)
        img_overlay = frame_bgr.copy()
        mask_bool = mask_resized > 0
        img_overlay[mask_bool] = (
            0.5 * np.array([255, 0, 0]) + 0.5 * img_overlay[mask_bool]
        ).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, "2_bead_overlay.png"), img_overlay)

        # 3) 오버레이 + 외곽선만
        img_contour = img_overlay.copy()
        if self.saved_contour is not None:
            cv2.drawContours(img_contour, [self.saved_contour], -1, (0, 255, 0), 2)
            for y_line in self.width_slices_y:
                xs = [p[0][0] for p in self.saved_contour if abs(p[0][1] - y_line) < 2]
                if len(xs) >= 2:
                    cv2.line(img_contour, (min(xs), y_line), (max(xs), y_line), (0, 255, 255), 2)
        cv2.imwrite(os.path.join(save_dir, "3_bead_overlay_contour.png"), img_contour)

        # PCA 중심 경로 계산
        path_pts = []
        ys, xs = np.where(mask_resized > 0)
        if len(xs) >= 10:
            pts = np.stack([xs, ys], axis=1).astype(np.float64)
            mean = pts.mean(axis=0)
            centered = pts - mean
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eig(cov)
            direction = eigvecs[:, np.argmax(eigvals)]

            t = centered @ direction
            t_min, t_max = t.min(), t.max()

            for ti in np.linspace(t_min, t_max, 50):
                pt = mean + ti * direction
                path_pts.append((int(pt[0]), int(pt[1])))

        # 4) 오버레이 + 외곽선 + 폭선 + PCA 중심 경로
        img_path = img_contour.copy()
        if path_pts:
            for i in range(len(path_pts) - 1):
                cv2.line(img_path, path_pts[i], path_pts[i + 1], (0, 0, 255), 2)
        cv2.imwrite(os.path.join(save_dir, "4_bead_overlay_contour_path.png"), img_path)

        # 5) 오버레이 + 외곽선만 (폭선 없이)
        img_contour_only = img_overlay.copy()
        if self.saved_contour is not None:
            cv2.drawContours(img_contour_only, [self.saved_contour], -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(save_dir, "5_bead_overlay_contour_only.png"), img_contour_only)

        # 6) 원본 + PCA 중심 경로만
        img_raw_path = frame_bgr.copy()
        if path_pts:
            for i in range(len(path_pts) - 1):
                cv2.line(img_raw_path, path_pts[i], path_pts[i + 1], (0, 0, 255), 2)
        cv2.imwrite(os.path.join(save_dir, "6_raw_path_only.png"), img_raw_path)

        self.get_logger().info(f"📁 이미지 7개 저장 완료: {save_dir}")

    def _republish_snapshot(self):
        """확정된 스냅샷을 주기적으로 재발행 (RViz/브릿지 놓침 방지)"""
        if self.final_mask_msg is not None:
            self.final_mask_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_mask.publish(self.final_mask_msg)
        if self.final_overlay_msg is not None:
            self.final_overlay_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_overlay.publish(self.final_overlay_msg)

    # ==============================
    # Image callback (안정 프레임 선정 → 스냅샷 확정)
    # ==============================
    def image_callback(self, msg: Image):
        if self.published_once:
            return

        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds / 1e9

        # warmup 구간 무시
        if elapsed < WARMUP_SEC:
            return

        # ROS Image → OpenCV BGR
        frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w, _ = frame_bgr.shape

        # BGR → RGB → Inference
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = transform(frame_rgb).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.model(img_tensor)
            probs  = torch.sigmoid(logits)
            pred   = (probs > 0.5).float()

        mask_np = pred.squeeze().cpu().numpy().astype(np.uint8) * 255
        mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

        # 품질 판정
        mask_area = int(cv2.countNonZero(mask_resized))
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour_area = int(cv2.contourArea(max(contours, key=cv2.contourArea))) if contours else 0

        good = (mask_area >= MIN_MASK_AREA_PX and contour_area >= MIN_CONTOUR_AREA_PX)

        if good:
            self.good_streak += 1
            # best score 갱신
            if contour_area > self.best_score:
                self.best_score = contour_area
                self.best_msg = msg
                self.best_frame_bgr = frame_bgr
                self.best_mask_resized = mask_resized
                self.best_contour = max(contours, key=cv2.contourArea)
        else:
            self.good_streak = 0

        self.get_logger().info(
            f"[{elapsed:.1f}s] mask_area={mask_area}, contour_area={contour_area}, "
            f"good_streak={self.good_streak}/{NEEDED_GOOD}"
        )

        # 확정 조건: 연속 good 달성 또는 타임아웃
        confirm = (self.good_streak >= NEEDED_GOOD) or (elapsed >= TIMEOUT_SEC and self.best_score > 0)

        if not confirm:
            return

        # ===== 스냅샷 확정 =====
        self.get_logger().info(f"📸 Snapshot confirmed! score={self.best_score}")

        frame_bgr = self.best_frame_bgr
        mask_resized = self.best_mask_resized
        self.saved_contour = self.best_contour

        # contour/width CSV 저장
        self.compute_and_save_contour_width_once(mask_resized)

        # mask + overlay 메시지 생성 & 저장
        mask_msg = self.bridge.cv2_to_imgmsg(mask_resized, encoding="mono8")
        mask_msg.header = self.best_msg.header
        self.final_mask_msg = mask_msg
        self.pub_mask.publish(mask_msg)

        overlay = self.make_overlay(frame_bgr, mask_resized)
        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        overlay_msg.header = self.best_msg.header
        self.final_overlay_msg = overlay_msg
        self.pub_overlay.publish(overlay_msg)

        # ===== PNG 이미지 저장 =====
        self.save_snapshot_images(frame_bgr, mask_resized)

        self.published_once = True
        self.get_logger().info("✅ Snapshot published! (contour + width saved)")

        # 카메라 구독 끊기 (GPU 절약)
        self.destroy_subscription(self.sub_image)

        # 1초마다 스냅샷 재발행 (RViz/브릿지 놓침 방지)
        self.create_timer(1.0, self._republish_snapshot)


# ==============================
# Main
# ==============================
def main(args=None):
    rclpy.init(args=args)
    node = BeadUNetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
