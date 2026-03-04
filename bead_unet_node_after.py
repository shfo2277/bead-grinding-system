#!/usr/bin/env python3

import os
from datetime import datetime

import cv2
import csv
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import time

# ==============================
# CONFIG
# ==============================
IMAGE_SIZE = 1280
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "/workspace/BEADtrain/modelresult/2026-02-12_19-28-57/unet_best.pth"

CONTOUR_CSV = "/workspace/BEADtrain/bead_contour.csv"
WIDTH_CSV   = "/workspace/BEADtrain/bead_width.csv"

IMAGE_SAVE_DIR = "/workspace/BEADtrain/image"

SHAPE_THRESH = 0.95
WIDTH_THRESH = 0.95

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

transform = A.Compose([
    A.LongestMaxSize(max_size=IMAGE_SIZE),
    A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE,
                  border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

# ==============================
# Utility functions
# ==============================
def load_contour_csv(path):
    contour = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for x, y in reader:
            contour.append([[int(float(x)), int(float(y))]])
    return np.array(contour, dtype=np.int32)

def load_width_csv(path):
    widths = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for y, x_start, w in reader:
            widths.append(
                (int(float(y)), int(float(x_start)), int(float(w)))
            )
    return widths


def shape_match_ratio(mask, contour):
    contour_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)

    overlap = np.logical_and(mask > 0, contour_mask > 0)
    contour_area = np.count_nonzero(contour_mask)

    return overlap.sum() / max(contour_area, 1)

def width_match_with_ratio(mask, width_refs):
    h, w = mask.shape 
    results = []
    all_ok = True

    for y, x_start, ref_w in width_refs:

        # ---- 범위 체크 ----
        if y < 0 or y >= h:
            all_ok = False
            results.append((y, 0, ref_w, 0.0))
            continue

        x_end = min(x_start + ref_w, w)
        if x_start < 0 or x_start >= w or x_end <= x_start:
            all_ok = False
            results.append((y, 0, ref_w, 0.0))
            continue

        # ---- 기준 폭 ROI ----
        roi = mask[y, x_start:x_end]

        # ---- 채워진 픽셀 수 ----
        cur_w = np.count_nonzero(roi)
        ratio = cur_w / ref_w if ref_w > 0 else 0.0

        if ratio < WIDTH_THRESH:
            all_ok = False

        results.append((y, cur_w, ref_w, ratio))

    return all_ok, results

def put_label(img, text, org, scale=0.8, thickness=2, color=(255, 255, 255)):
    # 검은 배경 박스 + 흰 글씨(기본)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 6, y - th - 10), (x + tw + 6, y + 6), (0, 0, 0), -1)
    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )
# ==============================
# ROS2 Node
# ==============================
class BeadStopJudgeNode(Node):
    def __init__(self):
        super().__init__("bead_stop_judge_node")

        self.bridge = CvBridge()
        self.stop_published = False
        self.image_saved = False

        self.qos_sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Load model
        self.get_logger().info("🔄 Loading UNet model...")
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        ).to(DEVICE)

        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.eval()
        self.get_logger().info("✅ Model loaded")

        # Load reference data
        self.ref_contour = load_contour_csv(CONTOUR_CSV)
        self.ref_widths  = load_width_csv(WIDTH_CSV)
        self.get_logger().info("📐 Reference contour & width loaded")

        # Subscribers
        self.sub_img = self.create_subscription(
            Image,
            "/camera/camera/color/image_rect_raw",
            self.image_callback,
            self.qos_sensor,
        )

        # Publishers (QoS: BEST_EFFORT for bridge compatibility)
        self.pub_mask = self.create_publisher(Image, "/grinding/mask", self.qos_sensor)
        self.pub_raw = self.create_publisher(Image, "/grinding/raw", self.qos_sensor)
        self.pub_overlay = self.create_publisher(Image, "/grinding/overlay", self.qos_sensor)
        self.pub_overlay_compressed = self.create_publisher(
            CompressedImage, "/grinding/overlay/compressed", self.qos_sensor)
        self.pub_stop = self.create_publisher(Bool, "/stop", 10)

    # ==============================
    # Callback
    # ==============================
    def image_callback(self, msg: Image):
        frame_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w, _ = frame_bgr.shape

        # Inference (학습과 동일: 비율 유지 리사이즈 + 패딩)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        augmented = transform(image=frame_rgb)
        img_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prob = torch.sigmoid(self.model(img_tensor))[0, 0]

        mask_full = (prob > 0.4).cpu().numpy().astype(np.uint8) * 255

        # 패딩 영역 제거: 중앙 정렬 패딩의 offset 계산 후 crop
        scale = IMAGE_SIZE / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        pad_top = (IMAGE_SIZE - new_h) // 2
        pad_left = (IMAGE_SIZE - new_w) // 2
        mask_cropped = mask_full[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
        mask = cv2.resize(mask_cropped, (w, h), interpolation=cv2.INTER_NEAREST)

        # Publish grinding mask
        mask_msg = self.bridge.cv2_to_imgmsg(mask, "mono8")
        mask_msg.header = msg.header
        self.pub_mask.publish(mask_msg)

        # Publish raw image (원본)
        raw_msg = self.bridge.cv2_to_imgmsg(frame_bgr, "bgr8")
        raw_msg.header = msg.header
        self.pub_raw.publish(raw_msg)

        # Overlay visualization
        overlay = frame_bgr.copy()
        overlay[mask > 0] = (
            0.5 * overlay[mask > 0] + 0.5 * np.array([255, 0, 0])
        ).astype(np.uint8)


        # ---- STOP JUDGE ----
        shape_ratio = shape_match_ratio(mask, self.ref_contour)
        width_ok, width_details = width_match_with_ratio(mask, self.ref_widths)
        min_width_ratio = min((r for (_, _, _, r) in width_details), default=0.0)

        shape_ok = (shape_ratio >= SHAPE_THRESH)
        ok = (shape_ok and width_ok)

        # 진행률처럼 보이게: 둘 중 부족한 쪽 기준(0~100%)
        progress = min(
            shape_ratio / max(SHAPE_THRESH, 1e-6),
            min_width_ratio / max(WIDTH_THRESH, 1e-6),
        )
        progress_pct = max(0.0, min(1.0, progress)) * 100.0
        will_stop = (shape_ratio >= SHAPE_THRESH and width_ok and not self.stop_published)

        # Draw reference contour
        cv2.drawContours(overlay, [self.ref_contour], -1, (0, 255, 0), 2)

        # Draw width lines
        for y, x_start, ref_w in self.ref_widths:
            if y < 0 or y >= h:
                continue

            x1 = x_start
            x2 = min(x_start + ref_w, w - 1)

            cv2.line(
                overlay,
                (x1, y),
                (x2, y),
                (0, 255, 255),
                2
            )

        # ---- TEXT OVERLAY (Progress / OK-NOK) ----
        put_label(overlay, f"Progress: {progress_pct:5.1f} %", (20, 40))
        put_label(overlay, f"Contour : {shape_ratio*100:5.1f} % (thr {SHAPE_THRESH*100:.0f}%)", (20, 80))
        put_label(overlay, f"WidthMin: {min_width_ratio*100:5.1f} % (thr {WIDTH_THRESH*100:.0f}%)", (20, 120))

        ok_color = (0, 255, 0) if ok else (0, 0, 255)
        put_label(overlay, f"JUDGE: {'OK' if ok else 'NOK'}", (20, 170), scale=1.0, thickness=3, color=ok_color)

        if self.stop_published or will_stop:
            put_label(overlay, "STOP SENT", (20, 215), scale=0.9, thickness=2, color=(0, 255, 255))
        
        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, "bgr8")
        overlay_msg.header = msg.header
        self.pub_overlay.publish(overlay_msg)

        # Compressed overlay for cross-DDS-version (Humble→Jazzy host RViz2)
        comp_msg = CompressedImage()
        comp_msg.header = msg.header
        comp_msg.format = "jpeg"
        comp_msg.data = bytes(cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 80])[1])
        self.pub_overlay_compressed.publish(comp_msg)

        # 이미지 저장 (1회만)
        if not self.image_saved:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_dir = os.path.join(IMAGE_SAVE_DIR, f"bead_grinding_{timestamp}")
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, "1_raw_camera.png"), frame_bgr)
            cv2.imwrite(os.path.join(save_dir, "2_grinding_overlay.png"), overlay)
            self.image_saved = True
            self.get_logger().info(f"📁 그라인딩 판정 이미지 저장: {save_dir}")


        # # ---- LOGGING ----
        # self.get_logger().info(
        #     f"🧩 Contour coverage: {shape_ratio * 100:.2f} %"
        # )

        # for y, cur_w, ref_w, ratio in width_details:
        #     self.get_logger().info(
        #         f"📏 Slice y={y:4d} : "
        #         f"{cur_w:4d}px / {ref_w:4d}px → {ratio * 100:.2f} %"
        #     )


        # self.get_logger().info(
        #     f"✅ WIDTH OK = {width_ok}, SHAPE OK = {shape_ratio >= SHAPE_THRESH}"
        # )
        # ---- LOGGING (1Hz) ----
        if not hasattr(self, "last_log_t"):
            self.last_log_t = 0.0

        now = time.time()
        if now - self.last_log_t >= 1.0:
            self.last_log_t = now

            self.get_logger().info(
                f"🧩 Contour coverage: {shape_ratio * 100:.2f} %"
            )

            for y, cur_w, ref_w, ratio in width_details:
                self.get_logger().info(
                    f"📏 Slice y={y:4d} : "
                    f"{cur_w:4d}px / {ref_w:4d}px → {ratio * 100:.2f} %"
                )

            self.get_logger().info(
                f"✅ WIDTH OK = {width_ok}, SHAPE OK = {shape_ratio >= SHAPE_THRESH}"
            )

        # ---- STOP ----
        if (
            shape_ratio >= SHAPE_THRESH
            and width_ok
            and not self.stop_published
        ):
            self.pub_stop.publish(Bool(data=True))
            self.stop_published = True
            self.get_logger().warn("🛑 STOP published!")

# ==============================
# Main
# ==============================
def main(args=None):
    rclpy.init(args=args)
    node = BeadStopJudgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
