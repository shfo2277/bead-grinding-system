#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single ROS2 node: Model1(초기 비드) -> Model2(그라인딩 후 비드) 상태머신

Trigger topics:
- /scan_trigger   (std_msgs/Bool) : 스캔 위치 도달(초기) -> Model1 실행(1회 확정)
- /rescan_trigger (std_msgs/Bool) : 재스캔 위치 도달      -> Model2 실행(계속 판단)
- /reset_trigger  (std_msgs/Bool) : (선택) 상태 리셋

Publish:
- /bead/mask        (sensor_msgs/Image) : Model1 mask (1회)
- /bead/overlay     (sensor_msgs/Image) : overlay (Model1/2 공용, 이름 혼선 싫으면 분리 추천)
- /grinding/mask    (sensor_msgs/Image) : Model2 mask (연속)
- /stop             (std_msgs/Bool)     : STOP (1회)
"""

import cv2
import csv
import numpy as np
import torch
import torchvision.transforms as T
import segmentation_models_pytorch as smp

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge


# ==============================
# CONFIG (너 프로젝트에 맞게 수정)
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

# ---- Model paths ----
MODEL1_PATH = "/workspace/BEADtrain/modelresult/1125unet.pth"  # 초기(온전한 비드)
MODEL2_PATH = "/workspace/BEADtrain/modelresult/2026-02-03_09-45-47/unet_best.pth"  # 그라인딩 후 비드

# ---- CSV paths (Model1이 생성, Model2가 읽음) ----
CONTOUR_CSV = "/workspace/BEADtrain/bead_contour.csv"
WIDTH_CSV   = "/workspace/BEADtrain/bead_width.csv"
NUM_WIDTH_SLICES = 5

# ---- Model1 안정 프레임 선정 ----
WARMUP_SEC = 1.0
TIMEOUT_SEC = 8.0
NEEDED_GOOD = 3
MIN_MASK_AREA_PX = 150
MIN_CONTOUR_AREA_PX = 120

# ---- Model2 stop judge threshold ----
SHAPE_THRESH = 0.95
WIDTH_THRESH = 0.95


# ==============================
# Utility: CSV load/save & judge
# ==============================
def load_contour_csv(path):
    contour = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for x, y in reader:
            contour.append([[int(float(x)), int(float(y))]])
    return np.array(contour, dtype=np.int32)

def load_width_csv(path):
    widths = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for y, x_start, w in reader:
            widths.append((int(float(y)), int(float(x_start)), int(float(w))))
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
        if y < 0 or y >= h:
            all_ok = False
            results.append((y, 0, ref_w, 0.0))
            continue

        x_end = min(x_start + ref_w, w)
        if x_start < 0 or x_start >= w or x_end <= x_start:
            all_ok = False
            results.append((y, 0, ref_w, 0.0))
            continue

        roi = mask[y, x_start:x_end]
        cur_w = np.count_nonzero(roi)
        ratio = cur_w / ref_w if ref_w > 0 else 0.0

        if ratio < WIDTH_THRESH:
            all_ok = False

        results.append((y, cur_w, ref_w, ratio))

    return all_ok, results


# ==============================
# State machine
# ==============================
class ProcState:
    IDLE = 0
    MODEL1_WAIT_STABLE = 1
    MODEL2_RUN_JUDGE = 2
    DONE = 3


# ==============================
# Single Node
# ==============================
class BeadProcessNode(Node):
    def __init__(self):
        super().__init__("bead_process_node")
        self.bridge = CvBridge()

        # QoS (sensor)
        self.qos_sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # State
        self.state = ProcState.IDLE
        self.stop_published = False

        # ---- Model1 안정 프레임 선택 상태 ----
        self.start_time = None
        self.good_streak = 0

        self.best_msg = None
        self.best_frame_bgr = None
        self.best_mask_resized = None
        self.best_contour = None
        self.best_score = -1  # largest contour area

        # ---- Model1 결과 저장 ----
        self.contour_extracted = False
        self.saved_contour = None
        self.width_slices_y = []

        # ---- Model2 reference ----
        self.ref_contour = None
        self.ref_widths = None

        # Load models once
        self.model1 = self._load_unet(MODEL1_PATH, tag="Model1(초기)")
        self.model2 = self._load_unet(MODEL2_PATH, tag="Model2(그라인딩후)")

        # Subscribers
        self.sub_img = self.create_subscription(
            Image,
            "/camera/camera/color/image_rect_raw",
            self.image_callback,
            self.qos_sensor,
        )

        self.sub_scan = self.create_subscription(
            Bool, "/scan_trigger", self.scan_trigger_cb, 10
        )
        self.sub_rescan = self.create_subscription(
            Bool, "/rescan_trigger", self.rescan_trigger_cb, 10
        )
        self.sub_reset = self.create_subscription(
            Bool, "/reset_trigger", self.reset_trigger_cb, 10
        )

        # Publishers
        self.pub_mask_model1 = self.create_publisher(Image, "/bead/mask", 10)
        self.pub_overlay = self.create_publisher(Image, "/bead/overlay", 10)

        self.pub_mask_model2 = self.create_publisher(Image, "/grinding/mask", 10)
        self.pub_stop = self.create_publisher(Bool, "/stop", 10)

        self.get_logger().info("✅ BeadProcessNode ready. Waiting triggers...")
        self._log_state()

    # --------------------------
    # Model load helper
    # --------------------------
    def _load_unet(self, path, tag="UNet"):
        self.get_logger().info(f"🔄 Loading {tag}: {path}")
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        ).to(DEVICE)
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        self.get_logger().info(f"✅ {tag} loaded on {DEVICE}")
        return model

    # --------------------------
    # Trigger callbacks
    # --------------------------
    def scan_trigger_cb(self, msg: Bool):
        if not msg.data:
            return
        # 스캔 위치 도달 -> Model1 실행 시작
        self.get_logger().warn("📍 scan_trigger ON -> start MODEL1 stable capture")
        self._enter_model1()

    def rescan_trigger_cb(self, msg: Bool):
        if not msg.data:
            return
        # 재스캔 위치 도달 -> Model2 실행 시작
        self.get_logger().warn("📍 rescan_trigger ON -> start MODEL2 judge")
        self._enter_model2()

    def reset_trigger_cb(self, msg: Bool):
        if not msg.data:
            return
        self.get_logger().warn("♻ reset_trigger ON -> reset state machine")
        self._reset_all()

    # --------------------------
    # State transitions
    # --------------------------
    def _enter_model1(self):
        self.state = ProcState.MODEL1_WAIT_STABLE
        self.start_time = self.get_clock().now()
        self.good_streak = 0

        self.best_msg = None
        self.best_frame_bgr = None
        self.best_mask_resized = None
        self.best_contour = None
        self.best_score = -1

        # Model1 결과(컨투어/폭) 다시 만들고 싶으면 초기화
        self.contour_extracted = False
        self.saved_contour = None
        self.width_slices_y = []

        self._log_state()

    def _enter_model2(self):
        # Model2는 Model1에서 만든 CSV가 필요
        try:
            self.ref_contour = load_contour_csv(CONTOUR_CSV)
            self.ref_widths = load_width_csv(WIDTH_CSV)
            self.get_logger().info("📐 Reference contour & width loaded for MODEL2")
        except Exception as e:
            self.get_logger().error(f"❌ Cannot load reference CSVs. Run MODEL1 first. err={e}")
            return

        self.state = ProcState.MODEL2_RUN_JUDGE
        self.stop_published = False
        self._log_state()

    def _reset_all(self):
        self.state = ProcState.IDLE
        self.stop_published = False

        self.start_time = None
        self.good_streak = 0

        self.best_msg = None
        self.best_frame_bgr = None
        self.best_mask_resized = None
        self.best_contour = None
        self.best_score = -1

        self.contour_extracted = False
        self.saved_contour = None
        self.width_slices_y = []

        self.ref_contour = None
        self.ref_widths = None

        self._log_state()

    def _log_state(self):
        name = {
            ProcState.IDLE: "IDLE",
            ProcState.MODEL1_WAIT_STABLE: "MODEL1_WAIT_STABLE",
            ProcState.MODEL2_RUN_JUDGE: "MODEL2_RUN_JUDGE",
            ProcState.DONE: "DONE",
        }.get(self.state, f"Unknown({self.state})")
        self.get_logger().info(f"🧠 STATE => {name}")

    # --------------------------
    # Model1: contour/width save (기존 유지)
    # --------------------------
    def compute_and_save_contour_width_once(self, mask_resized: np.ndarray):
        if self.contour_extracted:
            return

        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return

        bead_contour = max(contours, key=cv2.contourArea)
        self.saved_contour = bead_contour
        self.contour_extracted = True

        # contour CSV
        np.savetxt(
            CONTOUR_CSV,
            bead_contour.reshape(-1, 2),
            delimiter=",",
            header="x,y",
            comments=""
        )

        # width CSV
        x, y, bw, bh = cv2.boundingRect(bead_contour)
        ys = np.linspace(y, y + bh, NUM_WIDTH_SLICES + 2)[1:-1]

        self.width_slices_y = []
        with open(WIDTH_CSV, "w") as f:
            f.write("slice_y,x_start,width_px\n")
            for y_line in ys:
                xs = [p[0][0] for p in bead_contour if abs(p[0][1] - y_line) < 2]
                if len(xs) >= 2:
                    x_min = min(xs)
                    x_max = max(xs)
                    width = x_max - x_min
                    f.write(f"{int(y_line)},{int(x_min)},{int(width)}\n")
                    self.width_slices_y.append(int(y_line))

        self.get_logger().info("📐 Bead contour & width saved (MODEL1, once)")

    # --------------------------
    # Overlay (기존 유지)
    # --------------------------
    def make_overlay(self, frame_bgr: np.ndarray, mask_resized: np.ndarray) -> np.ndarray:
        overlay = frame_bgr.copy()

        # red alpha overlay
        color = np.array([255, 0, 0], dtype=np.uint8)
        alpha = 0.5
        mask_bool = mask_resized > 0
        overlay[mask_bool] = (alpha * color + (1 - alpha) * overlay[mask_bool]).astype(np.uint8)

        # draw contour + width lines (Model1 기준 데이터)
        if self.saved_contour is not None:
            cv2.drawContours(overlay, [self.saved_contour], -1, (0, 255, 0), 2)
            for y_line in self.width_slices_y:
                xs = [p[0][0] for p in self.saved_contour if abs(p[0][1] - y_line) < 2]
                if len(xs) >= 2:
                    x_min, x_max = min(xs), max(xs)
                    cv2.line(overlay, (x_min, y_line), (x_max, y_line), (0, 255, 255), 2)

        return overlay

    # --------------------------
    # Image callback (state-based)
    # --------------------------
    def image_callback(self, msg: Image):
        # IDLE에서는 아무것도 안 함 (필요하면 화면용 publish만 하도록 바꿔도 됨)
        if self.state == ProcState.IDLE or self.state == ProcState.DONE:
            return

        frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ===== MODEL1 =====
        if self.state == ProcState.MODEL1_WAIT_STABLE:
            now = self.get_clock().now()
            elapsed = (now - self.start_time).nanoseconds * 1e-9 if self.start_time else 0.0

            # warmup
            if elapsed < WARMUP_SEC:
                return

            # inference (model1)
            img_tensor = transform(frame_rgb).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = self.model1(img_tensor)
                probs = torch.sigmoid(logits)
                mask = (probs > 0.5).float()

            mask_np = mask.squeeze().cpu().numpy().astype(np.uint8) * 255
            mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

            # quality eval
            mask_area = int(cv2.countNonZero(mask_resized))
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            best_contour = None
            largest_area = 0
            if contours:
                best_contour = max(contours, key=cv2.contourArea)
                largest_area = int(cv2.contourArea(best_contour))

            is_good = (mask_area >= MIN_MASK_AREA_PX) and (largest_area >= MIN_CONTOUR_AREA_PX)

            if is_good:
                self.good_streak += 1
                score = largest_area
                if score > self.best_score:
                    self.best_score = score
                    self.best_msg = msg
                    self.best_frame_bgr = frame_bgr.copy()
                    self.best_mask_resized = mask_resized.copy()
                    self.best_contour = best_contour

                self.get_logger().info(
                    f"✅ MODEL1 GOOD[{self.good_streak}/{NEEDED_GOOD}] "
                    f"mask_area={mask_area}, contour_area={largest_area}, best_score={self.best_score}"
                )
            else:
                self.get_logger().info(f"❌ MODEL1 BAD reset (mask_area={mask_area}, contour_area={largest_area})")
                self.good_streak = 0
                self.best_msg = None
                self.best_frame_bgr = None
                self.best_mask_resized = None
                self.best_contour = None
                self.best_score = -1

            # success (N streak)
            if self.good_streak >= NEEDED_GOOD:
                self._publish_model1_final_once()
                return

            # timeout
            if elapsed >= TIMEOUT_SEC and self.best_msg is not None:
                self.get_logger().warn("⏱ MODEL1 TIMEOUT: publishing best candidate anyway.")
                self._publish_model1_final_once()
                return

            return

        # ===== MODEL2 =====
        if self.state == ProcState.MODEL2_RUN_JUDGE:
            if self.ref_contour is None or self.ref_widths is None:
                # reference 없으면 모델2 수행 불가
                return

            # inference (model2)
            img_tensor = transform(frame_rgb).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                prob = torch.sigmoid(self.model2(img_tensor))[0, 0]

            mask = (prob > 0.5).cpu().numpy().astype(np.uint8) * 255
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # publish model2 mask
            mask_msg = self.bridge.cv2_to_imgmsg(mask, "mono8")
            mask_msg.header = msg.header
            self.pub_mask_model2.publish(mask_msg)

            # overlay: 현재 마스크 + reference contour/width 표시
            overlay = frame_bgr.copy()
            overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)

            cv2.drawContours(overlay, [self.ref_contour], -1, (0, 255, 0), 2)
            for y, x_start, ref_w in self.ref_widths:
                if 0 <= y < h:
                    x1 = x_start
                    x2 = min(x_start + ref_w, w - 1)
                    cv2.line(overlay, (x1, y), (x2, y), (0, 255, 255), 2)

            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, "bgr8")
            overlay_msg.header = msg.header
            self.pub_overlay.publish(overlay_msg)

            # STOP judge
            shape_ratio = shape_match_ratio(mask, self.ref_contour)
            width_ok, width_details = width_match_with_ratio(mask, self.ref_widths)

            self.get_logger().info(f"🧩 Contour coverage: {shape_ratio * 100:.2f} %")
            for y, cur_w, ref_w, ratio in width_details:
                self.get_logger().info(
                    f"📏 Slice y={y:4d} : {cur_w:4d}px / {ref_w:4d}px → {ratio * 100:.2f} %"
                )
            self.get_logger().info(
                f"✅ WIDTH OK = {width_ok}, SHAPE OK = {shape_ratio >= SHAPE_THRESH}"
            )

            if (shape_ratio >= SHAPE_THRESH) and width_ok and (not self.stop_published):
                self.pub_stop.publish(Bool(data=True))
                self.stop_published = True
                self.get_logger().warn("🛑 STOP published!")
                # 필요하면 DONE으로 전환
                self.state = ProcState.DONE
                self._log_state()

            return

    # --------------------------
    # Model1 final publish (딱 1회)
    # --------------------------
    def _publish_model1_final_once(self):
        if self.best_msg is None or self.best_frame_bgr is None or self.best_mask_resized is None:
            self.get_logger().error("❌ MODEL1: No best frame stored.")
            return

        # best contour를 saved_contour로 넘겨 overlay/폭선에 사용
        self.saved_contour = self.best_contour

        # contour/width save once
        self.compute_and_save_contour_width_once(self.best_mask_resized)

        # publish /bead/mask (1회)
        mask_msg = self.bridge.cv2_to_imgmsg(self.best_mask_resized, "mono8")
        mask_msg.header = self.best_msg.header
        self.pub_mask_model1.publish(mask_msg)

        # publish /bead/overlay (1회)
        overlay = self.make_overlay(self.best_frame_bgr, self.best_mask_resized)
        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, "bgr8")
        overlay_msg.header = self.best_msg.header
        self.pub_overlay.publish(overlay_msg)

        self.get_logger().info("✅ MODEL1 published ONCE + CSV saved. Now waiting for rescan_trigger.")
        # Model1 종료 -> 다시 IDLE로 두고 rescan_trigger를 기다리게
        self.state = ProcState.IDLE
        self._log_state()


def main(args=None):
    rclpy.init(args=args)
    node = BeadProcessNode()
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
