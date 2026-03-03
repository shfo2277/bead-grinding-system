#!/usr/bin/env python3

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

# ==============================
# UNet Node
# ==============================
class BeadUNetNode(Node):
    def __init__(self):
        super().__init__("bead_unet_node")

        self.bridge = CvBridge()

        self.get_logger().info("✅ BeadUNetNode started")

        # ===== 상태 변수 =====
        self.contour_extracted = False
        self.saved_contour = None
        self.width_slices_y = []


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
        # 2) QoS (SensorData!)
        # -----------------------------
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
            self.qos_sensor,
        )

        self.pub_mask = self.create_publisher(
            Image,
            "/bead/mask",
            10,
        )

        self.pub_overlay = self.create_publisher(
            Image,
            "/bead/overlay",
            10,
        )

    # ==============================
    # Image callback
    # ==============================
    def image_callback(self, msg: Image):
        self.get_logger().info(
            f"📩 Got image: {msg.width}x{msg.height}, encoding={msg.encoding}"
        )

        # ROS Image → OpenCV BGR
        frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w, _ = frame_bgr.shape

        # BGR → RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # -----------------------------
        # Inference
        # -----------------------------
        img_tensor = transform(frame_rgb)
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.model(img_tensor)
            probs  = torch.sigmoid(logits)
            mask   = (probs > 0.5).float()

        mask_np = mask.squeeze().cpu().numpy().astype(np.uint8) * 255
        mask_resized = cv2.resize(
            mask_np, (w, h), interpolation=cv2.INTER_NEAREST
        )

        # -----------------------------
        # Publish mask (기존 유지)
        # -----------------------------
        mask_msg = self.bridge.cv2_to_imgmsg(mask_resized, encoding="mono8")
        mask_msg.header = msg.header
        self.pub_mask.publish(mask_msg)

        # -----------------------------
        # 최초 1회 contour + width 계산
        # -----------------------------
        if not self.contour_extracted:
            contours, _ = cv2.findContours(
                mask_resized,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE
            )

            if contours:
                bead_contour = max(contours, key=cv2.contourArea)
                self.saved_contour = bead_contour
                self.contour_extracted = True

                # contour 저장
                np.savetxt(
                    CONTOUR_CSV,
                    bead_contour.reshape(-1, 2),
                    delimiter=",",
                    header="x,y",
                    comments=""
                )

                # # width 계산
                # x, y, bw, bh = cv2.boundingRect(bead_contour)
                # ys = np.linspace(y, y + bh, NUM_WIDTH_SLICES + 2)[1:-1]

                # with open(WIDTH_CSV, "w") as f:
                #     f.write("slice_y,width_px\n")
                #     for y_line in ys:
                #         xs = [
                #             p[0][0] for p in bead_contour
                #             if abs(p[0][1] - y_line) < 2
                #         ]
                #         if len(xs) >= 2:
                #             width = max(xs) - min(xs)
                #             f.write(f"{int(y_line)},{int(width)}\n")

                # self.get_logger().info("📐 Bead contour & width saved")
                
                # width 계산 폭도 시각화 
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

        # -----------------------------
        # Overlay (기존 + contour 추가)
        # -----------------------------
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
                    (0, 255, 255),  # 노란색
                    2
                )


        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        overlay_msg.header = msg.header
        self.pub_overlay.publish(overlay_msg)

        self.get_logger().info("✅ Published /bead/mask & /bead/overlay")


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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
