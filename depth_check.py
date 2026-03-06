#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
from cv_bridge import CvBridge

rclpy.init()
node = rclpy.create_node("depth_check")
br = CvBridge()

qos = QoSProfile(
    depth=5,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
)

def cb(msg):
    depth = br.imgmsg_to_cv2(msg, "16UC1")
    h, w = depth.shape
    cy, cx = h // 2, w // 2
    roi = depth[cy - 50 : cy + 50, cx - 50 : cx + 50].astype(np.float64)
    valid = roi[roi > 0]
    if len(valid) > 0:
        mean_raw = np.mean(valid)
        print(f"해상도: {w}x{h}")
        print(f"중앙 ROI raw mean: {mean_raw:.1f}")
        print(f"  scale=0.001  -> {mean_raw*0.001*100:.1f} cm  ({mean_raw*0.001*1000:.1f} mm)")
        print(f"  scale=0.0001 -> {mean_raw*0.0001*100:.2f} cm  ({mean_raw*0.0001*1000:.1f} mm)")
        print(f"std: {np.std(valid):.2f}, min/max: {np.min(valid):.0f}/{np.max(valid):.0f}")
    else:
        print("유효한 depth 없음")
    rclpy.shutdown()

node.create_subscription(Image, "/camera/camera/depth/image_rect_raw", cb, qos)
rclpy.spin(node)
