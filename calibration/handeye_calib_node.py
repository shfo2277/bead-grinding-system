#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import tf2_ros
from tf2_ros.transform_listener import TransformListener

import numpy as np
import cv2
from cv2 import aruco


class HandEyeCalibNode(Node):
    def __init__(self):
        super().__init__('handeye_calib_node')

        # =============================
        # Frame 설정
        # =============================
        self.base_frame = 'base_link'
        self.tcp_frame = 'tool0'
        self.camera_frame = 'camera_color_optical_frame'

        # =============================
        # ArUco GridBoard
        # =============================
        self.marker_length = 0.038
        self.marker_separation = 0.0095
        self.rows = 3
        self.cols = 3

        self.num_samples_target = 35

        # =============================
        # TF
        # =============================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # =============================
        # Camera
        # =============================
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_rect_raw',
            self.image_callback,
            10
        )

        self.cinfo_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.cinfo_callback,
            10
        )

        self.camera_matrix = None
        self.dist_coeffs = None

        # =============================
        # ArUco
        # =============================
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
        self.aruco_params = aruco.DetectorParameters()
        self.board = aruco.GridBoard(
            (self.cols, self.rows),
            self.marker_length,
            self.marker_separation,
            self.aruco_dict
        )

        # =============================
        # 샘플 저장 리스트
        # =============================
        self.R_base_tool0_list = []
        self.t_base_tool0_list = []
        self.R_cam_target_list = []
        self.t_cam_target_list = []
        self.reproj_rms_list = []
        self.num_markers_list = []

        # =============================
        # 파일 저장 위치
        # =============================
        self.save_path = "/home/ho/BEADtrain/Calibration/handeye_samples_latest.npz"
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        self.get_logger().info("✅ HandEye calibration node started")
        self.get_logger().info(f"📁 Sample file: {self.save_path}")

    # =============================
    # CameraInfo
    # =============================
    def cinfo_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("📷 Camera info received")

    # =============================
    # Image callback
    # =============================
    def image_callback(self, msg: Image):

        if self.camera_matrix is None:
            return

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        if ids is None:
            cv2.imshow("Calibration", img)
            cv2.waitKey(1)
            return

        retval, rvec, tvec = aruco.estimatePoseBoard(
            corners, ids, self.board,
            self.camera_matrix, self.dist_coeffs,
            None, None
        )

        if retval <= 0:
            cv2.imshow("Calibration", img)
            cv2.waitKey(1)
            return

        # cam -> target
        R_cam_target, _ = cv2.Rodrigues(rvec)
        t_cam_target = tvec.reshape(3, 1)

        # reprojection error 계산
        num_markers = len(ids)
        obj_pts, img_pts = self.board.matchImagePoints(corners, ids)
        projected, _ = cv2.projectPoints(
            obj_pts, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        reproj_rms = np.sqrt(np.mean(
            np.sum((projected.reshape(-1, 2) - img_pts.reshape(-1, 2))**2, axis=1)
        ))

        # TF base -> tool0
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tcp_frame,
                Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        q = trans.transform.rotation
        t = trans.transform.translation

        T_base_tool0 = self.quaternion_to_matrix([q.x, q.y, q.z, q.w])
        T_base_tool0[0, 3] = t.x
        T_base_tool0[1, 3] = t.y
        T_base_tool0[2, 3] = t.z

        R_base_tool0 = T_base_tool0[0:3, 0:3]
        t_base_tool0 = T_base_tool0[0:3, 3].reshape(3, 1)

        # UI
        n = len(self.R_base_tool0_list)
        color = (0, 255, 0) if reproj_rms < 1.0 else (0, 165, 255) if reproj_rms < 3.0 else (0, 0, 255)
        cv2.putText(img, f"Markers: {num_markers}  Reproj: {reproj_rms:.2f}px", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img, f"Samples: {n}/{self.num_samples_target}  Press 's' to save", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Calibration", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            self.save_sample(R_base_tool0, t_base_tool0,
                             R_cam_target, t_cam_target,
                             reproj_rms, num_markers)

    # =============================
    # 샘플 저장
    # =============================
    def save_sample(self, Rb, tb, Rc, tc, reproj_rms, num_markers):

        self.R_base_tool0_list.append(Rb)
        self.t_base_tool0_list.append(tb)
        self.R_cam_target_list.append(Rc)
        self.t_cam_target_list.append(tc)
        self.reproj_rms_list.append(reproj_rms)
        self.num_markers_list.append(num_markers)

        n = len(self.R_base_tool0_list)

        self.get_logger().info(
            f"Sample {n}/{self.num_samples_target} saved "
            f"(reproj={reproj_rms:.2f}px, markers={num_markers})"
        )

        # 파일 저장
        self.save_samples_to_file()

        if n >= self.num_samples_target:
            self.compute_handeye()

    # =============================
    # npz 파일 저장
    # =============================
    def save_samples_to_file(self):

        np.savez(
            self.save_path,
            R_base_tool0=np.array(self.R_base_tool0_list),
            t_base_tool0=np.array(self.t_base_tool0_list),
            R_cam_target=np.array(self.R_cam_target_list),
            t_cam_target=np.array(self.t_cam_target_list),
            reproj_rms_px=np.array(self.reproj_rms_list),
            num_markers=np.array(self.num_markers_list)
        )

        self.get_logger().info("💾 Samples written to npz")

    # =============================
    # HandEye
    # =============================
    def compute_handeye(self):

        Rg = self.R_base_tool0_list
        tg = self.t_base_tool0_list
        Rt = self.R_cam_target_list
        tt = self.t_cam_target_list

        R, t = cv2.calibrateHandEye(
            Rg, tg, Rt, tt,
            method=cv2.CALIB_HAND_EYE_PARK
        )

        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = t.flatten()

        self.get_logger().info("🎉 HandEye result:")
        self.get_logger().info(str(T))

    # =============================
    # Quaternion → Matrix
    # =============================
    def quaternion_to_matrix(self, q):
        x, y, z, w = q
        n = x*x + y*y + z*z + w*w
        if n < 1e-8:
            return np.eye(4)
        s = 2.0 / n

        xx, yy, zz = x*x*s, y*y*s, z*z*s
        xy, xz, yz = x*y*s, x*z*s, y*z*s
        wx, wy, wz = w*x*s, w*y*s, w*z*s

        R = np.array([
            [1-(yy+zz), xy-wz, xz+wy],
            [xy+wz, 1-(xx+zz), yz-wx],
            [xz-wy, yz+wx, 1-(xx+yy)]
        ])

        T = np.eye(4)
        T[0:3, 0:3] = R
        return T


def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCalibNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()