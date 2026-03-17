#!/usr/bin/env python3
"""
Hand-Eye 캘리브레이션 현장 검증 노드

사용법:
1. 카메라 + 브릿지 + 로봇 TF 켜기
2. ArUco 보드를 고정된 곳에 놓기
3. 이 노드 실행
4. 로봇을 여러 자세로 움직이면서 's' 키로 기록
5. 보드가 항상 같은 base 좌표로 나오면 캘리 성공
   → std < 3mm 이면 양호
"""
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


class VerifyNode(Node):
    def __init__(self):
        super().__init__('verify_handeye')

        self.base_frame = 'base_link'
        self.tcp_frame = 'tool0'

        # ArUco GridBoard (handeye_calib_node와 동일)
        self.marker_length = 0.038
        self.marker_separation = 0.0095
        self.rows = 3
        self.cols = 3

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Camera
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, '/camera/camera/color/image_rect_raw',
            self.image_callback, 10
        )
        self.cinfo_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info',
            self.cinfo_callback, 10
        )
        self.camera_matrix = None
        self.dist_coeffs = None

        # ArUco
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
        self.aruco_params = aruco.DetectorParameters()
        self.board = aruco.GridBoard(
            (self.cols, self.rows),
            self.marker_length,
            self.marker_separation,
            self.aruco_dict
        )

        # Hand-Eye 결과 (cal.py TSAI 결과)
        self.T_tool0_cam = np.array([
            [-0.00246506, -0.01020528,  0.99994489,  0.14715761],
            [-0.99947224, -0.032364,   -0.00279419,  0.01186576],
            [ 0.03239073, -0.99942405, -0.01012011,  0.18413594],
            [ 0.,          0.,          0.,          1.        ]
        ])

        # 기록 저장
        self.recorded = []
        self.current_board_xyz = None

        self.get_logger().info("Verify node started")
        self.get_logger().info("보드를 고정하고 로봇을 움직이면서 's'로 기록하세요")

    def cinfo_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        if self.camera_matrix is None:
            return

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        if ids is None:
            self.current_board_xyz = None
            cv2.putText(img, "No markers detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            self.draw_stats(img)
            cv2.imshow("Verify", img)
            cv2.waitKey(1)
            return

        retval, rvec, tvec = aruco.estimatePoseBoard(
            corners, ids, self.board,
            self.camera_matrix, self.dist_coeffs,
            None, None
        )

        if retval <= 0:
            self.current_board_xyz = None
            self.draw_stats(img)
            cv2.imshow("Verify", img)
            cv2.waitKey(1)
            return

        # T_cam_target
        R_cam_target, _ = cv2.Rodrigues(rvec)
        T_cam_target = np.eye(4)
        T_cam_target[:3, :3] = R_cam_target
        T_cam_target[:3, 3] = tvec.flatten()

        # TF: T_base_tool0
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame, self.tcp_frame, Time()
            )
        except Exception:
            return

        q = trans.transform.rotation
        t = trans.transform.translation
        T_base_tool0 = self.quaternion_to_matrix([q.x, q.y, q.z, q.w])
        T_base_tool0[0, 3] = t.x
        T_base_tool0[1, 3] = t.y
        T_base_tool0[2, 3] = t.z

        # 보드 원점의 base_link 좌표
        T_base_target = T_base_tool0 @ self.T_tool0_cam @ T_cam_target
        self.current_board_xyz = T_base_target[:3, 3] * 1000.0  # mm

        # 화면 표시
        aruco.drawDetectedMarkers(img, corners, ids)
        cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeffs,
                          rvec, tvec, 0.05)

        b = self.current_board_xyz
        cv2.putText(img, f"Board in base: X={b[0]:.1f} Y={b[1]:.1f} Z={b[2]:.1f} mm",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"Markers: {len(ids)}  Press 's' to record",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        self.draw_stats(img)

        cv2.imshow("Verify", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and self.current_board_xyz is not None:
            self.recorded.append(self.current_board_xyz.copy())
            n = len(self.recorded)
            self.get_logger().info(
                f"Record {n}: X={b[0]:.1f} Y={b[1]:.1f} Z={b[2]:.1f} mm"
            )
            if n >= 2:
                arr = np.array(self.recorded)
                std = np.std(arr, axis=0)
                self.get_logger().info(
                    f"  std: X={std[0]:.2f} Y={std[1]:.2f} Z={std[2]:.2f} mm"
                )

    def draw_stats(self, img):
        n = len(self.recorded)
        if n == 0:
            cv2.putText(img, "No records yet", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            return

        arr = np.array(self.recorded)
        mean = np.mean(arr, axis=0)

        cv2.putText(img, f"Records: {n}  Mean: X={mean[0]:.1f} Y={mean[1]:.1f} Z={mean[2]:.1f}",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if n >= 2:
            std = np.std(arr, axis=0)
            total_std = np.linalg.norm(std)

            if total_std < 3.0:
                color = (0, 255, 0)
                grade = "GOOD"
            elif total_std < 5.0:
                color = (0, 165, 255)
                grade = "OK"
            else:
                color = (0, 0, 255)
                grade = "BAD"

            cv2.putText(img, f"Std: X={std[0]:.2f} Y={std[1]:.2f} Z={std[2]:.2f}  |std|={total_std:.2f}mm  [{grade}]",
                        (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 각 기록의 mean 대비 오차
        y = 175
        for i, rec in enumerate(self.recorded):
            diff = rec - mean
            err = np.linalg.norm(diff)
            cv2.putText(img, f"  #{i+1}: err={err:.1f}mm",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 22
            if y > img.shape[0] - 20:
                break

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
    node = VerifyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
