#!/usr/bin/env python3
"""ArUco 보드를 인식하고 TF 체인으로 base_link 좌표를 출력하는 테스트 노드.
   base_link → tool0 → camera_color_optical_frame → aruco_board
   GUI 없이 터미널 출력만.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros

ARUCO_DICT = cv2.aruco.DICT_5X5_100
MARKER_SIZE = 0.038        # 38mm
MARKER_SEP  = 0.0095       # 9.5mm
BOARD_ROWS  = 3
BOARD_COLS  = 3


class TFTestAruco(Node):
    def __init__(self):
        super().__init__('tf_test_aruco')
        self.bridge = CvBridge()

        # ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.board = cv2.aruco.GridBoard(
            (BOARD_COLS, BOARD_ROWS), MARKER_SIZE, MARKER_SEP, self.aruco_dict)
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, cv2.aruco.DetectorParameters())

        # Camera intrinsics
        self.camera_matrix = None
        self.dist_coeffs = None

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info',
                                 self.info_cb, 10)
        self.create_subscription(Image, '/camera/camera/color/image_rect_raw',
                                 self.image_cb, 10)

        self.count = 0
        self.get_logger().info('TF Test ArUco 시작. ArUco 보드를 카메라에 보여주세요.')

    def info_cb(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info('Camera intrinsics 수신 완료')

    def image_cb(self, msg):
        if self.camera_matrix is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        corners, ids, _ = self.detector.detectMarkers(frame)

        if ids is None or len(ids) < 3:
            self.count += 1
            if self.count % 30 == 0:
                self.get_logger().info('보드 미검출... 카메라에 보드를 보여주세요')
            return

        # Board pose estimation
        obj_pts, img_pts = self.board.matchImagePoints(corners, ids)
        if obj_pts is None or len(obj_pts) < 4:
            return

        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts,
                                       self.camera_matrix, self.dist_coeffs)
        if not ok:
            return

        # T_cam_target (camera → board)
        R_cam, _ = cv2.Rodrigues(rvec)
        t_cam = tvec.flatten()

        # TF lookup: base_link → tool0, base_link → camera
        try:
            # base_link → tool0
            tf_tool0 = self.tf_buffer.lookup_transform(
                'base_link', 'tool0', rclpy.time.Time())
            t0 = tf_tool0.transform.translation
            flange = np.array([t0.x, t0.y, t0.z]) * 1000

            # base_link → camera_color_optical_frame
            tf_cam = self.tf_buffer.lookup_transform(
                'base_link', 'camera_color_optical_frame', rclpy.time.Time())
            t = tf_cam.transform.translation
            q = tf_cam.transform.rotation
            cam_pos = np.array([t.x, t.y, t.z]) * 1000
            T_base_cam = self.quat_to_matrix(t.x, t.y, t.z, q.x, q.y, q.z, q.w)
        except Exception as e:
            self.get_logger().warn(f'TF error: {e}')
            return

        # T_cam_target 4x4
        T_cam_target = np.eye(4)
        T_cam_target[:3, :3] = R_cam
        T_cam_target[:3, 3] = t_cam

        # base_link에서 본 보드 위치
        T_base_target = T_base_cam @ T_cam_target
        marker = T_base_target[:3, 3] * 1000  # mm

        detected_ids = sorted(ids.flatten().tolist())

        # 각 프레임의 RPY 계산
        def mat_to_rpy_deg(T):
            R = T[:3, :3]
            sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
            if sy > 1e-6:
                roll  = np.arctan2(R[2,1], R[2,2])
                pitch = np.arctan2(-R[2,0], sy)
                yaw   = np.arctan2(R[1,0], R[0,0])
            else:
                roll  = np.arctan2(-R[1,2], R[1,1])
                pitch = np.arctan2(-R[2,0], sy)
                yaw   = 0.0
            return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

        # T_base_tool0
        q0 = tf_tool0.transform.rotation
        T_base_tool0 = self.quat_to_matrix(t0.x, t0.y, t0.z, q0.x, q0.y, q0.z, q0.w)
        rpy_flange = mat_to_rpy_deg(T_base_tool0)

        rpy_cam = mat_to_rpy_deg(T_base_cam)
        rpy_marker = mat_to_rpy_deg(T_base_target)

        self.get_logger().info(
            f'\n'
            f'  [1] base_link:  X=0.0     Y=0.0     Z=0.0 mm     | RPY=(0.0, 0.0, 0.0) deg\n'
            f'  [2] flange:     X={flange[0]:.1f}  Y={flange[1]:.1f}  Z={flange[2]:.1f} mm  | RPY=({rpy_flange[0]:.1f}, {rpy_flange[1]:.1f}, {rpy_flange[2]:.1f}) deg\n'
            f'  [3] camera:     X={cam_pos[0]:.1f}  Y={cam_pos[1]:.1f}  Z={cam_pos[2]:.1f} mm  | RPY=({rpy_cam[0]:.1f}, {rpy_cam[1]:.1f}, {rpy_cam[2]:.1f}) deg\n'
            f'  [4] marker:     X={marker[0]:.1f}  Y={marker[1]:.1f}  Z={marker[2]:.1f} mm  (base_link) | RPY=({rpy_marker[0]:.1f}, {rpy_marker[1]:.1f}, {rpy_marker[2]:.1f}) deg\n'
            f'  [4] marker:     X={marker[0]:.1f}  Y={marker[1]:.1f}  Z={marker[2]-550:.1f} mm  (staubli)\n'
            f'  cam_dist: {t_cam[2]*1000:.1f}mm | markers={len(ids)} {detected_ids}')

    def quat_to_matrix(self, tx, ty, tz, qx, qy, qz, qw):
        T = np.eye(4)
        T[0, 0] = 1 - 2*(qy*qy + qz*qz)
        T[0, 1] = 2*(qx*qy - qz*qw)
        T[0, 2] = 2*(qx*qz + qy*qw)
        T[1, 0] = 2*(qx*qy + qz*qw)
        T[1, 1] = 1 - 2*(qx*qx + qz*qz)
        T[1, 2] = 2*(qy*qz - qx*qw)
        T[2, 0] = 2*(qx*qz - qy*qw)
        T[2, 1] = 2*(qy*qz + qx*qw)
        T[2, 2] = 1 - 2*(qx*qx + qy*qy)
        T[0, 3] = tx
        T[1, 3] = ty
        T[2, 3] = tz
        return T


def main():
    rclpy.init()
    node = TFTestAruco()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
