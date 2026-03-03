#!/usr/bin/env python3
# 고정된 스캔 위치 로봇의 조인트를 값으로 주고 > 비드 경로 생성
import math
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

import tf2_ros
from tf2_ros import StaticTransformBroadcaster
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import TransformStamped
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker


class BeadPoseNode(Node):
    def __init__(self):
        super().__init__('bead_pose_node')

        self.bridge = CvBridge()

        # ===== 카메라 Intrinsic ===== 이거 필요한가..?
        self.fx = self.fy = self.cx = self.cy = None
        self.camera_frame = None

        # ===== 최신 Depth 이미지 =====
        self.depth_image = None
        self.depth_header = None

        # RealSense depth scale (16UC1 -> meters)
        self.depth_scale = 0.001  # 필요하면 수정

        # ===== 경로 계산용 고정 변환 행렬 (TF lookup 안 함) =====
        # world → tool0 (티치펜던트 Flange 좌표, Euler XYZ intrinsic) 팬턴트는 
        # 스캔 위치 joints: (80.14, 43.01, 80.39, -16.98, -30.36, 13.99)
        T_world_tool0 = self._make_tf_matrix(
            tx=0.213960, ty=1.325980, tz=0.251270,
            qx=-0.512170, qy=0.522979, qz=0.475362, qw=0.488062
        )
        # tool0 → camera_color_optical_frame
        # 35 samples, 2026-02-26, TSAI method, std<1mm verified
        T_tool0_cam = self._make_tf_matrix(
            tx=0.14715761, ty=0.01186576, tz=0.18413594,
            qx=0.5099066234780589, qy=-0.4950305962439862,
            qz=0.5061395424786499, qw=-0.4886335105731407
        )
        # # old calibration (before 2026-02-26)
        # T_tool0_cam = self._make_tf_matrix(
        #     tx=0.14673302, ty=0.01334886, tz=0.18633142,
        #     qx=-0.506893, qy=0.494631, qz=-0.510185, qw=0.487965
        # )

        # world → camera_color_optical_frame (합성)
        self.T_world_cam = T_world_tool0 @ T_tool0_cam
        
        ## 카메라 디버링용 출력 그니까 캘리 잘 됐는지 디버깅
        #self.get_logger().info('Direct TF matrix computed: world -> camera_color_optical_frame')

        # ===== QoS Profiles =====
        # 카메라 토픽은 RELIABLE로 publish됨
        camera_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        # bead/mask 는 BEST_EFFORT
        sensor_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        # ===== Subscribers =====
        self.sub_caminfo = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            # '/camera/camera/depth/camera_info',
            self.caminfo_callback,
            qos_profile=camera_qos,
        )

        self.sub_depth = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            # '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            qos_profile=camera_qos,
        )

        self.sub_mask = self.create_subscription(
            Image,
            '/bead/mask',
            self.mask_callback,
            qos_profile=sensor_qos,
        )

        # ===== Publishers =====
        self.start_pose_pub = self.create_publisher(
            PoseStamped,
            '/bead/grind_start_pose_base',
            10
        )

        self.end_pose_pub = self.create_publisher(
            PoseStamped,
            '/bead/grind_end_pose_base',
            10
        )

        self.start_marker_pub = self.create_publisher(
            Marker,
            '/bead/start_marker',
            10
        )
        self.end_marker_pub = self.create_publisher(
            Marker,
            '/bead/end_marker',
            10
        )

        self.path_pub = self.create_publisher(
            Path,
            '/bead/grind_path_base',
            10
        )

        # 비드 포인트 클라우드 (RViz 용)
        self.bead_points_pub = self.create_publisher(
            PointCloud2,
            '/bead/points_base',
            10
        )

        # ✅ 한 번 경로 계산하면 잠그는 플래그
        self.path_locked = False

        # ✅ 외부에서 경로 리셋하고 싶을 때 사용할 서비스
        self.reset_srv = self.create_service(
            Trigger,
            'reset_bead_path',
            self.reset_bead_path_cb
        )

        self.get_logger().info('✅ BeadPoseNode started.')

    @staticmethod
    def _make_tf_matrix(tx, ty, tz, qx, qy, qz, qw):
        """쿼터니언 + 이동 → 4x4 변환 행렬"""
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        return T

    # ---------------- reset 서비스 콜백 ----------------
    def reset_bead_path_cb(self, request, response):
        self.path_locked = False
        self.get_logger().info("🔁 reset_bead_path called. path_locked=False")
        response.success = True
        response.message = "Bead grind path reset. Will recompute on next valid mask."
        return response

    # ---------------- CameraInfo 콜백 ----------------
    def caminfo_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

        # 그냥 카메라가 주는 frame_id 그대로 사용
        self.camera_frame = msg.header.frame_id
        self.get_logger().info(f"CameraInfo from frame: {self.camera_frame}")

    # ---------------- Depth 콜백 ----------------
    def depth_callback(self, msg: Image):
        # 뎁스 디버깅 
        # self.get_logger().info(
        #     f"[depth_callback] got depth: {msg.width}x{msg.height}, "
        #     f"encoding={msg.encoding}, frame={msg.header.frame_id}"
        # )
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        self.depth_image = depth
        self.depth_header = msg.header

    # ---------------- Mask 콜백 (핵심) ----------------
    def mask_callback(self, msg: Image):
        self.get_logger().info("mask_callback called")

        # # ✅ 이미 경로를 한 번 만들었다면 이후 마스크는 무시
        # if self.path_locked:
        #     self.get_logger().info("path_locked=True, ignore new mask.")
        #     return

        if self.path_locked:
            self.get_logger().info("path_locked=True -> cloud only mode")
            only_cloud_mode = True      # ← 추가
        else:
            only_cloud_mode = False     # ← 추가

        # 아직 depth나 camera info 없으면 skip
        if self.depth_image is None:
            self.get_logger().warn("No depth_image yet, skip.")
            return
        if self.fx is None:
            self.get_logger().warn("No camera intrinsics yet (fx is None), skip.")
            return

        # 1) 마스크, depth 해상도 맞추기
        mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

        dh, dw = self.depth_image.shape[:2]
        mh, mw = mask.shape[:2]
        self.get_logger().info(f"depth shape={dh}x{dw}, mask shape={mh}x{mw}")

        if (mh, mw) != (dh, dw):
            mask_resized = cv2.resize(mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask

        depth = self.depth_image

        # 2) 비드 픽셀 좌표(u, v) 찾기
        ys, xs = np.where(mask_resized > 0)
        self.get_logger().info(f"non-zero mask pixels: {len(xs)}")

        if len(xs) == 0:
            self.get_logger().warn("Mask has no non-zero pixels, skip.")
            return

        # 3) 해당 픽셀들의 depth 값(m) 얻기 + 이상값 필터링
        depths = depth[ys, xs].astype(np.float32) * self.depth_scale
        valid = depths > 0.0
        # depth 이상값 필터 (median ± 3*MAD)
        if np.any(valid):
            d_valid = depths[valid]
            d_median = np.median(d_valid)
            d_mad = np.median(np.abs(d_valid - d_median))
            if d_mad > 0:
                depth_ok = np.abs(depths - d_median) < 3 * d_mad
                valid = valid & depth_ok
                self.get_logger().info(f"depth filter: median={d_median:.4f}m, MAD={d_mad:.4f}m, kept={np.count_nonzero(valid)}")
        self.get_logger().info(f"valid depth pixels: {np.count_nonzero(valid)}")

        if not np.any(valid):
            self.get_logger().warn("All depth values are 0, skip.")
            return

        xs = xs[valid].astype(np.float32)
        ys = ys[valid].astype(np.float32)
        depths = depths[valid]

        # 너무 많으면 subsample
        MAX_POINTS = 5000
        if len(xs) > MAX_POINTS:
            step = len(xs) // MAX_POINTS
            xs = xs[::step]
            ys = ys[::step]
            depths = depths[::step]

        # 4) (u, v, Z) -> 카메라 좌표계 (X, Y, Z)
        Z = depths
        X = (xs - self.cx) * Z / self.fx
        Y = (ys - self.cy) * Z / self.fy

        pts_cam = np.stack([X, Y, Z], axis=1)  # (N,3)

        # 5) 직접 행렬 변환으로 Staubli world 좌표계로 변환
        ones = np.ones((pts_cam.shape[0], 1))
        pts_cam_h = np.hstack([pts_cam, ones])  # (N,4)
        pts_world = (self.T_world_cam @ pts_cam_h.T).T[:, :3]  # (N,3)
        self.get_logger().info(f"World TF applied: {pts_cam.shape[0]} points transformed")

        # 포인트가 너무 적으면 중단
        if pts_world.shape[0] < 10:
            self.get_logger().warn('Not enough 3D points after TF transform.')
            return

        # ======================
        # 6) PCA 기반으로 비드 방향 및 양 끝점(start/end) 계산
        # ======================
        xy = pts_world[:, :2]  # (N,2)

        if xy.shape[0] < 10:
            self.get_logger().warn('Not enough points for path generation.')
            return

        xy_mean = xy.mean(axis=0, keepdims=True)
        xy_centered = xy - xy_mean

        cov = np.cov(xy_centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        idx_dir = int(np.argmax(eigvals))
        direction = eigvecs[:, idx_dir]
        direction = direction / np.linalg.norm(direction)

        # 각 점을 PCA 직선 상의 1D 파라미터 t로 투영
        t = xy_centered @ direction
        t_min, t_max = float(t.min()), float(t.max())

        # ✅ 비드의 "양 끝" 인덱스 (start/end)
        idx_min = int(np.argmin(t))
        idx_max = int(np.argmax(t))

        start_pt = pts_world[idx_min]
        end_pt = pts_world[idx_max]

        z_mean = float(np.median(pts_world[:, 2]))

        # # ======================
        # # 7) 시작/끝 PoseStamped 퍼블리시
        # # ======================
        # header_base = self.get_clock().now().to_msg()

        # start_pose = PoseStamped()
        # start_pose.header.stamp = header_base
        # start_pose.header.frame_id = 'world'
        # start_pose.pose.position.x = float(start_pt[0])
        # start_pose.pose.position.y = float(start_pt[1])
        # start_pose.pose.position.z = z_mean
        # start_pose.pose.orientation.w = 1.0

        # end_pose = PoseStamped()
        # end_pose.header.stamp = header_base
        # end_pose.header.frame_id = 'world'
        # end_pose.pose.position.x = float(end_pt[0])
        # end_pose.pose.position.y = float(end_pt[1])
        # end_pose.pose.position.z = z_mean
        # end_pose.pose.orientation.w = 1.0

        # self.start_pose_pub.publish(start_pose)
        # self.end_pose_pub.publish(end_pose)

        # # ======================
        # # RViz Marker 시각화
        # # ======================

        # # Start marker
        # start_marker = Marker()
        # start_marker.header.frame_id = 'world'
        # start_marker.header.stamp = self.get_clock().now().to_msg()
        # start_marker.ns = 'bead_start'
        # start_marker.id = 0
        # start_marker.type = Marker.SPHERE
        # start_marker.action = Marker.ADD
        # start_marker.pose = start_pose.pose
        # start_marker.scale.x = 0.03
        # start_marker.scale.y = 0.03
        # start_marker.scale.z = 0.03
        # start_marker.color.r = 1.0
        # start_marker.color.g = 0.0
        # start_marker.color.b = 0.0
        # start_marker.color.a = 1.0
        # self.start_marker_pub.publish(start_marker)

        # # End marker
        # end_marker = Marker()
        # end_marker.header.frame_id = 'world'
        # end_marker.header.stamp = start_marker.header.stamp
        # end_marker.ns = 'bead_end'
        # end_marker.id = 1
        # end_marker.type = Marker.SPHERE
        # end_marker.action = Marker.ADD
        # end_marker.pose = end_pose.pose
        # end_marker.scale.x = 0.03
        # end_marker.scale.y = 0.03
        # end_marker.scale.z = 0.03
        # end_marker.color.r = 0.0
        # end_marker.color.g = 0.0
        # end_marker.color.b = 1.0
        # end_marker.color.a = 1.0
        # self.end_marker_pub.publish(end_marker)

        # ======================
        # 8) 비드 경로(Path) 계산 (PCA 직선 따라 샘플링)
        # ======================
        NUM_WAYPOINTS = 30
        ts = np.linspace(t_min, t_max, NUM_WAYPOINTS)

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'world'

        for ti in ts:
            xy_i = xy_mean[0] + ti * direction
            pose_st = PoseStamped()
            pose_st.header = path_msg.header
            pose_st.pose.position.x = float(xy_i[0])
            pose_st.pose.position.y = float(xy_i[1])
            pose_st.pose.position.z = z_mean
            pose_st.pose.orientation.w = 1.0
            path_msg.poses.append(pose_st)

        self.path_pub.publish(path_msg)

        # ======================
        # 9) RViz 포인트클라우드 시각화
        # ======================
        header = self.depth_header
        header.frame_id = 'world'
        cloud_msg = pc2.create_cloud_xyz32(header, pts_world.tolist())
        self.bead_points_pub.publish(cloud_msg)

        self.get_logger().info(
            f"📌 start=({start_pt[0]:.3f}, {start_pt[1]:.3f}, {start_pt[2]:.3f}), "
            f"end=({end_pt[0]:.3f}, {end_pt[1]:.3f}, {end_pt[2]:.3f}), "
            f"path points={len(path_msg.poses)}, cloud points={len(pts_world)}"
        )

        # ✅ 첫 경로 계산 후 잠금
        self.path_locked = True
        self.last_path_msg = path_msg
        self.get_logger().info("✅ Bead grind path locked. Further masks will be ignored.")

        # 1초마다 경로 재퍼블리시 (브릿지/RViz 놓침 방지)
        if not hasattr(self, '_path_timer'):
            self._path_timer = self.create_timer(1.0, self._republish_path)

    def _republish_path(self):
        if hasattr(self, 'last_path_msg') and self.last_path_msg is not None:
            self.last_path_msg.header.stamp = self.get_clock().now().to_msg()
            self.path_pub.publish(self.last_path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BeadPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
