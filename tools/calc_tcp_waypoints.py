#!/usr/bin/env python3
"""
T_tool0_cam (캘리브레이션) + T_tool0_tcp (tTool) 를 이용해
TCP(그라인더) 기준 카메라 변환 및 웨이포인트를 계산하는 스크립트
"""
import numpy as np


def make_tf_matrix(tx, ty, tz, qx, qy, qz, qw):
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def rotmat_to_rpy(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        roll  = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll  = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = 0.0
    return roll, pitch, yaw


def print_matrix(name, T):
    print(f"\n{'='*50}")
    print(f" {name}")
    print(f"{'='*50}")
    print(T)
    tx, ty, tz = T[:3, 3]
    r, p, y = rotmat_to_rpy(T[:3, :3])
    print(f"  xyz = ({tx*1000:.1f}, {ty*1000:.1f}, {tz*1000:.1f}) [mm]")
    print(f"  rpy = ({np.degrees(r):.2f}, {np.degrees(p):.2f}, {np.degrees(y):.2f}) [deg]")


# ============================================================
# 1) T_tool0_cam: 캘리브레이션 결과 (플랜지 → 카메라)
# ============================================================
T_tool0_cam = make_tf_matrix(
    tx=0.068, ty=0.01334886, tz=0.18633142,
    qx=-0.506893, qy=0.494631, qz=-0.510185, qw=0.487965
)

# ============================================================
# 2) T_tool0_tcp: tTool (플랜지 → TCP/그라인더)
#    tTool: x=125mm, z=145.4mm, ry=-90°
# ============================================================
ry = np.radians(-90.0)
R_y = np.array([
    [ np.cos(ry), 0, np.sin(ry)],
    [ 0,          1, 0          ],
    [-np.sin(ry), 0, np.cos(ry)],
])
T_tool0_tcp = np.eye(4)
T_tool0_tcp[:3, :3] = R_y
T_tool0_tcp[:3, 3] = [0.125, 0.0, 0.1454]

# ============================================================
# 3) T_tcp_cam = T_tool0_tcp⁻¹ × T_tool0_cam
#    TCP(그라인더) 기준에서 카메라까지의 변환
# ============================================================
T_tcp_cam = np.linalg.inv(T_tool0_tcp) @ T_tool0_cam

# ============================================================
# 출력
# ============================================================
print_matrix("T_tool0_cam (플랜지 -> 카메라)", T_tool0_cam)
print_matrix("T_tool0_tcp (플랜지 -> TCP/그라인더)", T_tool0_tcp)
print_matrix("T_tcp_cam (TCP/그라인더 -> 카메라)", T_tcp_cam)

# ============================================================
# 4) 예시: 카메라 프레임에서 본 비드 포인트 → TCP 프레임으로 변환
# ============================================================
print(f"\n{'='*50}")
print(" 예시: 카메라 좌표 → TCP 좌표 변환")
print(f"{'='*50}")

# 카메라 광학 프레임 기준 예시 포인트 (비드가 카메라 앞 170mm)
example_points_cam = np.array([
    [0.0,   0.0,  0.170],   # 카메라 정면 170mm
    [0.01,  0.0,  0.170],   # 약간 오른쪽
    [-0.01, 0.0,  0.170],   # 약간 왼쪽
])

print(f"\n카메라 프레임 포인트 [mm]:")
for i, p in enumerate(example_points_cam):
    print(f"  P{i}: ({p[0]*1000:.1f}, {p[1]*1000:.1f}, {p[2]*1000:.1f})")

# 변환: P_tcp = T_tcp_cam × P_cam
ones = np.ones((example_points_cam.shape[0], 1))
pts_h = np.hstack([example_points_cam, ones])
pts_tcp = (T_tcp_cam @ pts_h.T).T[:, :3]

print(f"\nTCP(그라인더) 프레임 포인트 [mm]:")
for i, p in enumerate(pts_tcp):
    print(f"  P{i}: ({p[0]*1000:.1f}, {p[1]*1000:.1f}, {p[2]*1000:.1f})")

print(f"\n→ TCP 기준 비드까지 거리: x={pts_tcp[0,0]*1000:.1f}mm, y={pts_tcp[0,1]*1000:.1f}mm, z={pts_tcp[0,2]*1000:.1f}mm")
