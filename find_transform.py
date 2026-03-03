#!/usr/bin/env python3
"""
플랜지 좌표 기반 T_world_base 변환 탐색

새 스캔 위치:
  joints: (80.14, 43.01, 80.39, -16.98, -30.36, 13.99)
  티치펜던트 Flange (world): X=213.96, Y=1325.98, Z=251.27, Rx=-94.11, Ry=1.35, Rz=89.94
"""
import numpy as np

# ===== 회전행렬 기본 함수 =====
def rot_x(a): c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rot_y(a): c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def rot_z(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def make_T(R, t):
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=t; return T

def inv_T(T):
    R=T[:3,:3]; t=T[:3,3]; Ti=np.eye(4); Ti[:3,:3]=R.T; Ti[:3,3]=-R.T@t; return Ti

def R_to_quat(R):
    tr = R[0,0]+R[1,1]+R[2,2]
    if tr>0:
        s=np.sqrt(tr+1)*2; return [(R[2,1]-R[1,2])/s,(R[0,2]-R[2,0])/s,(R[1,0]-R[0,1])/s,0.25*s]
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        s=np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; return [0.25*s,(R[0,1]+R[1,0])/s,(R[0,2]+R[2,0])/s,(R[2,1]-R[1,2])/s]
    elif R[1,1]>R[2,2]:
        s=np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; return [(R[0,1]+R[1,0])/s,0.25*s,(R[1,2]+R[2,1])/s,(R[0,2]-R[2,0])/s]
    else:
        s=np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; return [(R[0,2]+R[2,0])/s,(R[1,2]+R[2,1])/s,0.25*s,(R[1,0]-R[0,1])/s]

# ===== URDF FK =====
def urdf_fk(j_deg):
    j=np.radians(j_deg)
    T01=make_T(rot_z(j[0]),[0,0,0.55])
    T12=make_T(rot_y(j[1]),[0.15,0,0])
    T23=make_T(rot_y(j[2]),[0,0,0.825])
    T34=make_T(rot_z(j[3]),[0,0,0])
    T45=make_T(rot_y(j[4]),[0,0,0.625])
    T56=make_T(rot_z(j[5]),[0,0,0.11])
    return T01@T12@T23@T34@T45@T56

# ===== Euler -> R (12 conventions) =====
def euler_to_R(rx_d, ry_d, rz_d):
    rx,ry,rz = np.radians(rx_d), np.radians(ry_d), np.radians(rz_d)
    Rx,Ry,Rz = rot_x(rx), rot_y(ry), rot_z(rz)
    return {
        'XYZ_int (=ZYX_ext)': Rx@Ry@Rz,
        'XZY_int (=YZX_ext)': Rx@Rz@Ry,
        'YXZ_int (=ZXY_ext)': Ry@Rx@Rz,
        'YZX_int (=XZY_ext)': Ry@Rz@Rx,
        'ZXY_int (=YXZ_ext)': Rz@Rx@Ry,
        'ZYX_int (=XYZ_ext)': Rz@Ry@Rx,
    }

def rotation_angle(R):
    """회전행렬의 회전각도 (rad)"""
    val = (np.trace(R)-1)/2
    val = np.clip(val, -1, 1)
    return np.degrees(np.arccos(val))

# ===== 데이터 =====
new_joints = [80.14, 43.01, 80.39, -16.98, -30.36, 13.99]

# 티치펜던트 Flange (Staubli world frame, mm)
flange_pos_mm = np.array([213.96, 1325.98, 251.27])
flange_rx, flange_ry, flange_rz = -94.11, 1.35, 89.94

# ===== Step 1: URDF FK =====
T_base_flange = urdf_fk(new_joints)
urdf_pos_mm = T_base_flange[:3,3] * 1000
R_urdf = T_base_flange[:3,:3]

print("=" * 70)
print("Step 1: URDF FK vs 티치펜던트 Flange 위치 비교")
print("=" * 70)
print(f"URDF FK  (base_link, mm): ({urdf_pos_mm[0]:.2f}, {urdf_pos_mm[1]:.2f}, {urdf_pos_mm[2]:.2f})")
print(f"Pendant  (world,    mm): ({flange_pos_mm[0]:.2f}, {flange_pos_mm[1]:.2f}, {flange_pos_mm[2]:.2f})")
pos_diff = flange_pos_mm - urdf_pos_mm
print(f"위치 차이 (world-base):   ({pos_diff[0]:.2f}, {pos_diff[1]:.2f}, {pos_diff[2]:.2f}) mm")
print(f"위치 차이 크기: {np.linalg.norm(pos_diff):.2f} mm")
print()

# ===== Step 2: Euler 컨벤션 탐색 =====
print("=" * 70)
print("Step 2: Euler 컨벤션별 T_world_base 회전각도")
print("(작을수록 world ≈ base_link)")
print("=" * 70)

euler_Rs = euler_to_R(flange_rx, flange_ry, flange_rz)
results = []

for name, R_pendant in euler_Rs.items():
    # T_world_base = T_world_flange @ inv(T_base_flange)
    # R_world_base = R_pendant @ R_urdf^T
    R_world_base = R_pendant @ R_urdf.T
    angle = rotation_angle(R_world_base)
    t_world_base = flange_pos_mm/1000 - R_world_base @ (urdf_pos_mm/1000)
    results.append((name, angle, R_world_base, t_world_base, R_pendant))
    print(f"  {name:30s}: 회전각={angle:6.2f}°, t=({t_world_base[0]*1000:7.2f}, {t_world_base[1]*1000:7.2f}, {t_world_base[2]*1000:7.2f}) mm")

results.sort(key=lambda x: x[1])
print()

# ===== Step 3: 가장 좋은 결과 =====
print("=" * 70)
print("Step 3: 최적 컨벤션 (회전각 최소)")
print("=" * 70)
best = results[0]
print(f"컨벤션: {best[0]}")
print(f"T_world_base 회전각: {best[1]:.2f}°")
print(f"T_world_base 이동: ({best[3][0]*1000:.2f}, {best[3][1]*1000:.2f}, {best[3][2]*1000:.2f}) mm")
print(f"R_world_base:\n{best[2]}")
print()

# ===== Step 4: 검증 - 이전 데이터로 =====
print("=" * 70)
print("Step 4: 검증 - 이전 bead 좌표 변환")
print("=" * 70)

T_world_base = make_T(best[2], best[3])

# 이전 bead 좌표 (base_link, mm -> m)
bead_base_start = np.array([203.9, 1533.0, 490.8]) / 1000
bead_base_end = np.array([217.4, 1409.6, 490.8]) / 1000

ws = (T_world_base @ np.append(bead_base_start, 1))[:3] * 1000
we = (T_world_base @ np.append(bead_base_end, 1))[:3] * 1000

print(f"변환 결과 Start: ({ws[0]:.1f}, {ws[1]:.1f}, {ws[2]:.1f})")
print(f"변환 결과 End:   ({we[0]:.1f}, {we[1]:.1f}, {we[2]:.1f})")
print(f"기대값   Start:  (200.0, 1555.0, -3.0)")
print(f"기대값   End:    (200.0, 1380.0, -3.0)")
print(f"Start 에러: {np.linalg.norm(ws - np.array([200, 1555, -3])):.1f} mm")
print(f"End 에러:   {np.linalg.norm(we - np.array([200, 1380, -3])):.1f} mm")
print()

# ===== Step 5: 새 스캔 위치의 T_world_flange =====
print("=" * 70)
print("Step 5: 새 스캔 위치의 T_world_flange (쿼터니언)")
print("=" * 70)

R_best = best[4]  # R_world_flange (pendant Euler)
t_best = flange_pos_mm / 1000  # m

qx, qy, qz, qw = R_to_quat(R_best)
print(f"T_world_flange:")
print(f"  tx={t_best[0]:.6f}, ty={t_best[1]:.6f}, tz={t_best[2]:.6f}")
print(f"  qx={qx:.6f}, qy={qy:.6f}, qz={qz:.6f}, qw={qw:.6f}")
print()

# ===== Step 6: T_world_cam 계산 =====
print("=" * 70)
print("Step 6: bead_pose_node.py에 넣을 T_world_cam 계산")
print("=" * 70)

T_world_flange = make_T(R_best, t_best)

# T_tool0_cam (기존 하드코딩 값)
cam_qx, cam_qy, cam_qz, cam_qw = -0.510185, 0.494631, -0.506893, 0.487965
R_cam = np.array([
    [1-2*(cam_qy**2+cam_qz**2), 2*(cam_qx*cam_qy-cam_qz*cam_qw), 2*(cam_qx*cam_qz+cam_qy*cam_qw)],
    [2*(cam_qx*cam_qy+cam_qz*cam_qw), 1-2*(cam_qx**2+cam_qz**2), 2*(cam_qy*cam_qz-cam_qx*cam_qw)],
    [2*(cam_qx*cam_qz-cam_qy*cam_qw), 2*(cam_qy*cam_qz+cam_qx*cam_qw), 1-2*(cam_qx**2+cam_qy**2)],
])
T_tool0_cam = make_T(R_cam, [0.14673302, 0.01334886, 0.18633142])

T_world_cam = T_world_flange @ T_tool0_cam

print("T_world_cam (4x4):")
print(T_world_cam)
print()

# 쿼터니언 변환
R_wc = T_world_cam[:3,:3]
t_wc = T_world_cam[:3,3]
qx_wc, qy_wc, qz_wc, qw_wc = R_to_quat(R_wc)

print("bead_pose_node.py에 적용할 값:")
print(f"  T_world_tool0 = self._make_tf_matrix(")
print(f"      tx={t_best[0]:.6f}, ty={t_best[1]:.6f}, tz={t_best[2]:.6f},")
print(f"      qx={qx:.6f}, qy={qy:.6f}, qz={qz:.6f}, qw={qw:.6f}")
print(f"  )")
print()

# ===== 모든 컨벤션 top 3 상세 =====
print("=" * 70)
print("참고: 상위 3개 컨벤션 상세 결과")
print("=" * 70)
for name, angle, R_wb, t_wb, R_p in results[:3]:
    T_wb = make_T(R_wb, t_wb)
    ws2 = (T_wb @ np.append(bead_base_start,1))[:3]*1000
    we2 = (T_wb @ np.append(bead_base_end,1))[:3]*1000
    print(f"\n{name}:")
    print(f"  회전각={angle:.2f}°")
    print(f"  bead start: ({ws2[0]:.1f}, {ws2[1]:.1f}, {ws2[2]:.1f})")
    print(f"  bead end:   ({we2[0]:.1f}, {we2[1]:.1f}, {we2[2]:.1f})")
    print(f"  start err: {np.linalg.norm(ws2-np.array([200,1555,-3])):.1f}mm, end err: {np.linalg.norm(we2-np.array([200,1380,-3])):.1f}mm")
