#!/usr/bin/env python3
# reproj (재투영 오차)                                                                  
"""카메라가 ArUco 마커를 검출한 2D 픽셀 위치와, 계산된 3D 자세로부터 다시 이미지에 투영한
    위치 사이의 차이. 단위는 pixel. 낮을수록 ArUco 검출이 정확한 것. 보통 1px 이하면 양호."""

# R_base_tool0 (3x3 회전행렬)
#   로봇 base_link 기준으로 본 tool0(플랜지)의 방향. 로봇이 어떤 자세를 취하고 있는지를
#   나타냄. TF의 base_link → tool0 회전 부분.

#   t_base_tool0 (x, y, z)
#   로봇 base_link 기준으로 본 tool0(플랜지)의 위치. 단위 미터

#t_cam_target (x, y, z)
#   카메라 기준으로 본 ArUco 보드의 위치. 단위 미터. 예: (-71.53, +96.97, +412.48) mm =
#   보드가 카메라에서 Z방향(앞) 412mm 거리에 있음.

#--------------------------------------------------

"""handeye_samples_latest.npz 샘플 뷰어"""
import os
import numpy as np

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "handeye_samples_latest.npz")
data = np.load(path)

Rg = data["R_base_tool0"]
tg = data["t_base_tool0"]
Rt = data["R_cam_target"]
tt = data["t_cam_target"]
reproj = data["reproj_rms_px"]
markers = data["num_markers"]

N = len(Rg)
print(f"Total samples: {N}\n")

for i in range(N):
    print(f"{'='*60}")
    print(f"  Sample {i:02d}  |  reproj={reproj[i]:.3f}px  |  markers={int(markers[i])}")
    print(f"{'='*60}")

    print("  R_base_tool0:")
    for r in range(3):
        print(f"    [{Rg[i][r,0]:+.7f}  {Rg[i][r,1]:+.7f}  {Rg[i][r,2]:+.7f}]")
    t = tg[i].flatten()
    print(f"  t_base_tool0: [{t[0]:+.6f}, {t[1]:+.6f}, {t[2]:+.6f}] m")
    print(f"                ({t[0]*1000:+.2f}, {t[1]*1000:+.2f}, {t[2]*1000:+.2f}) mm")

    print()
    print("  R_cam_target:")
    for r in range(3):
        print(f"    [{Rt[i][r,0]:+.7f}  {Rt[i][r,1]:+.7f}  {Rt[i][r,2]:+.7f}]")
    t = tt[i].flatten()
    print(f"  t_cam_target: [{t[0]:+.6f}, {t[1]:+.6f}, {t[2]:+.6f}] m")
    print(f"                ({t[0]*1000:+.2f}, {t[1]*1000:+.2f}, {t[2]*1000:+.2f}) mm")
    print()
