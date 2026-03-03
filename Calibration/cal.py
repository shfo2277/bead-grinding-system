#!/usr/bin/env python3
import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot


# ==============================
# Residual 계산
# ==============================
# T_base_target = T_base_tool0 @ X @ T_cam_target
# 타겟이 고정이므로 모든 샘플에서 T_base_target이 동일해야 함. 여기서 벗어나는 정도가 residual
# 각 샘플의 T_base_target과 중앙값(median)의 차이가 곧 오차.
# ==============================
#   reproj ≤ 3.0px  AND  markers ≥ 5
#   재투영 오차가 너무 크거나 마커가 너무 적은 샘플을 먼저 제거.

def residual_per_sample(Rg, tg, Rt, tt, R, t):
    X = np.eye(4)
    X[:3, :3] = R
    X[:3, 3] = t.flatten()

    # 각 샘플의 T_base_target 계산
    T_list = []
    for i in range(len(Rg)):
        A = np.eye(4)
        A[:3, :3] = Rg[i]
        A[:3, 3] = tg[i].flatten()

        B = np.eye(4)
        B[:3, :3] = Rt[i]
        B[:3, 3] = tt[i].flatten()

        T_base_target = A @ X @ B
        T_list.append(T_base_target)

    # 기준: translation 중앙값, rotation 평균
    positions = np.array([T[:3, 3] for T in T_list])
    ref_t = np.median(positions, axis=0)

    rotations = Rot.from_matrix([T[:3, :3] for T in T_list])
    ref_R = rotations.mean().as_matrix()

    T_ref = np.eye(4)
    T_ref[:3, :3] = ref_R
    T_ref[:3, 3] = ref_t

    # 각 샘플과 기준의 차이
    trans_mm = []
    rot_deg = []
    for T in T_list:
        delta = np.linalg.inv(T_ref) @ T

        trans = np.linalg.norm(delta[:3, 3]) * 1000.0

        c = np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)
        rot = np.degrees(np.arccos(c))

        trans_mm.append(trans)
        rot_deg.append(rot)

    return np.array(trans_mm), np.array(rot_deg)


# ==============================
# MAD 기반 아웃라이어 threshold
# ==============================
def mad_threshold(values, k=2.5):
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    # MAD가 0에 가까우면 (데이터가 매우 균일) 최소 fallback
    mad = max(mad, 1e-6)
    return med + k * 1.4826 * mad


# ==============================
# HandEye 계산
# ==============================
METHODS = {
    "TSAI":      cv2.CALIB_HAND_EYE_TSAI,
    "PARK":      cv2.CALIB_HAND_EYE_PARK,
    "HORAUD":    cv2.CALIB_HAND_EYE_HORAUD,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    "ANDREFF":   cv2.CALIB_HAND_EYE_ANDREFF,
}


def solve_handeye(Rg, tg, Rt, tt, mask, method=cv2.CALIB_HAND_EYE_PARK):
    R, t = cv2.calibrateHandEye(
        list(Rg[mask]),
        list(tg[mask]),
        list(Rt[mask]),
        list(tt[mask]),
        method=method
    )
    return R, t


# ==============================
# 메인
# ==============================
def main():

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "handeye_samples_latest.npz")
    data = np.load(path)

    Rg = data["R_base_tool0"]
    tg = data["t_base_tool0"]
    Rt = data["R_cam_target"]
    tt = data["t_cam_target"]

    if "reproj_rms_px" in data.files:
        reproj = data["reproj_rms_px"]
    else:
        print("reprojection 없음 -> 전체 사용")
        reproj = np.zeros(len(Rg))

    if "num_markers" in data.files:
        num_markers = data["num_markers"]
    else:
        print("num_markers 없음 -> 전체 사용")
        num_markers = np.full(len(Rg), 9)

    N = len(Rg)
    print(f"Loaded samples: {N}")

    print("\nPer-sample quality:")
    for i in range(N):
        print(f"  {i:02d}: reproj={reproj[i]:.2f}px, markers={num_markers[i]}")

    # ==========================================
    # STEP 1: reprojection + marker 품질 필터
    # ==========================================
    reproj_thresh = 3.0
    marker_thresh = 5
    keep = (reproj <= reproj_thresh) & (num_markers >= marker_thresh)

    print(f"\nSTEP 1 quality filter (reproj<={reproj_thresh}px, markers>={marker_thresh}): {np.sum(keep)}/{N}")

    if np.sum(keep) < 6:
        print("ERROR: 품질 필터 후 샘플 부족 (최소 6개 필요). threshold를 완화하세요.")
        return

    # ==========================================
    # STEP 2: 반복 MAD 필터
    # ==========================================
    # MAD 반복 필터 (통계 기반 threshold)

    # 고정 숫자가 아니라 데이터 분포에서 자동 계산:

    # threshold = median + 2.5 × 1.4826 × MAD

    # - MAD = Median Absolute Deviation (중앙값에서 각 값이 얼마나 떨어져 있나의 중앙값)
    # - 1.4826 = 정규분포 가정시 표준편차 변환 계수
    # - 2.5 = 몇 시그마까지 허용할지 (2.5σ) 

    MAX_ITER = 5
    MIN_SAMPLES = 10

    for iteration in range(MAX_ITER):
        n_before = np.sum(keep)

        R_cur, t_cur = solve_handeye(Rg, tg, Rt, tt, keep)
        trans_mm, rot_deg = residual_per_sample(Rg, tg, Rt, tt, R_cur, t_cur)

        # keep인 샘플들의 residual로 threshold 계산
        trans_thresh = mad_threshold(trans_mm[keep])
        rot_thresh = mad_threshold(rot_deg[keep])

        keep_new = keep & (trans_mm <= trans_thresh) & (rot_deg <= rot_thresh)

        n_after = np.sum(keep_new)
        removed = n_before - n_after

        print(f"\nIter {iteration+1}: "
              f"trans_thresh={trans_thresh:.2f}mm, rot_thresh={rot_thresh:.2f}deg, "
              f"removed={removed}, remaining={n_after}/{N}")

        if n_after < MIN_SAMPLES:
            print(f"  -> 샘플이 {MIN_SAMPLES}개 미만이 되므로 이번 라운드 제거 취소")
            break

        keep = keep_new

        if removed == 0:
            print("  -> 수렴 (더 이상 제거할 샘플 없음)")
            break

    # ==========================================
    # STEP 3: 필터 후 per-sample residual 출력
    # ==========================================
    R_filtered, t_filtered = solve_handeye(Rg, tg, Rt, tt, keep)
    trans_final, rot_final = residual_per_sample(Rg, tg, Rt, tt, R_filtered, t_filtered)

    print(f"\nResidual per sample (after filtering, {np.sum(keep)} samples):")
    for i in range(N):
        tag = "  OK" if keep[i] else "  XX"
        print(f"  {i:02d}: {trans_final[i]:6.2f} mm, {rot_final[i]:5.2f} deg {tag}")

    # ==========================================
    # STEP 4: 전체 방법 비교
    # ==========================================

    #   필터링 끝난 샘플로 5가지 알고리즘 전부 돌림:

    #   TSAI       : median=0.55mm
    #   PARK       : median=0.56mm
    #   HORAUD     : median=0.56mm
    #   DANIILIDIS : median=0.64mm
    #   ANDREFF    : median=0.99mm

    #   median residual이 가장 낮은 방법을 선택합니다. mean이 아니라 median을 쓰는 이유는 남은
    #    샘플 중에서도 한두 개 큰 값이 있으면 mean이 왜곡되기 때문.

    #   → TSAI가 0.55mm로 가장 낮아서 최종 선택

    best_name = None
    best_median = float('inf')
    best_R = None
    best_t = None

    for name, method in METHODS.items():
        try:
            R_m, t_m = solve_handeye(Rg, tg, Rt, tt, keep, method=method)
            tr, ro = residual_per_sample(Rg, tg, Rt, tt, R_m, t_m)

            mean_tr = np.mean(tr[keep])
            median_tr = np.median(tr[keep])
            max_tr = np.max(tr[keep])
            mean_ro = np.mean(ro[keep])

            print(f"  {name:12s}: mean={mean_tr:5.2f}mm, median={median_tr:5.2f}mm, "
                  f"max={max_tr:5.2f}mm, rot_mean={mean_ro:.3f}deg")

            # median residual이 가장 낮은 방법 선택
            if median_tr < best_median:
                best_median = median_tr
                best_name = name
                best_R = R_m
                best_t = t_m

        except Exception as e:
            print(f"  {name:12s}: FAILED ({e})")

    print(f"\n  -> Best method: {best_name} (median={best_median:.2f}mm)")

    # ==========================================
    # STEP 5: 최종 결과
    # ==========================================
    trans_best, rot_best = residual_per_sample(Rg, tg, Rt, tt, best_R, best_t)

    print(f"\n{'='*60}")
    print("FINAL RESULT")
    print(f"{'='*60}")
    print(f"Method: {best_name}")
    print(f"Samples used: {np.sum(keep)}/{N}")
    print(f"Residual mean:   {np.mean(trans_best[keep]):.2f} mm, {np.mean(rot_best[keep]):.3f} deg")
    print(f"Residual median: {np.median(trans_best[keep]):.2f} mm, {np.median(rot_best[keep]):.3f} deg")
    print(f"Residual max:    {np.max(trans_best[keep]):.2f} mm, {np.max(rot_best[keep]):.3f} deg")

    T = np.eye(4)
    T[:3, :3] = best_R
    T[:3, 3] = best_t.flatten()

    print(f"\nT_tool0_cam =")
    print(T)

    # ==========================================
    # STEP 6: ROS TF 출력
    # ==========================================
    quat = Rot.from_matrix(best_R).as_quat()

    print(f"\nROS static_transform_publisher:")
    print(
        f"{best_t[0,0]} {best_t[1,0]} {best_t[2,0]} "
        f"{quat[0]} {quat[1]} {quat[2]} {quat[3]} "
        "tool0 camera_color_optical_frame"
    )


if __name__ == "__main__":
    main()
