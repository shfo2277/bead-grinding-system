#!/usr/bin/env python3
"""
비드 높이(Z 오차) 측정 스크립트
- 평면(배경) depth와 비드 영역 depth의 차이를 측정
- 마우스 클릭: 기준 평면 영역 / 비드 영역을 각각 지정
- 실시간으로 두 영역의 depth 차이(= 비드 높이) 표시

사용법:
  1. 실행 후 '1' 키 → 마우스로 기준 평면(비드 옆 바닥) 클릭
  2. '2' 키 → 마우스로 비드 위 클릭
  3. 실시간으로 높이 차이(mm) 확인
  4. 실제 비드 높이(캘리퍼스 등)와 비교
"""

import pyrealsense2 as rs
import numpy as np
import cv2


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    W, H = 1280, 720
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[INFO] Depth scale = {depth_scale}")

    align = rs.align(rs.stream.color)

    # 두 영역의 중심
    roi_half = 10  # 20x20 ROI
    ref_point = None   # 기준 평면
    bead_point = None  # 비드 위

    select_mode = 0  # 0=none, 1=ref, 2=bead
    display_scale = 0.7

    BUFFER_SIZE = 30
    ref_buffer = []
    bead_buffer = []

    def mouse_cb(event, x, y, flags, param):
        nonlocal ref_point, bead_point
        if event == cv2.EVENT_LBUTTONDOWN:
            real_x = int(x / display_scale)
            real_y = int(y / display_scale)
            if select_mode == 1:
                ref_point = (real_x, real_y)
                ref_buffer.clear()
                print(f"  기준 평면 설정: ({real_x}, {real_y})")
            elif select_mode == 2:
                bead_point = (real_x, real_y)
                bead_buffer.clear()
                print(f"  비드 영역 설정: ({real_x}, {real_y})")

    cv2.namedWindow("Bead Height Test")
    cv2.setMouseCallback("Bead Height Test", mouse_cb)

    print("=" * 55)
    print("Bead Height (Z Difference) Measurement")
    print("=" * 55)
    print("  '1': 기준 평면 선택 모드 → 클릭")
    print("  '2': 비드 영역 선택 모드 → 클릭")
    print("  'r': ROI 크기 변경 (5/10/15/25)")
    print("  'q': 종료")
    print("=" * 55)

    roi_options = [5, 10, 15, 25]
    roi_opt_idx = 1

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float64)
            color_image = np.asanyarray(color_frame.get_data())

            vis = color_image.copy()

            # 각 영역의 depth(mm) 측정
            def measure_roi(cx, cy):
                y1 = max(0, cy - roi_half)
                y2 = min(H, cy + roi_half)
                x1 = max(0, cx - roi_half)
                x2 = min(W, cx + roi_half)
                roi = depth_image[y1:y2, x1:x2]
                valid = roi[roi > 0]
                if len(valid) == 0:
                    return None, 0.0
                mm = valid * depth_scale * 1000.0
                return float(np.mean(mm)), float(np.std(mm))

            ref_mm = ref_std = bead_mm = bead_std = 0.0
            height_mm = None

            if ref_point:
                val, std = measure_roi(*ref_point)
                if val is not None:
                    ref_buffer.append(val)
                    if len(ref_buffer) > BUFFER_SIZE:
                        ref_buffer.pop(0)
                    ref_mm = np.mean(ref_buffer)
                    ref_std = std

                # 시각화: 파란 사각형
                rx, ry = ref_point
                cv2.rectangle(vis,
                              (rx - roi_half, ry - roi_half),
                              (rx + roi_half, ry + roi_half),
                              (255, 0, 0), 2)
                cv2.putText(vis, "REF", (rx - roi_half, ry - roi_half - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if bead_point:
                val, std = measure_roi(*bead_point)
                if val is not None:
                    bead_buffer.append(val)
                    if len(bead_buffer) > BUFFER_SIZE:
                        bead_buffer.pop(0)
                    bead_mm = np.mean(bead_buffer)
                    bead_std = std

                # 시각화: 빨간 사각형
                bx, by = bead_point
                cv2.rectangle(vis,
                              (bx - roi_half, by - roi_half),
                              (bx + roi_half, by + roi_half),
                              (0, 0, 255), 2)
                cv2.putText(vis, "BEAD", (bx - roi_half, by - roi_half - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 높이 차이 계산
            if ref_mm > 0 and bead_mm > 0:
                # 비드가 튀어나왔으므로 ref가 더 멀고 bead가 가까움
                height_mm = ref_mm - bead_mm

            # 텍스트 표시
            info = [
                f"REF  depth: {ref_mm:.2f} mm  (std: {ref_std:.3f})",
                f"BEAD depth: {bead_mm:.2f} mm  (std: {bead_std:.3f})",
            ]
            if height_mm is not None:
                info.append(f"BEAD HEIGHT: {height_mm:.3f} mm  "
                            f"({height_mm:.1f} um = {height_mm*1000:.0f} um)")
                info.append(f"ROI: {roi_half*2}x{roi_half*2}px  "
                            f"Buffer: {BUFFER_SIZE} frames")

            if select_mode == 1:
                info.append(">>> MODE: Click to set REF point <<<")
            elif select_mode == 2:
                info.append(">>> MODE: Click to set BEAD point <<<")

            for i, line in enumerate(info):
                cv2.putText(vis, line, (10, 30 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            disp = cv2.resize(vis, None, fx=display_scale, fy=display_scale)
            cv2.imshow("Bead Height Test", disp)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('1'):
                select_mode = 1
                print("  >>> 기준 평면 선택 모드. 클릭하세요.")
            elif key == ord('2'):
                select_mode = 2
                print("  >>> 비드 영역 선택 모드. 클릭하세요.")
            elif key == ord('r'):
                roi_opt_idx = (roi_opt_idx + 1) % len(roi_options)
                roi_half = roi_options[roi_opt_idx]
                ref_buffer.clear()
                bead_buffer.clear()
                print(f"  ROI: {roi_half*2}x{roi_half*2}px")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    # 결과 출력
    if ref_mm > 0 and bead_mm > 0 and height_mm is not None:
        print(f"\n{'='*50}")
        print(f"최종 측정 결과")
        print(f"{'='*50}")
        print(f"  기준 평면 depth: {ref_mm:.2f} mm")
        print(f"  비드 영역 depth: {bead_mm:.2f} mm")
        print(f"  비드 높이 (차이): {height_mm:.3f} mm")
        print(f"{'='*50}")
        print(f"  이 값을 캘리퍼스/마이크로미터 실측값과 비교하세요.")


if __name__ == "__main__":
    main()
