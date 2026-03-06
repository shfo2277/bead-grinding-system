#!/usr/bin/env python3
"""
RealSense D405 뎁스 정확도 테스트
- ROI 중심영역의 depth 통계(평균, 표준편차, min, max)를 실시간 표시
- 's' 키: 현재 depth 통계를 기록 (여러 거리에서 측정 후 비교)
- 'r' 키: ROI 크기 변경 (10x10, 20x20, 50x50 순환)
- 'q' 키: 종료 및 결과 출력

사용법:
  1. 평평한 면에 카메라를 수직으로 놓고 알려진 거리에서 측정
  2. 게이지 블록(높이 아는 물체)을 올려놓고 depth 차이 확인
  3. 여러 거리에서 's'로 기록 → 종료 시 요약 출력
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import sys


def main():
    # ── 스트림 설정 ──
    pipeline = rs.pipeline()
    config = rs.config()

    # D405 최대 해상도: 1280x720
    W, H = 1280, 720
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    # depth scale 확인
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[INFO] Depth scale = {depth_scale}")  # D405: 보통 0.0001 (0.1mm 단위)

    # depth 필터 (선택)
    # temporal = rs.temporal_filter()
    # spatial = rs.spatial_filter()

    align = rs.align(rs.stream.color)

    # ── ROI 설정 ──
    roi_sizes = [10, 20, 50]
    roi_idx = 0
    roi_half = roi_sizes[roi_idx] // 2

    # ── 기록 저장 ──
    records = []  # [(label, mean_mm, std_mm, min_mm, max_mm)]

    # ── N 프레임 평균용 버퍼 ──
    BUFFER_SIZE = 30
    depth_buffer = []

    print("=" * 60)
    print("RealSense D405 Depth Accuracy Test")
    print("=" * 60)
    print(f"  ROI size: {roi_sizes[roi_idx]}x{roi_sizes[roi_idx]} px (center)")
    print(f"  's': 현재 값 기록 (known distance 입력)")
    print(f"  'r': ROI 크기 변경")
    print(f"  'c': 클릭으로 ROI 중심 이동")
    print(f"  'q': 종료")
    print("=" * 60)

    # ROI 중심 (기본: 이미지 중앙)
    roi_cx, roi_cy = W // 2, H // 2
    click_mode = False

    def mouse_cb(event, x, y, flags, param):
        nonlocal roi_cx, roi_cy
        if event == cv2.EVENT_LBUTTONDOWN and click_mode:
            # 디스플레이 스케일 보정
            roi_cx = int(x / display_scale)
            roi_cy = int(y / display_scale)
            print(f"  ROI center moved to ({roi_cx}, {roi_cy})")

    display_scale = 0.7
    cv2.namedWindow("Depth Accuracy Test")
    cv2.setMouseCallback("Depth Accuracy Test", mouse_cb)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 필터 적용 (필요 시)
            # depth_frame = temporal.process(depth_frame)
            # depth_frame = spatial.process(depth_frame)

            depth_image = np.asanyarray(depth_frame.get_data())  # uint16, raw
            color_image = np.asanyarray(color_frame.get_data())

            # ── ROI 영역 depth 추출 ──
            y1 = max(0, roi_cy - roi_half)
            y2 = min(H, roi_cy + roi_half)
            x1 = max(0, roi_cx - roi_half)
            x2 = min(W, roi_cx + roi_half)

            roi_depth = depth_image[y1:y2, x1:x2].astype(np.float64)

            # 0(invalid) 제거
            valid = roi_depth[roi_depth > 0]

            if len(valid) == 0:
                mean_mm = std_mm = min_mm = max_mm = 0.0
                fill_rate = 0.0
            else:
                # raw값 × depth_scale × 1000 = mm
                values_mm = valid * depth_scale * 1000.0
                mean_mm = float(np.mean(values_mm))
                std_mm = float(np.std(values_mm))
                min_mm = float(np.min(values_mm))
                max_mm = float(np.max(values_mm))
                fill_rate = len(valid) / roi_depth.size * 100.0

            # N-프레임 버퍼에 추가 (시간 평균)
            if mean_mm > 0:
                depth_buffer.append(mean_mm)
                if len(depth_buffer) > BUFFER_SIZE:
                    depth_buffer.pop(0)

            avg_of_means = np.mean(depth_buffer) if depth_buffer else 0.0
            std_of_means = np.std(depth_buffer) if len(depth_buffer) > 1 else 0.0

            # ── 시각화 ──
            vis = color_image.copy()

            # ROI 사각형
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 텍스트 정보
            info_lines = [
                f"ROI: {roi_sizes[roi_idx]}x{roi_sizes[roi_idx]}px  "
                f"center=({roi_cx},{roi_cy})",
                f"Depth(mm): mean={mean_mm:.2f}  std={std_mm:.3f}  "
                f"range=[{min_mm:.2f}, {max_mm:.2f}]",
                f"Fill rate: {fill_rate:.1f}%",
                f"{BUFFER_SIZE}-frame avg: {avg_of_means:.2f} mm  "
                f"jitter(std): {std_of_means:.3f} mm",
            ]

            for i, line in enumerate(info_lines):
                cv2.putText(
                    vis, line, (10, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

            # depth colormap (하단 반)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (255, 255, 255), 2)

            combined = np.vstack([vis, depth_colormap])
            disp = cv2.resize(combined, None, fx=display_scale, fy=display_scale)
            cv2.imshow("Depth Accuracy Test", disp)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('s'):
                # 현재 값 기록
                label = input(
                    f"\n  실제 거리(mm) 입력 (현재 측정값: {avg_of_means:.2f}mm): "
                ).strip()
                if not label:
                    label = "unknown"
                records.append((label, avg_of_means, std_of_means, min_mm, max_mm))
                print(f"  ✅ 기록됨: actual={label}mm, measured={avg_of_means:.2f}mm")

            elif key == ord('r'):
                roi_idx = (roi_idx + 1) % len(roi_sizes)
                roi_half = roi_sizes[roi_idx] // 2
                depth_buffer.clear()
                print(f"  ROI changed: {roi_sizes[roi_idx]}x{roi_sizes[roi_idx]}px")

            elif key == ord('c'):
                click_mode = not click_mode
                state = "ON" if click_mode else "OFF"
                print(f"  Click mode: {state}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    # ── 결과 요약 출력 ──
    if records:
        print("\n" + "=" * 70)
        print("Depth Accuracy Test Results")
        print("=" * 70)
        print(f"{'Actual(mm)':>12} {'Measured(mm)':>14} {'Error(mm)':>12} "
              f"{'Jitter(std)':>12} {'Range(mm)':>20}")
        print("-" * 70)

        errors = []
        for actual_str, meas, jitter, mn, mx in records:
            try:
                actual = float(actual_str)
                err = meas - actual
                errors.append(abs(err))
                print(f"{actual:>12.2f} {meas:>14.2f} {err:>12.3f} "
                      f"{jitter:>12.3f} [{mn:.2f}, {mx:.2f}]")
            except ValueError:
                print(f"{actual_str:>12} {meas:>14.2f} {'N/A':>12} "
                      f"{jitter:>12.3f} [{mn:.2f}, {mx:.2f}]")

        if errors:
            print("-" * 70)
            print(f"  Mean absolute error: {np.mean(errors):.3f} mm")
            print(f"  Max absolute error:  {np.max(errors):.3f} mm")
        print("=" * 70)
    else:
        print("\n기록 없음. 다음에는 's'로 측정값을 기록해주세요.")


if __name__ == "__main__":
    main()
