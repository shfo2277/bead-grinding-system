# RealSense D405 컬러 스트림 뷰어 + 이미지 캡처 스크립트
# 's' 키로 원본 해상도(1280x720) PNG 저장, 'q' 키로 종료.

import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

# 저장 경로
save_dir = "/home/ho/BEADtrain/REAL/endg"
os.makedirs(save_dir, exist_ok=True)

# ✅ D405 컬러 최대 해상도
W, H, FPS = 1280, 720, 30

pipeline = rs.pipeline()
config = rs.config()

# 컬러 스트림만 (라벨링용)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)

profile = pipeline.start(config)

print(f"✅ Streaming Color at {W}x{H} @ {FPS}fps")
print("🎥 Press 's' to SAVE image, 'q' to QUIT")

count = 0
display_scale = 0.6  # 화면 표시만 축소 (저장은 원본)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())  # (H,W,3) BGR

        # 화면 표시용 축소
        disp = cv2.resize(img, None, fx=display_scale, fy=display_scale)
        cv2.imshow("D405 Color (MAX)", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(save_dir, f"d405_{ts}_{count:04d}_{W}x{H}.png")
            cv2.imwrite(path, img)  # ✅ PNG(무손실) 추천
            print("✅ Saved:", path)
            count += 1
        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("🛑 종료")
