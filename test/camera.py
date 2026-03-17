# RealSense D405 컬러 영상 녹화 + 개별 프레임 캡처 스크립트
# 's' 키로 PNG 이미지 저장, 'q' 키로 종료. 영상 녹화는 현재 비활성화 상태.

import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

# ======================
# 저장 경로
# ======================
save_dir = "/home/ho/BEADtrain/REAL"
os.makedirs(save_dir, exist_ok=True)

# ======================
# 스트림 설정
# ======================
W, H, FPS = 1280, 720, 30

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
pipeline.start(config)

# ======================
# VideoWriter 설정
# ======================
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
video_path = os.path.join(save_dir, f"d4053_{ts}_{W}x{H}_{FPS}fps.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (W, H))

print("🎥 Recording video:", video_path)
print("Press 's' = save image | 'q' = quit")

count = 0
display_scale = 0.6

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())  # BGR

        # ✅ 영상에 프레임 저장 영상 저장 내용
#        video_writer.write(img)

        # 화면 표시용
        disp = cv2.resize(img, None, fx=display_scale, fy=display_scale)
        cv2.imshow("D405 Color Recording", disp)

        key = cv2.waitKey(1) & 0xFF

        # 이미지 단일 저장
        if key == ord('s'):
            img_path = os.path.join(
                save_dir, f"frame__{ts}_{count:04d}.png"
            )
            cv2.imwrite(img_path, img)
            print("📸 Image saved:", img_path)
            count += 1

        elif key == ord('q'):
            break

finally:
    video_writer.release()
    pipeline.stop()
    cv2.destroyAllWindows()
    print("🛑 Recording finished")
