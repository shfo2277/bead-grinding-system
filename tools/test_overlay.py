#!/usr/bin/env python3
"""단일 이미지에 UNet 모델 추론 → 오버레이 이미지 저장"""

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---- CONFIG ----
#IMAGE_PATH = "/workspace/BEADtrain/image/2026-03-04_08-32-46/1_raw_camera.png"
#IMAGE_PATH = "/workspace/BEADtrain/image/2026-03-04_06-42-35/1_raw_camera.png"
IMAGE_PATH = "/workspace/BEADtrain/image/2026-03-05_05-49-02/recognition/1_raw_camera.png"
MODEL_PATH = "/workspace/BEADtrain/modelresult/2026-03-07_23-23-27/unet_best.pth"
OUTPUT_OVERLAY_PATH = "/workspace/BEADtrain/test_overlay_only.png"
OUTPUT_CONTOUR_PATH = "/workspace/BEADtrain/test_overlay_contour.png"
IMAGE_SIZE = 1280

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

transform = A.Compose([
    A.LongestMaxSize(max_size=IMAGE_SIZE),
    A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE,
                  border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

# ---- Load model ----
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3, classes=1,
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"Model loaded: {MODEL_PATH}")
print(f"Device: {DEVICE}")

# ---- Load image ----
frame_bgr = cv2.imread(IMAGE_PATH)
h, w = frame_bgr.shape[:2]
print(f"Image: {IMAGE_PATH} ({w}x{h})")

frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
augmented = transform(image=frame_rgb)
img_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

# ---- Inference ----
with torch.no_grad():
    prob = torch.sigmoid(model(img_tensor))[0, 0]

mask_full = (prob > 0.7).cpu().numpy().astype(np.uint8) * 255

# 패딩 제거 → 원본 해상도로 복원
scale = IMAGE_SIZE / max(h, w)
new_h, new_w = int(h * scale), int(w * scale)
pad_top = (IMAGE_SIZE - new_h) // 2
pad_left = (IMAGE_SIZE - new_w) // 2
mask_cropped = mask_full[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
mask = cv2.resize(mask_cropped, (w, h), interpolation=cv2.INTER_NEAREST)

mask_area = np.count_nonzero(mask)
print(f"Mask area: {mask_area} pixels ({mask_area/(h*w)*100:.2f}%)")

# ---- Overlay (외곽선 없이) ----
overlay_only = frame_bgr.copy()
overlay_only[mask > 0] = (0.5 * overlay_only[mask > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
cv2.imwrite(OUTPUT_OVERLAY_PATH, overlay_only)
print(f"Saved: {OUTPUT_OVERLAY_PATH}")

# ---- Overlay + contour (외곽선 포함) ----
overlay_contour = overlay_only.copy()
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(overlay_contour, contours, -1, (0, 255, 0), 2)
cv2.imwrite(OUTPUT_CONTOUR_PATH, overlay_contour)
print(f"Saved: {OUTPUT_CONTOUR_PATH}")

# ---- 흑백 마스크 저장 ----
OUTPUT_MASK_PATH = "/workspace/BEADtrain/test_mask_bw.png"
cv2.imwrite(OUTPUT_MASK_PATH, mask)
print(f"Saved: {OUTPUT_MASK_PATH}")

print(f"Contours found: {len(contours)}")
if contours:
    largest = max(contours, key=cv2.contourArea)
    print(f"Largest contour area: {cv2.contourArea(largest):.0f} px")
