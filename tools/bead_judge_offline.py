#!/usr/bin/env python3
"""
bead_unet_node_after.py 기반 오프라인 판정 시각화
- 이미지 파일 경로로 입력받아 UNet 추론
- 기준 외곽선(bead_contour.csv) 대비 shape match 판정
- 마스크, 오버레이(판정 결과 포함) 이미지 저장
"""

import os
import csv
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==============================
# CONFIG
# ==============================
SESSION_DIR = "/workspace/BEADtrain/image/2026-03-04_08-24-56 실험2"
IMAGE_PATHS = [f"{SESSION_DIR}/scan_{i}/1_raw_camera.png" for i in range(1, 12)]
MODEL_PATH = "/workspace/BEADtrain/modelresult/2026-03-07_23-23-27/unet_best.pth"
CONTOUR_CSV = f"{SESSION_DIR}/recognition/bead_contour.csv"

IMAGE_SIZE = 1280
THRESHOLD = 0.5
SHAPE_THRESH = 0.93

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

# ==============================
# Utility
# ==============================
def load_contour_csv(path):
    contour = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for x, y in reader:
            contour.append([[int(float(x)), int(float(y))]])
    return np.array(contour, dtype=np.int32)

def shape_match_ratio(mask, contour):
    contour_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)
    overlap = np.logical_and(mask > 0, contour_mask > 0)
    contour_area = np.count_nonzero(contour_mask)
    return overlap.sum() / max(contour_area, 1)

def put_label(img, text, org, scale=0.8, thickness=2, color=(255, 255, 255)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 6, y - th - 10), (x + tw + 6, y + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

# ==============================
# Main
# ==============================
# Load model
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"Model: {MODEL_PATH}")
print(f"Device: {DEVICE}")

# Load reference contour
ref_contour = load_contour_csv(CONTOUR_CSV)
print(f"Reference contour: {CONTOUR_CSV} ({len(ref_contour)} points)")

for IMAGE_PATH in IMAGE_PATHS:
    # Load image
    frame_bgr = cv2.imread(IMAGE_PATH)
    h, w = frame_bgr.shape[:2]
    print(f"\nImage: {IMAGE_PATH} ({w}x{h})")

    # Inference
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    augmented = transform(image=frame_rgb)
    img_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(model(img_tensor))[0, 0]

    mask_full = (prob > THRESHOLD).cpu().numpy().astype(np.uint8) * 255

    # 패딩 제거 → 원본 해상도
    scale = IMAGE_SIZE / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    pad_top = (IMAGE_SIZE - new_h) // 2
    pad_left = (IMAGE_SIZE - new_w) // 2
    mask_cropped = mask_full[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    mask = cv2.resize(mask_cropped, (w, h), interpolation=cv2.INTER_NEAREST)

    # Shape match 판정
    shape_ratio = shape_match_ratio(mask, ref_contour)
    shape_ok = shape_ratio >= SHAPE_THRESH
    progress = shape_ratio / max(SHAPE_THRESH, 1e-6)
    progress_pct = max(0.0, min(1.0, progress)) * 100.0

    print(f"  Contour coverage: {shape_ratio*100:.2f} % (thr {SHAPE_THRESH*100:.0f}%)")
    print(f"  Progress: {progress_pct:.1f} %")
    print(f"  JUDGE: {'OK' if shape_ok else 'NOK'}")

    # 저장 경로 (입력 이미지와 같은 폴더)
    save_dir = os.path.dirname(IMAGE_PATH)

    # 1) 흑백 마스크
    mask_path = os.path.join(save_dir, "judge_mask.png")
    cv2.imwrite(mask_path, mask)
    print(f"  Saved: {mask_path}")

    # 2) 오버레이 — 영역별 색상 구분
    overlay = frame_bgr.copy()
    alpha = 0.5

    # 기준 contour 영역 마스크 생성
    ref_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(ref_mask, [ref_contour], -1, 255, -1)

    mask_bool = mask > 0
    ref_bool = ref_mask > 0

    grinding = mask_bool & ref_bool
    residual = ref_bool & ~mask_bool
    overcut = mask_bool & ~ref_bool

    overlay[grinding] = (alpha * overlay[grinding] + alpha * np.array([0, 255, 0])).astype(np.uint8)
    overlay[residual] = (alpha * overlay[residual] + alpha * np.array([0, 165, 255])).astype(np.uint8)
    overlay[overcut]  = (alpha * overlay[overcut]  + alpha * np.array([0, 0, 255])).astype(np.uint8)

    overlay_path = os.path.join(save_dir, "judge_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"  Saved: {overlay_path}")

print("\n완료!")
