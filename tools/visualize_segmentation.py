"""
세그멘테이션 결과 시각화
원본 RGB | 예측 마스크 | 오버레이 (1행 3열) x 각 이미지별 저장
"""
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
MODEL_PATH = "/workspace/BEADtrain/modelresult/2026-03-07_23-23-27/unet_best.pth"
SAVE_DIR   = "/workspace/BEADtrain/image/2026-03-05_05-49-02/scan_2"

IMAGE_PATHS = [
    # ("/workspace/BEADtrain/image/2026-03-04_07-15-55/1_raw_camera.png",     "segmentation_result_0304_0715"),
    # ("/workspace/BEADtrain/image/2026-03-04_08-24-56/1_raw_camera.png",     "segmentation_result_0304_0824"),
    # ("/workspace/BEADtrain/image/2026-03-05_05-49-02/recognition/1_raw_camera.png", "segmentation_result_0305_0549"),
    # ("/workspace/BEADtrain/image/2026-03-05_05-49-02/scan_3/raw.png", "scan3"),
    ("/workspace/BEADtrain/image/2026-03-05_05-49-02/scan_2/raw.png", "scan2"),
]

IMAGE_SIZE = 1280
THRESHOLD  = 0.5

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.LongestMaxSize(max_size=IMAGE_SIZE),
    A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE,
                  border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

# =========================
# 모델 로드
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
).to(device)

state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()
print(f"Model loaded: {MODEL_PATH}")

# =========================
# 각 이미지별 추론 + 저장
# =========================
for img_path, save_name in IMAGE_PATHS:
    frame_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = frame_rgb.shape

    augmented = transform(image=frame_rgb)
    img_tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        prob = torch.sigmoid(logits)[0, 0]

    mask_full = (prob > THRESHOLD).cpu().numpy().astype(np.uint8) * 255

    # 패딩 제거 → 원본 해상도로 복원
    scale = IMAGE_SIZE / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    pad_top = (IMAGE_SIZE - new_h) // 2
    pad_left = (IMAGE_SIZE - new_w) // 2
    mask_cropped = mask_full[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    mask_resized = cv2.resize(mask_cropped, (w, h), interpolation=cv2.INTER_NEAREST)

    print(f"  {save_name} - {w}x{h}, mask pixels: {np.count_nonzero(mask_resized)}")

    # 1) 흑백 마스크 저장
    mask_path = os.path.join(SAVE_DIR, f"{save_name}_mask.png")
    cv2.imwrite(mask_path, mask_resized)
    print(f"    저장: {mask_path}")

    # 2) 오버레이 저장 (BGR)
    overlay = frame_bgr.copy()
    overlay[mask_resized > 0] = (
        0.5 * overlay[mask_resized > 0] + 0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)
    overlay_path = os.path.join(SAVE_DIR, f"{save_name}_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"    저장: {overlay_path}")

print("완료!")
