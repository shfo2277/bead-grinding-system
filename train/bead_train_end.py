# UNet 비드 세그멘테이션 학습 스크립트 (개선 버전, 1280 입력)
# - LongestMaxSize + PadIfNeeded로 비율 유지 리사이즈 + 패딩
# - pos_weight 자동 계산, gradient clipping, 학습 시간/ETA 표시
# - BCE + Dice 복합 손실, EarlyStopping, ReduceLROnPlateau
# - 학습 후 Test IoU/Dice/Precision/Recall 평가
# - 결과: modelresult/{timestamp}/unet_best.pth, loss_curve.png, loss_history.csv, train_result.txt

#==============================================
# import
#==============================================
import os
import glob
import random
import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from datetime import datetime



#==============================================
# 0) CONFIG
#==============================================
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

MODEL_SAVE_ROOT = "/home/ho/BEADtrain/modelresult"
MODEL_SAVE_DIR  = os.path.join(MODEL_SAVE_ROOT, RUN_TIMESTAMP)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "unet_best.pth")
LOSS_FIG_PATH   = os.path.join(MODEL_SAVE_DIR, "loss_curve.png")

DATA_ROOT = "/home/ho/BEADtrain/REAL"
IMAGE_DIR = f"{DATA_ROOT}/end/endimg"
MASK_DIR  = f"{DATA_ROOT}/end/endmask"

IMAGE_SIZE = 1280          # 모델 입력 크기 (정사각)
BATCH_SIZE = 2             # 1024면 1~2가 안정적 (원래 6은 터질 확률 높음)
NUM_EPOCHS = 60

# EarlyStopping
EARLY_STOP_PATIENCE = 7
EARLY_STOP_DELTA    = 1e-4

# Aug 옵션: end 라벨이 "방향 의미"가 있으면 flip은 끄는 게 안전
USE_HFLIP = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#==============================================
# 1) 데이터 경로 로드 / 매칭
#==============================================
image_paths = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) +
    glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)
print("총 이미지 개수:", len(image_paths))

def get_mask_path(img_path):
    base = os.path.splitext(os.path.basename(img_path))[0]
    for ext in [".png", ".jpg"]:
        cand = os.path.join(MASK_DIR, base + ext)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Mask not found for {img_path}")

# shuffle + splitm,.'/ m,ㅡ,.
random.shuffle(image_paths)
n_total = len(image_paths)

train_ratio = 0.7
val_ratio   = 0.15

n_train = int(n_total * train_ratio)
n_val   = int(n_total * val_ratio)
n_test  = n_total - n_train - n_val

train_imgs = image_paths[:n_train]
val_imgs   = image_paths[n_train:n_train+n_val]
test_imgs  = image_paths[n_train+n_val:]

print("Train:", len(train_imgs), "Val:", len(val_imgs), "Test:", len(test_imgs))

#==============================================
# 2) Transform (핵심 수정: 비율 유지 + 패딩)
#==============================================
def build_train_transform(image_size=1024, use_hflip=False):
    tfms = [
        # ✅ 비율 유지 리사이즈 + 패딩 (왜곡 방지)
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
    ]

    if use_hflip:
        tfms.append(A.HorizontalFlip(p=0.5))

    tfms += [
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.05,
            rotate_limit=3,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.8
        ),
        A.GaussNoise(p=0.2),
        A.MotionBlur(blur_limit=5, p=0.15),

        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(tfms)

def build_val_transform(image_size=1024):
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),

        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

train_transform = build_train_transform(IMAGE_SIZE, use_hflip=USE_HFLIP)
val_transform   = build_val_transform(IMAGE_SIZE)

#==============================================
# 3) Dataset
#==============================================
class BeadDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = get_mask_path(img_path)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        # ✅ 0/1로 확실히 이진화 (회색값/안티앨리어싱 제거)
        mask = (mask > 127).astype(np.float32)   # 0 or 1
        mask = np.expand_dims(mask, axis=-1)     # (H,W,1)


        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]       # torch (C,H,W)
            mask  = augmented["mask"]        # torch (H,W,1) or (H,W)

        # mask -> (1,H,W)
        if isinstance(mask, np.ndarray):
            # 혹시 ToTensorV2가 없으면 대비
            if mask.ndim == 2:
                mask = mask[None, ...]
            else:
                mask = np.transpose(mask, (2, 0, 1))
            mask = torch.from_numpy(mask).float()
        else:
            # torch tensor
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3:
                mask = mask.permute(2, 0, 1)

        return image, mask

train_dataset = BeadDataset(train_imgs, transform=train_transform)
val_dataset   = BeadDataset(val_imgs,   transform=val_transform)
test_dataset  = BeadDataset(test_imgs,  transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=1,
                          shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=1,
                          shuffle=False, num_workers=2, pin_memory=True)

#==============================================
# 4) device / model
#==============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
).to(device)

#==============================================
# 5) Loss (Dice + BCE) + pos_weight 자동 계산
#==============================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (preds * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (
            preds.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1 - dice.mean()

def estimate_pos_weight(train_imgs, max_samples=200):
    # train 마스크 픽셀 비율로 pos_weight = neg/pos 근사
    sample_imgs = train_imgs[:min(len(train_imgs), max_samples)]
    pos = 0
    neg = 0
    for ip in sample_imgs:
        mp = get_mask_path(ip)
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        # 0/255 -> 0/1
        m = (m > 127).astype(np.uint8)
        pos += int(m.sum())
        neg += int(m.size - m.sum())

    if pos == 0:
        return 1.0  # 전부 배경이면 fallback
    return float(neg / pos)

pos_w_value = estimate_pos_weight(train_imgs)
pos_w_value = min(pos_w_value, 10.0)   # ✅ 상한 (50은 너무 높아 gradient 폭발 위험)
pos_weight = torch.tensor([pos_w_value], device=device)
print(f"[pos_weight] estimated(clipped): {pos_w_value:.3f}")


bce_loss  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
dice_loss = DiceLoss()

def total_loss(preds, targets):
    return bce_loss(preds, targets) + dice_loss(preds, targets)

#==============================================
# 6) Optim / Scheduler
#==============================================
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

#==============================================
# 7) Train / Val
#==============================================
def train_one_epoch(model, loader):
    model.train()
    epoch_loss = 0.0

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = total_loss(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)

    return epoch_loss / len(loader.dataset)

def validate(model, loader):
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)

            logits = model(images)
            loss = total_loss(logits, masks)
            epoch_loss += loss.item() * images.size(0)

    return epoch_loss / len(loader.dataset)

def compute_iou_dice_from_logits(logits, targets, threshold=0.4, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets_bin = (targets > 0.5).float()

    preds_flat   = preds.view(preds.size(0), -1)
    targets_flat = targets_bin.view(targets_bin.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection

    iou  = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (preds_flat.sum(dim=1) + targets_flat.sum(dim=1) + eps)

    return iou.mean().item(), dice.mean().item()

#==============================================
# 8) Loop + EarlyStopping + Best Save
#==============================================
best_val_loss = np.inf
train_loss_history = []
val_loss_history   = []
no_improve_count = 0

import time as _time
_train_start = _time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    _epoch_start = _time.time()
    train_loss = train_one_epoch(model, train_loader)
    val_loss   = validate(model, val_loader)

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    _epoch_sec = _time.time() - _epoch_start
    _elapsed = _time.time() - _train_start
    _eta = _epoch_sec * (NUM_EPOCHS - epoch)
    print(f"[{epoch:02d}/{NUM_EPOCHS}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
          f"({_epoch_sec:.0f}s/epoch, elapsed {_elapsed/60:.1f}min, ETA ~{_eta/60:.1f}min)")
    scheduler.step(val_loss)

    if val_loss < (best_val_loss - EARLY_STOP_DELTA):
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("  ✅ Best model updated:", MODEL_SAVE_PATH)
        no_improve_count = 0
    else:
        no_improve_count += 1
        print(f"  ⏳ No improvement: {no_improve_count}/{EARLY_STOP_PATIENCE}")

        if no_improve_count >= EARLY_STOP_PATIENCE:
            print("🛑 Early stopping triggered!")
            break

#==============================================
# 9) Loss Curve Save
#==============================================
plt.figure()
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history,   label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training / Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(LOSS_FIG_PATH, dpi=200)
plt.close()
print("📉 Loss curve saved:", LOSS_FIG_PATH)

_total_min = (_time.time() - _train_start) / 60.0
print(f"⏱ 총 학습 시간: {_total_min:.1f}분 ({_total_min/60:.1f}시간)")

#==============================================
# 10) Test Eval (Best model)
#==============================================
state_dict = torch.load(MODEL_SAVE_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("✅ Best model loaded:", MODEL_SAVE_PATH)

def compute_full_metrics(logits, targets, threshold=0.4, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    t = (targets > 0.5).float()
    p = preds.view(preds.size(0), -1)
    t = t.view(t.size(0), -1)
    TP = (p * t).sum(dim=1)
    FP = (p * (1 - t)).sum(dim=1)
    FN = ((1 - p) * t).sum(dim=1)
    intersection = TP
    union = p.sum(dim=1) + t.sum(dim=1) - intersection
    iou  = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (p.sum(dim=1) + t.sum(dim=1) + eps)
    precision = (TP + eps) / (TP + FP + eps)
    recall    = (TP + eps) / (TP + FN + eps)
    return iou.mean().item(), dice.mean().item(), precision.mean().item(), recall.mean().item()

sum_iou = sum_dice = sum_prec = sum_rec = 0.0
n = 0
with torch.no_grad():
    for images, masks in tqdm(test_loader, desc="Test"):
        images = images.to(device)
        masks  = masks.to(device)
        logits = model(images)
        iou, dice, prec, rec = compute_full_metrics(logits, masks)
        sum_iou  += iou
        sum_dice += dice
        sum_prec += prec
        sum_rec  += rec
        n += 1

avg_iou  = sum_iou  / max(n, 1)
avg_dice = sum_dice / max(n, 1)
avg_prec = sum_prec / max(n, 1)
avg_rec  = sum_rec  / max(n, 1)

print("==============================")
print(f"📊 Test Mean IoU       : {avg_iou:.4f}")
print(f"📊 Test Mean Dice      : {avg_dice:.4f}")
print(f"📊 Test Mean Precision : {avg_prec:.4f}")
print(f"📊 Test Mean Recall    : {avg_rec:.4f}")
print("==============================")

#==============================================
# 11) 결과 저장 (loss CSV + 성능 txt)
#==============================================
import csv

# epoch별 loss CSV
loss_csv_path = os.path.join(MODEL_SAVE_DIR, "loss_history.csv")
with open(loss_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'val_loss'])
    for i, (tl, vl) in enumerate(zip(train_loss_history, val_loss_history), 1):
        writer.writerow([i, f'{tl:.6f}', f'{vl:.6f}'])
print(f"📄 Loss CSV 저장: {loss_csv_path}")

# 성능 결과 txt
result_path = os.path.join(MODEL_SAVE_DIR, "train_result.txt")
with open(result_path, 'w') as f:
    f.write("==============================\n")
    f.write(f"Model: {MODEL_SAVE_PATH}\n")
    f.write(f"Data : {IMAGE_DIR}\n")
    f.write(f"Image Size: {IMAGE_SIZE}  |  Batch: {BATCH_SIZE}  |  Epochs: {len(train_loss_history)}/{NUM_EPOCHS}\n")
    f.write(f"Training Time: {_total_min:.1f}min ({_total_min/60:.1f}h)\n")
    f.write("------------------------------\n")
    f.write(f"Best Val Loss  : {best_val_loss:.6f}\n")
    f.write(f"Final Train Loss: {train_loss_history[-1]:.6f}\n")
    f.write(f"Final Val Loss  : {val_loss_history[-1]:.6f}\n")
    f.write("------------------------------\n")
    f.write(f"Test Mean IoU       : {avg_iou:.4f}\n")
    f.write(f"Test Mean Dice      : {avg_dice:.4f}\n")
    f.write(f"Test Mean Precision : {avg_prec:.4f}\n")
    f.write(f"Test Mean Recall    : {avg_rec:.4f}\n")
    f.write("==============================\n")
print(f"📄 결과 저장: {result_path}")
