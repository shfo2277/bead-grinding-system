# 기존 학습된 모델의 Train/Val/Test 성능 평가 스크립트 (학습 없이 평가만)
# - bead_train_end.py와 동일한 데이터 분할(SEED=42)을 재현하여 Train/Val/Test 각각 평가
# - Loss(BCE+Dice), IoU, Dice, Precision, Recall, Pixel Accuracy 산출
# - 결과: 모델 폴더에 full_evaluation_result.txt 저장

#==============================================
import os
import glob
import random
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


#==============================================
# 0) CONFIG
#==============================================
# DATA_ROOT = "/home/ho/BEADtrain/REAL"
DATA_ROOT = "/workspace/BEADtrain/BEADtrain0305/BEADtrain/REAL"
IMAGE_DIR = f"{DATA_ROOT}/end/endimg"
MASK_DIR  = f"{DATA_ROOT}/end/endmask"

MODEL_PATH = "/workspace/BEADtrain/modelresult/2026-02-12_19-28-57/unet_best.pth"

IMAGE_SIZE = 1280
THRESHOLD  = 0.4
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
print(f"총 이미지 개수: {len(image_paths)}")
print(f"데이터 경로: {IMAGE_DIR}")

if len(image_paths) == 0:
    print("[오류] 이미지가 없습니다. 경로를 확인하세요.")
    exit(1)

def get_mask_path(img_path):
    base = os.path.splitext(os.path.basename(img_path))[0]
    for ext in [".png", ".jpg"]:
        cand = os.path.join(MASK_DIR, base + ext)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Mask not found for {img_path}")

# 학습과 동일한 split 재현 (같은 SEED)
random.shuffle(image_paths)
n_total = len(image_paths)

n_train = int(n_total * 0.7)
n_val   = int(n_total * 0.15)
n_test  = n_total - n_train - n_val

train_imgs = image_paths[:n_train]
val_imgs   = image_paths[n_train:n_train+n_val]
test_imgs  = image_paths[n_train+n_val:]

print(f"Split -> Train:{len(train_imgs)} Val:{len(val_imgs)} Test:{len(test_imgs)}")


#==============================================
# 2) Transform (평가용 - augmentation 없음)
#==============================================
eval_transform = A.Compose([
    A.LongestMaxSize(max_size=IMAGE_SIZE),
    A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE,
                  border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


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

        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]

        if isinstance(mask, np.ndarray):
            if mask.ndim == 2:
                mask = mask[None, ...]
            else:
                mask = np.transpose(mask, (2, 0, 1))
            mask = torch.from_numpy(mask).float()
        else:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3:
                mask = mask.permute(2, 0, 1)

        return image, mask


#==============================================
# 4) 모델 로드 (학습 안 함, 기존 모델 그대로)
#==============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
).to(device)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print(f"Model loaded: {MODEL_PATH}")


#==============================================
# 5) Loss 함수 (bead_train_end.py와 동일)
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

# pos_weight 계산 (학습 때와 동일 방식)
def estimate_pos_weight(img_list, max_samples=200):
    sample_imgs = img_list[:min(len(img_list), max_samples)]
    pos = neg = 0
    for ip in sample_imgs:
        mp = get_mask_path(ip)
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = (m > 127).astype(np.uint8)
        pos += int(m.sum())
        neg += int(m.size - m.sum())
    if pos == 0:
        return 1.0
    return float(neg / pos)

pos_w_value = estimate_pos_weight(train_imgs)
pos_w_value = min(pos_w_value, 10.0)
pos_weight = torch.tensor([pos_w_value], device=device)
print(f"[pos_weight] estimated(clipped): {pos_w_value:.3f}")

bce_loss_fn  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
dice_loss_fn = DiceLoss()

def total_loss_fn(preds, targets):
    return bce_loss_fn(preds, targets) + dice_loss_fn(preds, targets)


#==============================================
# 6) 평가 함수 (Train/Val/Test 공통)
#==============================================
def compute_metrics(logits, targets, threshold=0.4, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    t = (targets > 0.5).float()

    p = preds.view(preds.size(0), -1)
    t = t.view(t.size(0), -1)

    TP = (p * t).sum(dim=1)
    FP = (p * (1 - t)).sum(dim=1)
    FN = ((1 - p) * t).sum(dim=1)
    TN = ((1 - p) * (1 - t)).sum(dim=1)

    precision = (TP + eps) / (TP + FP + eps)
    recall    = (TP + eps) / (TP + FN + eps)
    pixel_acc = (TP + TN + eps) / (TP + TN + FP + FN + eps)
    intersection = TP
    union = p.sum(dim=1) + t.sum(dim=1) - intersection
    iou  = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (p.sum(dim=1) + t.sum(dim=1) + eps)

    return iou.mean().item(), dice.mean().item(), precision.mean().item(), recall.mean().item(), pixel_acc.mean().item()


def evaluate_set(name, img_list):
    dataset = BeadDataset(img_list, transform=eval_transform)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=2, pin_memory=True)

    sum_loss = sum_bce = sum_dice_l = 0.0
    sum_iou = sum_dice = sum_prec = sum_rec = sum_acc = 0.0
    n = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc=f"{name}"):
            images = images.to(device)
            masks  = masks.to(device)

            logits = model(images)

            # Loss
            bce_val  = bce_loss_fn(logits, masks).item()
            dice_val = dice_loss_fn(logits, masks).item()
            sum_bce    += bce_val
            sum_dice_l += dice_val
            sum_loss   += bce_val + dice_val

            # Metrics
            iou, dice, prec, rec, acc = compute_metrics(logits, masks, threshold=THRESHOLD)
            sum_iou  += iou
            sum_dice += dice
            sum_prec += prec
            sum_rec  += rec
            sum_acc  += acc
            n += 1

    if n == 0:
        print(f"  [{name}] 데이터 없음!")
        return None

    return {
        'name': name,
        'n': n,
        'total_loss': sum_loss / n,
        'bce_loss': sum_bce / n,
        'dice_loss': sum_dice_l / n,
        'iou': sum_iou / n,
        'dice': sum_dice / n,
        'precision': sum_prec / n,
        'recall': sum_rec / n,
        'pixel_acc': sum_acc / n,
    }


#==============================================
# 7) Train / Val / Test 전부 평가
#==============================================
print()
print("=" * 60)
print("  Train / Val / Test Loss + Metrics (평가 모드)")
print("=" * 60)

results_all = []
for name, imgs in [("Train", train_imgs), ("Val", val_imgs), ("Test", test_imgs)]:
    r = evaluate_set(name, imgs)
    if r:
        results_all.append(r)
        print(f"\n  [{r['name']}] ({r['n']}개)")
        print(f"    Total Loss : {r['total_loss']:.4f}  (BCE: {r['bce_loss']:.4f} + Dice: {r['dice_loss']:.4f})")
        print(f"    IoU        : {r['iou']:.4f}")
        print(f"    Dice       : {r['dice']:.4f}")
        print(f"    Precision  : {r['precision']:.4f}")
        print(f"    Recall     : {r['recall']:.4f}")
        print(f"    Pixel Acc  : {r['pixel_acc']:.4f}")


#==============================================
# 8) 요약 테이블
#==============================================
print()
print("=" * 78)
print(f"  {'Set':<8} {'Total Loss':>10} {'BCE':>8} {'Dice':>8} {'IoU':>8} {'Dice(M)':>8} {'Prec':>8} {'Recall':>8} {'Acc':>8}")
print("-" * 78)
for r in results_all:
    print(f"  {r['name']:<8} {r['total_loss']:>10.4f} {r['bce_loss']:>8.4f} {r['dice_loss']:>8.4f} "
          f"{r['iou']:>8.4f} {r['dice']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f} {r['pixel_acc']:>8.4f}")
print("=" * 78)


#==============================================
# 9) 결과 파일 저장
#==============================================
result_path = os.path.join(os.path.dirname(MODEL_PATH), "full_evaluation_result.txt")
with open(result_path, 'w') as f:
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Data : {IMAGE_DIR}\n")
    f.write(f"Threshold: {THRESHOLD}  |  ImageSize: {IMAGE_SIZE}\n")
    f.write(f"pos_weight: {pos_w_value:.3f}\n\n")

    f.write(f"{'Set':<8} {'Total Loss':>10} {'BCE':>8} {'Dice':>8} {'IoU':>8} {'Dice(M)':>8} {'Prec':>8} {'Recall':>8} {'Acc':>8}\n")
    f.write("-" * 78 + "\n")
    for r in results_all:
        f.write(f"{r['name']:<8} {r['total_loss']:>10.4f} {r['bce_loss']:>8.4f} {r['dice_loss']:>8.4f} "
                f"{r['iou']:>8.4f} {r['dice']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f} {r['pixel_acc']:>8.4f}\n")

print(f"\n결과 저장: {result_path}")
