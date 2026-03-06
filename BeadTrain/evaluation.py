import os, glob, random, time
import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# =========================
# CONFIG (너 환경에 맞게)
# =========================
DATA_ROOT = "/workspace/BEADtrain/REAL"
IMAGE_DIR = f"{DATA_ROOT}/end/endimg"
MASK_DIR  = f"{DATA_ROOT}/end/endmask"
MODEL_PATH = "/workspace/BEADtrain/modelresult/2026-02-12_19-28-57/unet_best.pth"
IMAGE_SIZE = 1280
THRESHOLD  = 0.4
SEED = 42

NUM_WORKERS = 2
BATCH_SIZE = 1  # 평가 안정적으로 1 추천 (GPU 여유 있으면 늘려도 됨)


# =========================
# Utils
# =========================
def get_mask_path(img_path):
    base = os.path.splitext(os.path.basename(img_path))[0]
    for ext in [".png", ".jpg"]:
        cand = os.path.join(MASK_DIR, base + ext)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Mask not found for {img_path}")

# (학습 때 val/test에 쓰던 transform과 동일)
eval_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE,
                      border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)        # 0 or 1
        mask = np.expand_dims(mask, axis=-1)        # H,W,1

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # HWC -> CHW
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.permute(2, 0, 1)

        return image, mask


def compute_metrics_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    """
    logits:  (B,1,H,W) (sigmoid 전)
    targets: (B,1,H,W) (0~1)
    returns: iou, dice, precision, recall, pixel_acc
    """
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
    union = (p.sum(dim=1) + t.sum(dim=1) - intersection)
    iou  = (intersection + eps) / (union + eps)

    dice = (2 * intersection + eps) / (p.sum(dim=1) + t.sum(dim=1) + eps)

    return (
        iou.mean().item(),
        dice.mean().item(),
        precision.mean().item(),
        recall.mean().item(),
        pixel_acc.mean().item(),
    )


# =========================
# 1) 이미지 목록 로드 + (학습과 동일하게) split 재현
# =========================
image_paths = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) +
    glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)
print("총 이미지 개수:", len(image_paths))

random.seed(SEED)
random.shuffle(image_paths)

n_total = len(image_paths)
n_train = int(n_total * 0.7)
n_val   = int(n_total * 0.15)
n_test  = n_total - n_train - n_val

train_imgs = image_paths[:n_train]
val_imgs   = image_paths[n_train:n_train+n_val]
test_imgs  = image_paths[n_train+n_val:]

print(f"Split -> Train:{len(train_imgs)} Val:{len(val_imgs)} Test:{len(test_imgs)}")


# =========================
# 2) Test loader
# =========================
test_dataset = BeadDataset(test_imgs, transform=eval_transform)
test_loader  = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# =========================
# 3) 모델 로드
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,   # 평가 시엔 None (가중치는 .pth에서 로드)
    in_channels=3,
    classes=1,
).to(device)

state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()
print("✅ Model loaded:", MODEL_PATH)


# =========================
# 4) 평가 (IoU/Dice/Precision/Recall/Accuracy + inference time)
# =========================
sum_iou = sum_dice = sum_prec = sum_rec = sum_acc = 0.0
n = 0

# inference time 측정(모델 forward만, ms)
time_list_ms = []

with torch.no_grad():
    for images, masks in tqdm(test_loader, desc="Test evaluating"):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        # GPU 타이밍 정확히 재려면 synchronize 필요
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        time_list_ms.append((t1 - t0) * 1000.0)

        iou, dice, prec, rec, acc = compute_metrics_from_logits(
            logits, masks, threshold=THRESHOLD
        )

        sum_iou  += iou
        sum_dice += dice
        sum_prec += prec
        sum_rec  += rec
        sum_acc  += acc
        n += 1

avg_iou  = sum_iou / n
avg_dice = sum_dice / n
avg_prec = sum_prec / n
avg_rec  = sum_rec / n
avg_acc  = sum_acc / n

avg_time = float(np.mean(time_list_ms))
p95_time = float(np.percentile(time_list_ms, 95))

result_lines = [
    "==============================",
    f"Model: {MODEL_PATH}",
    f"Data : {IMAGE_DIR}",
    f"Test samples: {n}  |  Threshold: {THRESHOLD}  |  ImageSize: {IMAGE_SIZE}",
    "------------------------------",
    f"📊 Test Mean IoU        : {avg_iou:.4f}",
    f"📊 Test Mean Dice       : {avg_dice:.4f}",
    f"📊 Test Mean Precision  : {avg_prec:.4f}",
    f"📊 Test Mean Recall     : {avg_rec:.4f}",
    f"📊 Test Mean Pixel Acc  : {avg_acc:.4f}",
    f"⏱  Inference Time (ms) : mean {avg_time:.2f} ms / p95 {p95_time:.2f} ms  (batch={BATCH_SIZE})",
    "==============================",
]

print("\n" + "\n".join(result_lines))

# 모델 폴더에 결과 저장
result_path = os.path.join(os.path.dirname(MODEL_PATH), "evaluation_result.txt")
with open(result_path, 'w') as f:
    f.write("\n".join(result_lines) + "\n")
print(f"\n📁 결과 저장: {result_path}")