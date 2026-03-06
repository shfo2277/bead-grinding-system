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
# 0) CONFIG — 경로 / 설정 한 곳에서 관리
#==============================================
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

MODEL_SAVE_ROOT = "/home/ho/BEADtrain/modelresult"
MODEL_SAVE_DIR  = os.path.join(MODEL_SAVE_ROOT, RUN_TIMESTAMP)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "unet_best.pth")
LOSS_FIG_PATH   = os.path.join(MODEL_SAVE_DIR, "loss_curve.png")


DATA_ROOT = "/home/ho/BEADtrain/REAL"

# IMAGE_DIR = f"{DATA_ROOT}/endgrinding"
# MASK_DIR  = f"{DATA_ROOT}/fitmask"

IMAGE_DIR = f"{DATA_ROOT}/endgrinding"
MASK_DIR  = f"{DATA_ROOT}/endgrindingmask"

IMAGE_SIZE = 1024          # 원래 코드에서 쓰던 값
BATCH_SIZE = 6             # 원래 batch_size
NUM_EPOCHS = 60            # 원래 num_epochs

# 🔥 EarlyStopping 설정
EARLY_STOP_PATIENCE = 7     # 몇 epoch 동안 개선 없으면 stop
EARLY_STOP_DELTA    = 1e-4  # 이 정도 이상 좋아져야 "개선"으로 인정
#==============================================


# 이미지 파일 목록 가져오기 (jpg, png 둘 다 대응)
image_dir = IMAGE_DIR
mask_dir  = MASK_DIR

image_paths = sorted(
    glob.glob(os.path.join(image_dir, "*.jpg")) +
    glob.glob(os.path.join(image_dir, "*.png"))
)

print("총 이미지 개수:", len(image_paths))  # 70 근처 나와야 함


def get_mask_path(img_path):
    base = os.path.splitext(os.path.basename(img_path))[0]
    for ext in [".png", ".jpg"]:
        cand = os.path.join(mask_dir, base + ext)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Mask not found for {img_path}")


random.seed(42)
random.shuffle(image_paths)

# ✅ 총 데이터 기반 비율 분할 (Train 70%, Val 15%, Test 15%)
n_total = len(image_paths)
print("총 이미지 개수:", n_total)

train_ratio = 0.7
val_ratio   = 0.15  # 나머지 0.15는 test

n_train = int(n_total * train_ratio)
n_val   = int(n_total * val_ratio)
n_test  = n_total - n_train - n_val  # 자동 계산

train_imgs = image_paths[:n_train]
val_imgs   = image_paths[n_train:n_train+n_val]
test_imgs  = image_paths[n_train+n_val:]

print("Train:", len(train_imgs), "Val:", len(val_imgs), "Test:", len(test_imgs))


# 살짝 과적합 버전 (augment)
train_transform = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),

        # 회전/이동은 너무 과하지 않게 (실제 영상 느낌으로)
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.05,
            rotate_limit=3,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7,
        ),

        # 밝기/대비 변화 강화 (조명 변화 대응)
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.8
        ),

        # 노이즈/블러로 실제 카메라 느낌 추가
        A.GaussNoise(p=0.3),
        A.MotionBlur(blur_limit=5, p=0.3),

        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
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

        # 이미지 읽기 (BGR → RGB)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 마스크 읽기 (그레이스케일)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 마스크를 0~1로 스케일링 (binary segmentation 기준)
        mask = mask / 255.0
        mask = mask.astype("float32")

        # H, W, 1 채널로 확장
        mask = np.expand_dims(mask, axis=-1)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # mask shape: [1, H, W] 로 맞추기 (원래 로직 유지)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.permute(2, 0, 1)  # HWC -> CHW

        return image, mask
    

batch_size = BATCH_SIZE  # 1024 해상도면 1~2 정도가 안전 (원래 값 6)

train_dataset = BeadDataset(train_imgs, transform=train_transform)
val_dataset   = BeadDataset(val_imgs,   transform=val_transform)
test_dataset  = BeadDataset(test_imgs,  transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=1,
                          shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=1,
                          shuffle=False, num_workers=2, pin_memory=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",  # 필요 없으면 None
    in_channels=3,
    classes=1,                  # binary segmentation
)

model = model.to(device)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds: (B,1,H,W) logits
        preds = torch.sigmoid(preds)
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (preds * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (
            preds.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1 - dice.mean()


# device와 model 정의된 뒤에 위치 (원래 주석 유지)
pos_weight = torch.tensor([2.0], device=device)  # 2~5 사이 튜닝 가능
bce_loss   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
dice_loss  = DiceLoss()


def total_loss(preds, targets):
    return bce_loss(preds, targets) + dice_loss(preds, targets)


optimizer = optim.Adam(model.parameters(), lr=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,   # val_loss 나빠지면 lr 1/2
    patience=3,   # 3 epoch 동안 개선 없으면
)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    epoch_loss = 0.0

    for images, masks in tqdm(loader):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # (B,1,H,W)
        loss = total_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)

    return epoch_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)

            outputs = model(images)
            loss = total_loss(outputs, masks)
            epoch_loss += loss.item() * images.size(0)

    return epoch_loss / len(loader.dataset)


def compute_iou_dice_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    """
    logits: (B,1,H,W)  모델 출력 (sigmoid 전)
    targets: (B,1,H,W)  GT 마스크 (0~1)
    """
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


num_epochs = NUM_EPOCHS

best_val_loss = np.inf

train_loss_history = []
val_loss_history   = []

# 🔥 EarlyStopping용 변수
no_improve_count = 0

for epoch in range(1, num_epochs+1):
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    val_loss   = validate(model, val_loader, device)

    # ✅ 기록
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    print(f"[{epoch}/{num_epochs}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    # 🔥 여기서 스케줄러에 val_loss 넘겨주기
    scheduler.step(val_loss)


# #----------------------
#     # 가장 좋은 모델 저장
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save(model.state_dict(), MODEL_SAVE_PATH)
#         print("  ✅ Best model updated")

    # ==========================
    # ✅ Best 모델 저장 + EarlyStopping 체크
    # ==========================
    # 이전 best보다 EARLY_STOP_DELTA 이상 좋아졌는지 확인
    if val_loss < (best_val_loss - EARLY_STOP_DELTA):
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("  ✅ Best model updated")
        no_improve_count = 0  # 개선됐으니 카운터 리셋
    else:
        no_improve_count += 1
        print(f"  ⏳ No improvement count: {no_improve_count}/{EARLY_STOP_PATIENCE}")

        # 🔔 EarlyStopping 발동
        if no_improve_count >= EARLY_STOP_PATIENCE:
            print("🛑 Early stopping triggered! (validation loss not improving)")
            break


#-------------평가 파트 -----------------------
# ==========================
# ✅  학습 곡선(Loss) 그래프 저장
# ==========================
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

print(f"📉 Loss 곡선 저장 완료: {LOSS_FIG_PATH}")

# ==========================
# ✅  Test set IoU / Dice 평가
# ==========================
best_model_path = MODEL_SAVE_PATH
state_dict = torch.load(best_model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print(f"✅ Best 모델 로드 완료: {best_model_path}")

total_iou  = 0.0
total_dice = 0.0
num_batches = 0

with torch.no_grad():
    for images, masks in tqdm(test_loader, desc="Test 평가 중"):
        images = images.to(device)
        masks  = masks.to(device)

        logits = model(images)
        batch_iou, batch_dice = compute_iou_dice_from_logits(logits, masks)

        total_iou  += batch_iou
        total_dice += batch_dice
        num_batches += 1

avg_iou  = total_iou  / num_batches
avg_dice = total_dice / num_batches

#이건 분리해 놓은 테스트 데이터로 평가하고 있음
print("==============================")
print(f"📊 Test Mean IoU : {avg_iou:.4f}")
print(f"📊 Test Mean Dice: {avg_dice:.4f}")
print("    (IoU/Dice는 1.0에 가까울수록 좋음)")
print("==============================")
