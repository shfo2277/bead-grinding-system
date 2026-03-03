import os

# =========================
# 설정 (둘 다 .png)
# =========================
image_dir = "/home/ho/Downloads/saveim"
mask_dir  = "/home/ho/Downloads/save"

DRY_RUN = False   # ✅ True: 삭제 안 하고 목록만 출력 / False: 실제 삭제

# =========================
# 파일 목록 로드
# =========================
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".png")]
mask_files  = [f for f in os.listdir(mask_dir)  if f.lower().endswith(".png")]

image_names = {os.path.splitext(f)[0] for f in image_files}
mask_names  = {os.path.splitext(f)[0] for f in mask_files}

common_names = image_names & mask_names
only_images  = image_names - common_names   # 마스크 없는 이미지
only_masks   = mask_names  - common_names   # 이미지 없는 마스크

print(f"📷 이미지(.png) 파일 수: {len(image_files)}")
print(f"🧷 마스크(.png) 파일 수: {len(mask_files)}")
print(f"✅ 1:1 매칭된 파일 수: {len(common_names)}")
print(f"❌ 마스크 없는 이미지 수: {len(only_images)}")
print(f"❌ 이미지 없는 마스크 수: {len(only_masks)}")

# =========================
# 삭제 대상 목록 출력
# =========================
if only_images:
    print("\n[삭제 대상] 마스크 없는 이미지:")
    for name in sorted(only_images):
        print(" -", name + ".png")

if only_masks:
    print("\n[삭제 대상] 이미지 없는 마스크:")
    for name in sorted(only_masks):
        print(" -", name + ".png")

# =========================
# 실제 삭제
# =========================
if DRY_RUN:
    print("\n⚠️ DRY_RUN=True 상태라서 삭제는 하지 않았어.")
    print("✅ 삭제하려면 DRY_RUN=False로 바꾸고 다시 실행해.")
else:
    # 마스크 없는 이미지 삭제
    for name in only_images:
        img_path = os.path.join(image_dir, name + ".png")
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"🗑️ 삭제된 이미지: {img_path}")

    # 이미지 없는 마스크 삭제
    for name in only_masks:
        mask_path = os.path.join(mask_dir, name + ".png")
        if os.path.exists(mask_path):
            os.remove(mask_path)
            print(f"🗑️ 삭제된 마스크: {mask_path}")

    print("\n✅ 1:1 매칭 안 되는 파일 삭제 완료!")
