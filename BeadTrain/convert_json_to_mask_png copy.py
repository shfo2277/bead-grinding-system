
import os
import json
import cv2
import numpy as np

JSON_DIR = "/home/ho/Downloads/besd"
OUT_DIR  = "/home/ho/Downloads/save"   # 저장 폴더
TARGET_LABELS = ("end",)  # 너 JSON 라벨이 "end"

def convert_one(json_path: str, out_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    h = data["imageHeight"]
    w = data["imageWidth"]
    mask = np.zeros((h, w), dtype=np.uint8)

    for s in data.get("shapes", []):
        if s.get("label") not in TARGET_LABELS:
            continue

        pts = np.array(s.get("points", []), dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 255)  # 비드=255, 배경=0

    cv2.imwrite(out_path, mask)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    files = sorted([f for f in os.listdir(JSON_DIR) if f.lower().endswith(".json")])
    if not files:
        print("No json files in:", JSON_DIR)
        return

    ok, fail = 0, 0
    for fn in files:
        jp = os.path.join(JSON_DIR, fn)
        outp = os.path.join(OUT_DIR, os.path.splitext(fn)[0] + ".png")
        try:
            convert_one(jp, outp)
            ok += 1
        except Exception as e:
            fail += 1
            print("[FAIL]", fn, "->", e)

    print(f"Done. success={ok}, fail={fail}")
    print("Saved to:", OUT_DIR)

if __name__ == "__main__":
    main()



# #===================================================

# import json
# import cv2
# import numpy as np
# import os

# import os
# import json
# import cv2
# import numpy as np
# from glob import glob
# from tqdm import tqdm

# json_path = "/home/ho/Downloads/비드라벨링/비드"   # 네 JSON 폴더 경로로 수정
# img_dir = "/home/ho/BEADtrain/REAL/fitimage"        # capture_*.jpg 폴더
# save_dir = "/home/ho/BEADtrain/REAL/fitmask"
# os.makedirs(save_dir, exist_ok=True)

# # 저장 폴더 생성
# os.makedirs(save_dir, exist_ok=True)

# def json_to_mask(json_path):
#     with open(json_path, "r") as f:
#         data = json.load(f)

#     h = data["imageHeight"]
#     w = data["imageWidth"]

#     # 빈 마스크 생성
#     mask = np.zeros((h, w), dtype=np.uint8)

#     # polygon 그리기
#     for shape in data["shapes"]:
#         if shape["shape_type"] == "polygon":
#             pts = np.array(shape["points"], dtype=np.int32)
#             cv2.fillPoly(mask, [pts], 255)

#     return mask, data["imagePath"]


# # 🟦 모든 JSON 파일 가져오기
# json_files = sorted(glob(os.path.join(json_path, "*.json")))

# for json_file in tqdm(json_files, desc="🔄 JSON → Mask 생성 중"):
#     mask, image_name = json_to_mask(json_file)

#     # 마스크 저장 이름
#     out_name = os.path.splitext(image_name)[0] + ".png"
#     out_path = os.path.join(save_dir, out_name)

#     cv2.imwrite(out_path, mask)

# print("✅ 모든 JSON 마스크 생성 완료!")







#=================================================
# import os
# import json
# import cv2
# import numpy as np
# from glob import glob
# from tqdm import tqdm

# # # 경로 설정 test
# # JSON_DIR = "/home/ho/BEADtrain/DATA/Test/json"
# # OUTPUT_MASK_DIR = "/home/ho/BEADtrain/DATA/Test/mask"

# # #train
# # JSON_DIR = "/home/ho/BEADtrain/DATA/Training/label/json"
# # OUTPUT_MASK_DIR = "/home/ho/BEADtrain/DATA/Training/label/mask"

# # #validation
# JSON_DIR = "/home/ho/BEADtrain/DATA/Validation/Vlabel/VL_VTST_정상"
# OUTPUT_MASK_DIR = "/home/ho/BEADtrain/DATA/Validation/Vlabel/mask"

# # 저장 디렉토리 생성
# os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# def json_to_mask(json_path):
#     with open(json_path, 'r') as f:
#         data = json.load(f)

#     # JSON에서 직접 이미지 크기 가져오기
#     height = data["image_data"]["height"]
#     width = data["image_data"]["width"]
#     mask = np.zeros((height, width), dtype=np.uint8)

#     for anno in data.get("annotations", []):
#         x_coords = anno["coordinate"]["x"]
#         y_coords = anno["coordinate"]["y"]
#         points = np.array(list(zip(x_coords, y_coords)), dtype=np.int32)
#         cv2.fillPoly(mask, [points], color=255)  # binary mask: 255 for foreground

#     return mask

# # JSON 파일 리스트
# json_files = sorted(glob(os.path.join(JSON_DIR, "*.json")))

# # 마스크 생성 및 저장
# for json_file in tqdm(json_files, desc="🔄 마스크 생성 중"):
#     file_name = os.path.splitext(os.path.basename(json_file))[0]
#     mask = json_to_mask(json_file)

#     mask_path = os.path.join(OUTPUT_MASK_DIR, f"{file_name}.png")
#     cv2.imwrite(mask_path, mask)

# print("✅ 모든 마스크가 성공적으로 저장되었습니다!")
