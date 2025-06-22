import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# === 설정 ===
DATA_ROOT = 'DATASET'
CLASSES = ["A", "B", "C", "D", "E", "F"]
LABEL_MAP = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
RESIZE_SHAPE = (256, 256)
X, y = [], []

# --------------<세선화 함수>-----------------------------
def morphological_skeleton(bin_img):
    blurred = cv2.GaussianBlur(bin_img, (3, 3), 0)
    _, bin_img = cv2.threshold(bin_img, 127, 255, cv2.THRESH_BINARY)
    skel = np.zeros(bin_img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    img = bin_img.copy()
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            done = True
    return skel


# --------------<고립된 노이즈 제거>------------------------
def remove_isolated_white_pixels(bin_img):
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_count = cv2.filter2D((bin_img == 255).astype(np.uint8), -1, kernel)
    mask = (bin_img == 255) & (neighbor_count == 0)
    output = bin_img.copy()
    output[mask] = 0
    return output

# --------------<스켈레톤 두께 증가>------------------------
def thicken_skeleton(skel, kernel_size=3, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    return cv2.dilate(skel, kernel, iterations=iterations)

# --------------<손가락 끝점 추출>------------------------
def find_fingertips(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return []

    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 3:
        return []

    fingertips = []
    try:
        defects = cv2.convexityDefects(cnt, hull)
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                if d > 700:
                    fingertips.append(start)
                    fingertips.append(end)
    except cv2.error:
        return []

    # 중복 제거
    dedup = []
    for pt in fingertips:
        if all(np.linalg.norm(np.array(pt) - np.array(p)) > 10 for p in dedup):
            dedup.append(pt)

    dedup = sorted(dedup, key=lambda p: (p[1], p[0]))
    
    return dedup

# --------------<휴 모멘트 + 손가락 끝점 추출 피쳐 저장>-------------------
def extract_hu_moments_and_fingertips(img_path, cls_name, size=(196, 196), max_points=10):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None

    # -----------<HSV기반, 이진마스크>---------------
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(img_hsv, lower_skin, upper_skin)

    # -----------<이진마스크 후처리>------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=5)

    # -----------<손 영역 크롭 + 리사이즈>------------
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    x, y_, w, h = cv2.boundingRect(cnt)
    cropped = skin_mask[y_:y_+h, x:x+w]
    resized = cv2.resize(cropped, size, interpolation=cv2.INTER_NEAREST)

    # -----------< Hu Moments (skeleton 기반) >------
    skel = morphological_skeleton(resized)
    skel = remove_isolated_white_pixels(skel)
    skel = thicken_skeleton(skel)
    hu = cv2.HuMoments(cv2.moments(skel)).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    # -----------< Fingertips >-----------------------
    fingertips = find_fingertips(resized)
    norm_fingertips = [(x / size[0], y / size[1]) for (x, y) in fingertips]
    norm_fingertips = norm_fingertips[:max_points] + [(0.0, 0.0)] * max(0, max_points - len(norm_fingertips))
    fingertip_vec = np.array(norm_fingertips).flatten().astype(np.float32)

    # -------<디버깅용 코드, 마스크된 이미지 손 끝점 추출>----------
    # vis = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    # for pt in fingertips:
    #     cv2.circle(vis, pt, 6, (0, 255, 255), -1)

    # save_dir = f"HuFingertip_Debug/{cls_name}"
    # os.makedirs(save_dir, exist_ok=True)
    # base = os.path.splitext(os.path.basename(img_path))[0]
    # cv2.imwrite(os.path.join(save_dir, f"{base}_fingertips.png"), vis)

    final_feature = np.concatenate([hu_log, fingertip_vec])
    return final_feature

# -------------< 전체 데이터셋 구축 >----------------
for cls in CLASSES:
    cls_dir = os.path.join(DATA_ROOT, cls)
    if not os.path.isdir(cls_dir): continue
    for fname in os.listdir(cls_dir):
        if not fname.lower().endswith(('.jpg', '.png', '.bmp')): continue
        fpath = os.path.join(cls_dir, fname)
        feature = extract_hu_moments_and_fingertips(fpath, cls)
        if feature is None: continue
        X.append(feature)
        y.append(LABEL_MAP[cls])

# --------------< 저장 >-----------------------
X = np.array(X)
y = np.array(y, dtype=np.int32)
np.save("knn_train_features.npy", X)
np.save("knn_train_labels.npy", y)
print(f"Hu + Fingertip features saved: shape={X.shape}, labels={y.shape}")

# -------------------< 이진 마스크 + 핑거팁 시각화>---------------------
# DEBUG_DIR = "HuFingertip_Debug"
# plt.figure(figsize=(15, 3))
# for idx, cls in enumerate(CLASSES):
#     cls_path = os.path.join(DEBUG_DIR, cls)
#     if not os.path.isdir(cls_path): continue
#     image_files = sorted([f for f in os.listdir(cls_path) if f.endswith("_fingertips.png")])
#     if not image_files: continue
#     img_path = os.path.join(cls_path, image_files[0])
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     plt.subplot(1, len(CLASSES), idx + 1)
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title(f"{cls}")
#     plt.axis('off')
# plt.tight_layout()
# plt.show()
