import os
import cv2
import numpy as np

# --------< 설정 >-------------
DATA_ROOT = 'DATASET'
CLASSES = ["A", "B", "C", "D", "E", "F"]
LABEL_MAP = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
RESIZE_SHAPE = (196, 196)
X, y = [], []

# ---------< SOTA 기반 Hu + Height/Width Ratio 특징 추출 >-------------
def extract_sota_feature(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None

    # --- < HSV 마스크 >---
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 30, 60], dtype=np.uint8)
    upper_hsv = np.array([30, 150, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

    # ---< YCrCb 마스크 >---
    img_ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    lower_ycc = np.array([0, 131, 80], dtype=np.uint8)
    upper_ycc = np.array([255, 185, 135], dtype=np.uint8)
    mask_ycc = cv2.inRange(img_ycc, lower_ycc, upper_ycc)

    # ---< 두 마스크 OR 병합 >---
    skin_mask = cv2.bitwise_or(mask_hsv, mask_ycc)

    # ---< Morphology >---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=3)

    # ---< ROI 추출 >---
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    x, y_, w, h = cv2.boundingRect(cnt)
    cropped = skin_mask[y_:y_+h, x:x+w]
    if cropped.size == 0:
        return None
    resized = cv2.resize(cropped, RESIZE_SHAPE, interpolation=cv2.INTER_NEAREST)

    # ---< Hu Moments >---
    hu = cv2.HuMoments(cv2.moments(resized)).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    # ---< Height/Width Ratio >---
    ratio = np.array([h / w], dtype=np.float32)

    # ---< 최종 벡터 >---
    final_feature = np.concatenate([hu_log.astype(np.float32), ratio])  # shape: (8,)
    return final_feature

# ---< 전체 데이터셋 처리 >---
for cls in CLASSES:
    cls_dir = os.path.join(DATA_ROOT, cls)
    if not os.path.isdir(cls_dir): continue
    for fname in os.listdir(cls_dir):
        if not fname.lower().endswith(('.jpg', '.png', '.bmp')): continue
        fpath = os.path.join(cls_dir, fname)
        feature = extract_sota_feature(fpath)
        if feature is None: continue
        X.append(feature)
        y.append(LABEL_MAP[cls])

X = np.array(X)
y = np.array(y, dtype=np.int32)
np.save("sota_train_features.npy", X)
np.save("sota_train_labels.npy", y)
print(f"[저장 완료] sota features shape: {X.shape}, labels: {y.shape}")
