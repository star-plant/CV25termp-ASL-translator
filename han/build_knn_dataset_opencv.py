import os
import cv2
import numpy as np

# ===== [1] 데이터 전처리 및 특징 추출 함수 =====


# Zhang-Suen Thinning Algorithm for Skeletonization
# Original Author: Linbo Jin (https://github.com/linbojin/Skeletonization-by-Zhang-Suen-Thinning-Algorithm.git)
def zhangSuen(bin_img):
    img = (bin_img > 0).astype(np.uint8)  # binary 0/1로 변환

    def neighbours(x, y, image):
        x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
        return [
            image[x_1][y],
            image[x_1][y1],
            image[x][y1],
            image[x1][y1],
            image[x1][y],
            image[x1][y_1],
            image[x][y_1],
            image[x_1][y_1],
        ]

    def transitions(neigh):
        n = neigh + neigh[0:1]
        return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

    changing1 = changing2 = 1
    while changing1 or changing2:
        changing1 = []
        rows, cols = img.shape
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, img)
                if (
                    img[x][y] == 1
                    and 2 <= sum(n) <= 6
                    and transitions(n) == 1
                    and P2 * P4 * P6 == 0
                    and P4 * P6 * P8 == 0
                ):
                    changing1.append((x, y))
        for x, y in changing1:
            img[x][y] = 0

        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, img)
                if (
                    img[x][y] == 1
                    and 2 <= sum(n) <= 6
                    and transitions(n) == 1
                    and P2 * P4 * P8 == 0
                    and P2 * P6 * P8 == 0
                ):
                    changing2.append((x, y))
        for x, y in changing2:
            img[x][y] = 0

    skel = img * 255  # OpenCV 호환을 위해

    return skel


def find_endpoints(skel):
    endpoints = []
    for y in range(1, skel.shape[0] - 1):
        for x in range(1, skel.shape[1] - 1):
            if skel[y, x]:
                neigh = skel[y - 1 : y + 2, x - 1 : x + 2]
                if np.count_nonzero(neigh) == 2:
                    endpoints.append((x, y))
    return endpoints


def extract_features(skel):
    moments = cv2.moments(skel)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-9)
    eps = find_endpoints(skel)
    eps_flat = np.array(eps).flatten()
    if len(eps_flat) < 20:
        eps_flat = np.pad(eps_flat, (0, 20 - len(eps_flat)), "constant")
    else:
        eps_flat = eps_flat[:20]
    return np.hstack([hu_log, eps_flat])  # shape: (27,)


def histogram_backprojection(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backproj = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    _, mask = cv2.threshold(backproj, 50, 255, cv2.THRESH_BINARY)
    return mask


def denoise_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# ===== [2] 경로 및 클래스 정의 =====
root_dir = "C:/Users/User/SeoulTech/SeoulTech/Term_project/ASL_label/subdir"
classes = ["A", "B", "C"]
label_map = {"A": 0, "B": 1, "C": 2}

# ===== [3] 손 색상 히스토그램 불러오기 =====
hand_hist = np.load("hand_histogram.npy")

# ===== [4] 데이터셋 생성 =====
features = []
labels = []

# 전처리 이미지 저장 경로 생성
os.makedirs("preprocess_dataset/mask", exist_ok=True)
os.makedirs("preprocess_dataset/edges", exist_ok=True)
os.makedirs("preprocess_dataset/skeleton", exist_ok=True)

for cls in classes:
    folder = os.path.join(root_dir, cls)
    for fname in os.listdir(folder):
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is None:
            continue

        mask = histogram_backprojection(img, hand_hist)
        mask = denoise_mask(mask)
        edges = cv2.Canny(mask, 50, 150)
        skel = zhangSuen(edges)
        feat = extract_features(skel)

        features.append(feat)
        labels.append([label_map[cls]])  # OpenCV expects shape (N,1)

        # === 전처리 이미지 저장 ===
        fname_no_ext = os.path.splitext(fname)[0]
        cv2.imwrite(f"preprocess_dataset/mask/{cls}_{fname_no_ext}.png", mask)
        cv2.imwrite(f"preprocess_dataset/edges/{cls}_{fname_no_ext}.png", edges)
        cv2.imwrite(f"preprocess_dataset/skeleton/{cls}_{fname_no_ext}.png", skel)


# ===== [5] 저장 =====
X_train = np.array(features, dtype=np.float32)
y_train = np.array(labels, dtype=np.int32)

np.save("opencv_knn_features.npy", X_train)
np.save("opencv_knn_labels.npy", y_train)

print(f"✅ 저장 완료: 총 샘플 수 = {X_train.shape[0]}")
print("파일 생성: opencv_knn_features.npy, opencv_knn_labels.npy")
