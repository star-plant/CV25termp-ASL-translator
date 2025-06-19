import os
import cv2
import numpy as np

# ===== [1] 전처리 및 특징 추출 함수들 =====
def morphological_skeleton(bin_img):
    skel = np.zeros(bin_img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = bin_img.copy()
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel

def find_endpoints(skel):
    endpoints = []
    for y in range(1, skel.shape[0]-1):
        for x in range(1, skel.shape[1]-1):
            if skel[y, x]:
                neigh = skel[y-1:y+2, x-1:x+2]
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
        eps_flat = np.pad(eps_flat, (0, 20 - len(eps_flat)), 'constant')
    else:
        eps_flat = eps_flat[:20]
    return np.hstack([hu_log, eps_flat])  # shape: (27,)

def histogram_backprojection(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backproj = cv2.calcBackProject([hsv], [0,1], hist, [0,180,0,256], 1)
    _, mask = cv2.threshold(backproj, 50, 255, cv2.THRESH_BINARY)
    return mask

def denoise_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# ===== [2] 경로 및 클래스 정의 =====
root_dir = 'C:/Users/User/SeoulTech/SeoulTech/Term_project/ASL_label/subdir'
classes = ['A', 'B', 'C']
label_map = {'A': 0, 'B': 1, 'C': 2}

# ===== [3] 손 색상 히스토그램 불러오기 =====
hand_hist = np.load('hand_histogram.npy')

# ===== [4] 데이터셋 생성 =====
features = []
labels = []

for cls in classes:
    folder = os.path.join(root_dir, cls)
    for fname in os.listdir(folder):
        if not fname.lower().endswith(('.jpg', '.png')):
            continue
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is None:
            continue

        mask = histogram_backprojection(img, hand_hist)
        mask = denoise_mask(mask)
        edges = cv2.Canny(mask, 50, 150)
        skel = morphological_skeleton(edges)
        feat = extract_features(skel)

        features.append(feat)
        labels.append([label_map[cls]])  # OpenCV expects shape (N,1)

# ===== [5] 저장 =====
X_train = np.array(features, dtype=np.float32)
y_train = np.array(labels, dtype=np.int32)

np.save('opencv_knn_features.npy', X_train)
np.save('opencv_knn_labels.npy', y_train)

print(f"✅ 저장 완료: 총 샘플 수 = {X_train.shape[0]}")
print("파일 생성: opencv_knn_features.npy, opencv_knn_labels.npy")
