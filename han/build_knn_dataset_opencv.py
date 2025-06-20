import os
import cv2
import numpy as np
import time

# === 진행 타이머 시작 ===
start_time = time.time()

# ===== [1] 전처리 관련 함수 정의 =====

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
    return np.hstack([hu_log, eps_flat])

def histogram_backprojection(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backproj = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    _, mask = cv2.threshold(backproj, 10, 255, cv2.THRESH_BINARY)
    return mask

# ✅ 사용자 요청에 따라 denoise 단계 변경
def denoise_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# ===== [2] 경로 및 클래스 정의 =====
root_dir = "C:/Users/User/SeoulTech/SeoulTech/Term_project/resized_data"
classes = ["A", "B", "C"]
label_map = {"A": 0, "B": 1, "C": 2}

# ===== [3] 손 색상 히스토그램 불러오기 =====
hand_hist = np.load("hand_histogram.npy")

# ===== [4] 전처리 및 특징 추출 =====
features = []
labels = []

os.makedirs("preprocess_dataset/mask", exist_ok=True)
os.makedirs("preprocess_dataset/edges", exist_ok=True)
os.makedirs("preprocess_dataset/skeleton", exist_ok=True)

total_count = 0

for cls in classes:
    folder = os.path.join(root_dir, cls)
    image_files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]
    print(f"\n📂 클래스 {cls} - 총 {len(image_files)}장 처리 시작")

    for idx, fname in enumerate(image_files, 1):
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ 이미지 로드 실패: {path}")
            continue

        try:
            mask = histogram_backprojection(img, hand_hist)
            mask = denoise_mask(mask)
            edges = cv2.Canny(mask, 50, 150)
            skel = morphological_skeleton(edges)
            feat = extract_features(skel)

            features.append(feat)
            labels.append([label_map[cls]])
            total_count += 1

            fname_no_ext = os.path.splitext(fname)[0]
            cv2.imwrite(f"preprocess_dataset/mask/{cls}_{fname_no_ext}.png", mask)
            cv2.imwrite(f"preprocess_dataset/edges/{cls}_{fname_no_ext}.png", edges)
            cv2.imwrite(f"preprocess_dataset/skeleton/{cls}_{fname_no_ext}.png", skel)

            if idx % 100 == 0 or idx == len(image_files):
                print(f"  └─ 진행 중: {idx}/{len(image_files)}장 완료")

        except Exception as e:
            print(f"❌ 전처리 오류 [{cls}/{fname}]: {e}")

# ===== [5] 저장 =====
X_train = np.array(features, dtype=np.float32)
y_train = np.array(labels, dtype=np.int32)

np.save("opencv_knn_features.npy", X_train)
np.save("opencv_knn_labels.npy", y_train)

elapsed = time.time() - start_time
print(f"\n✅ 저장 완료: 총 샘플 수 = {X_train.shape[0]}")
print("📝 파일 생성: opencv_knn_features.npy, opencv_knn_labels.npy")
print(f"⏱️ 총 소요 시간: {elapsed:.2f}초")
