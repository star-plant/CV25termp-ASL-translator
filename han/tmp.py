import cv2
import numpy as np
import glob
import os
from sklearn.neighbors import KNeighborsClassifier

# — (1) 이전 코드에서 정의한 함수들 임포트 또는 복사 —
# histogram_backprojection, denoise_mask,
# morphological_skeleton, extract_features

def histogram_backprojection(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backproj = cv2.calcBackProject([hsv], [0,1], hist, [0,180,0,256], 1)
    _, mask = cv2.threshold(backproj, 50, 255, cv2.THRESH_BINARY)
    return mask

def denoise_mask(mask, kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def morphological_skeleton(bin_img):
    skel = np.zeros(bin_img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
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
    return np.hstack([hu_log, eps_flat])

# — (2) 학습 모델 및 히스토그램 로드 —
train_features = np.load('train_features.npy')
train_labels   = np.load('train_labels.npy')
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_features, train_labels)

hand_hist = np.load('hand_histogram.npy')

# — (3) 준비된 이미지 폴더에서 파일 경로 목록 가져오기 —
# train_images/A/*.png, B/*.png, C/*.png ...
image_paths = glob.glob('C:/Users/User/SeoulTech/SeoulTech/Term_project/ASL_label/subdir/*/*.jpg')



results = []
for img_path in image_paths:
    frame = cv2.imread(img_path)
    mask  = histogram_backprojection(frame, hand_hist)
    mask  = denoise_mask(mask, kernel_size=5)
    edges = cv2.Canny(mask, 50, 150)
    skel  = morphological_skeleton(edges)
    feat  = extract_features(skel).reshape(1, -1)
    pred  = knn.predict(feat)[0]
    true_label = os.path.basename(os.path.dirname(img_path))
    results.append((img_path, true_label, pred))

    # 시각화
    color = (0, 255, 0) if pred == true_label else (0, 0, 255)  # 맞으면 초록, 틀리면 빨강
    text = f'GT: {true_label} | Pred: {pred}'
    display = frame.copy()
    cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Prediction Result', display)

    key = cv2.waitKey(0)  # 키 입력 대기
    if key == 27:  # ESC 누르면 종료
        break

cv2.destroyAllWindows()

# 통계 요약
correct = sum(1 for _, t, p in results if t == p)
print(f"\n총 이미지: {len(results)}, 정확도: {correct}/{len(results)} = {correct/len(results):.2%}")
