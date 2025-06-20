import cv2
import numpy as np
from collections import deque, Counter

import os  # ← 폴더 생성용 추가

# ===== 저장 디렉토리 생성 =====
os.makedirs("preprocess_test/mask", exist_ok=True)
os.makedirs("preprocess_test/edges", exist_ok=True)
os.makedirs("preprocess_test/skeleton", exist_ok=True)

frame_count = 0  # 프레임 번호 추적
save_interval = 10  # N프레임마다 저장


# # === 전처리 함수 ===
# def zhangSuen(bin_img):
#     img = (bin_img > 0).astype(np.uint8)  # binary 0/1로 변환

#     def neighbours(x, y, image):
#         x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
#         return [
#             image[x_1][y],
#             image[x_1][y1],
#             image[x][y1],
#             image[x1][y1],
#             image[x1][y],
#             image[x1][y_1],
#             image[x][y_1],
#             image[x_1][y_1],
#         ]

#     def transitions(neigh):
#         n = neigh + neigh[0:1]
#         return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

#     changing1 = changing2 = 1
#     while changing1 or changing2:
#         changing1 = []
#         rows, cols = img.shape
#         for x in range(1, rows - 1):
#             for y in range(1, cols - 1):
#                 P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, img)
#                 if (
#                     img[x][y] == 1
#                     and 2 <= sum(n) <= 6
#                     and transitions(n) == 1
#                     and P2 * P4 * P6 == 0
#                     and P4 * P6 * P8 == 0
#                 ):
#                     changing1.append((x, y))
#         for x, y in changing1:
#             img[x][y] = 0

#         changing2 = []
#         for x in range(1, rows - 1):
#             for y in range(1, cols - 1):
#                 P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, img)
#                 if (
#                     img[x][y] == 1
#                     and 2 <= sum(n) <= 6
#                     and transitions(n) == 1
#                     and P2 * P4 * P8 == 0
#                     and P2 * P6 * P8 == 0
#                 ):
#                     changing2.append((x, y))
#         for x, y in changing2:
#             img[x][y] = 0

#     skel = img * 255  # OpenCV 호환을 위해

#     return skel
## 넘파이 버젼으로 최적화를 시도 했지만, 여전히 느리다.
# def zhangSuen(bin_img):
#     img = (bin_img > 0).astype(np.uint8)

#     def transitions(neigh):
#         n = np.r_[neigh, neigh[0]]
#         return np.sum((n[:-1] == 0) & (n[1:] == 1))

#     kernel = [(-1, 0), (-1, 1), (0, 1), (1, 1),
#               (1, 0), (1, -1), (0, -1), (-1, -1)]

#     changing = True
#     while changing:
#         changing = False
#         for step in range(2):
#             rows, cols = img.shape
#             markers = []

#             for x in range(1, rows - 1):
#                 for y in range(1, cols - 1):
#                     if img[x, y] != 1:
#                         continue

#                     neighbors = np.array([img[x + dx, y + dy] for dx, dy in kernel])
#                     total = np.sum(neighbors)
#                     t = transitions(neighbors)

#                     if step == 0:
#                         conds = [
#                             2 <= total <= 6,
#                             t == 1,
#                             neighbors[0] * neighbors[2] * neighbors[4] == 0,
#                             neighbors[2] * neighbors[4] * neighbors[6] == 0,
#                         ]
#                     else:
#                         conds = [
#                             2 <= total <= 6,
#                             t == 1,
#                             neighbors[0] * neighbors[2] * neighbors[6] == 0,
#                             neighbors[0] * neighbors[4] * neighbors[6] == 0,
#                         ]

#                     if all(conds):
#                         markers.append((x, y))

#             for x, y in markers:
#                 img[x, y] = 0
#                 changing = True

#     skel = img * 255  # ← 원래 함수와 동일하게 맞춤
#     return skel


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


# === KNN 모델 로드 ===
label_map = {0: "A", 1: "B", 2: "C"}
knn = cv2.ml.KNearest_create()
X_train = np.load("opencv_knn_features.npy").astype(np.float32)
y_train = np.load("opencv_knn_labels.npy").astype(np.int32)
knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

# 손 색상 히스토그램 로드
hand_hist = np.load("hand_histogram.npy")

# === 영상 열기 ===
video_path = (
    "C:/Users/User/SeoulTech/SeoulTech/Term_project/Re_Test2.mp4"  # <- 여기 경로 변경 가능
)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"[오류] 동영상을 열 수 없습니다: {video_path}")
    exit()

print("🎥 영상 분석 중... ESC 키로 종료")

# 최근 예측값 저장 버퍼
prediction_queue = deque(maxlen=5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    try:
        # 전처리
        mask = histogram_backprojection(frame, hand_hist)
        mask = denoise_mask(mask)
        edges = cv2.Canny(mask, 50, 150)
        skel = morphological_skeleton(edges)

        # 저장 (10프레임 간격)
        if frame_count % save_interval == 0:
            fname = f"frame_{frame_count:04d}.png"
            cv2.imwrite(f"preprocess_test/mask/{fname}", mask)
            cv2.imwrite(f"preprocess_test/edges/{fname}", edges)
            cv2.imwrite(f"preprocess_test/skeleton/{fname}", skel)

        # 특징 추출 및 예측
        feat = extract_features(skel).reshape(1, -1).astype(np.float32)
        ret, result, neighbours, dist = knn.findNearest(feat, k=3)
        pred = int(result[0][0])
        prediction_queue.append(pred)

        most_common = Counter(prediction_queue).most_common(1)
        label = label_map.get(most_common[0][0], "?")
    except:
        label = "?"

    # 결과 표시
    cv2.putText(
        frame,
        f"Prediction: {label}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        2,
    )
    # cv2.imshow("Video Hand Sign Prediction", frame)
    # print("Distances to Neighbors:", dist)
    # print("Neighbor Labels:", neighbours)

    # # === 시각화 병합 ===
    vis_frame = cv2.resize(frame.copy(), (320, 240))
    vis_mask = cv2.cvtColor(cv2.resize(mask, (320, 240)), cv2.COLOR_GRAY2BGR)
    vis_edges = cv2.cvtColor(cv2.resize(edges, (320, 240)), cv2.COLOR_GRAY2BGR)
    vis_skel = cv2.cvtColor(cv2.resize(skel, (320, 240)), cv2.COLOR_GRAY2BGR)

    # # 예측 결과 넣기
    cv2.putText(vis_frame, f"Prediction: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # # 영상 병합
    combined = np.hstack([vis_frame, vis_mask, vis_edges, vis_skel])
    cv2.imshow("Frame | Mask | Edges | Skeleton", combined)


    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
