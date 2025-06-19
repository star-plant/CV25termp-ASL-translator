import cv2
import numpy as np
from collections import deque, Counter


# === 전처리 함수 ===
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
    return np.hstack([hu_log, eps_flat])


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
    "C:/Users/User/SeoulTech/SeoulTech/Term_project/Test.mp4"  # <- 여기 경로 변경 가능
)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"[오류] 동영상을 열 수 없습니다: {video_path}")
    exit()

print("🎥 영상 분석 중... ESC 키로 종료")

# 최근 예측값 저장 버퍼
prediction_queue = deque(maxlen=9)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # 손 분리 → 윤곽 → 스켈레톤 → 특징 추출
        mask = histogram_backprojection(frame, hand_hist)
        mask = denoise_mask(mask)
        edges = cv2.Canny(mask, 50, 150)
        skel = zhangSuen(edges)
        feat = extract_features(skel).reshape(1, -1).astype(np.float32)

        # 예측
        ret, result, neighbours, dist = knn.findNearest(feat, k=3)
        pred = int(result[0][0])
        prediction_queue.append(pred)

        # 최빈값(가장 많이 나온 예측값) 사용
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
    cv2.imshow("Video Hand Sign Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
