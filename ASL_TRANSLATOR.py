import os
import cv2
import numpy as np
from collections import deque, Counter

# ----------------------<설정>-----------------------------------
VIDEO_PATH = 'Test_video.mp4'
RESIZE_SHAPE = (256, 256)
K = 5
ENSEMBLE_SIZE = 30
FPS_LIMIT = 60
DELAY = int(1000 / FPS_LIMIT)

CLASSES = ["A", "B", "C", "D", "E", "F"]
IDX_TO_CLASS = {i: cls for i, cls in enumerate(CLASSES)}

#-----------------------<특징데이터 로딩, 학습>--------------------------
X_train = np.load("knn_train_features.npy").astype(np.float32)
y_train = np.load("knn_train_labels.npy").astype(np.int32)

knn = cv2.ml.KNearest_create()
knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

# --------------<세선화 함수>-----------------------------
def morphological_skeleton(bin_img):
    blurred = cv2.GaussianBlur(bin_img, (3, 3), 0)
    _, bin_img = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
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
    dedup = []
    for pt in fingertips:
        if all(np.linalg.norm(np.array(pt) - np.array(p)) > 10 for p in dedup):
            dedup.append(pt)

    dedup = sorted(dedup, key=lambda p: (p[1], p[0]))

    return dedup

# --------------<휴모멘트 + 끝점 추출, 특징 저장>------------------------
def extract_hu_moments_and_fingertips_from_frame(frame, size=(196, 196), max_points=10):
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(img_hsv, lower_skin, upper_skin)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=5)

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    cnt = max(contours, key=cv2.contourArea)
    x, y_, w, h = cv2.boundingRect(cnt)
    cropped = skin_mask[y_:y_+h, x:x+w]
    resized = cv2.resize(cropped, size, interpolation=cv2.INTER_NEAREST)

    skel = morphological_skeleton(resized)
    skel = remove_isolated_white_pixels(skel)
    skel = thicken_skeleton(skel)
    hu = cv2.HuMoments(cv2.moments(skel)).flatten()
    hu_log = (-np.sign(hu) * np.log10(np.abs(hu) + 1e-10)).astype(np.float32)

    fingertips = find_fingertips(resized)
    norm_pts = [(x / size[0], y / size[1]) for (x, y) in fingertips]
    norm_pts = norm_pts[:max_points] + [(0.0, 0.0)] * max(0, max_points - len(norm_pts))
    fingertip_vec = np.array(norm_pts).flatten().astype(np.float32)

    vis = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    for pt in fingertips:
        cv2.circle(vis, pt, 5, (0, 255, 255), -1)

    feature_vec = np.concatenate([hu_log, fingertip_vec])
    return feature_vec, vis

# --------------<영상처리>------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[오류] 영상을 열 수 없습니다: {VIDEO_PATH}")
    exit()

print(f"\n 영상 예측 시작: {os.path.basename(VIDEO_PATH)} | 제한 FPS: {FPS_LIMIT}")

recent_preds = deque(maxlen=ENSEMBLE_SIZE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    feature_vec, debug_vis = extract_hu_moments_and_fingertips_from_frame(frame)
    if feature_vec is not None:
        sample = feature_vec.reshape(1, -1)
        _, result, _, _ = knn.findNearest(sample, k=K)
        predicted_label = int(result[0, 0])
        recent_preds.append(predicted_label)

        most_common_label, _ = Counter(recent_preds).most_common(1)[0]
        predicted_class = IDX_TO_CLASS.get(most_common_label, '?')

        cv2.putText(frame, f"Pred: {predicted_class}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    if debug_vis is not None:
        h, w = frame.shape[:2]
        debug_vis_resized = cv2.resize(debug_vis, (w, h), interpolation=cv2.INTER_NEAREST)
        stacked = np.hstack([frame, debug_vis_resized])
    else:
        stacked = frame

    cv2.imshow("Hu + Fingertip Prediction", stacked)
    if cv2.waitKey(DELAY) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
