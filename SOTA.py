import cv2
import numpy as np
from collections import deque, Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt

# ----------------------<설정>-----------------------------------
VIDEO_PATH = 'ReTest6.mp4'
RESIZE_SHAPE = (196, 196)
FPS_LIMIT = 60
DELAY = int(1000 / FPS_LIMIT)
ENSEMBLE_SIZE = 30

CLASSES = ["A", "B", "C", "D", "E", "F"]
IDX_TO_CLASS = {i: cls for i, cls in enumerate(CLASSES)}

# ----------------------<학습 데이터 로딩>-----------------------
X_train = np.load("sota_train_features.npy")
y_train = np.load("sota_train_labels.npy")

# ----------------------<피처 추출 함수>------------------------
def extract_sota_feature_from_frame(frame):
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    img_ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    mask_hsv = cv2.inRange(img_hsv, (0, 30, 60), (30, 150, 255))
    mask_ycc = cv2.inRange(img_ycc, (0, 131, 80), (255, 185, 135))
    skin_mask = cv2.bitwise_or(mask_hsv, mask_ycc)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=3)

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    x, y_, w, h = cv2.boundingRect(cnt)
    cropped = skin_mask[y_:y_ + h, x:x + w]
    if cropped.size == 0:
        return None
    resized = cv2.resize(cropped, RESIZE_SHAPE, interpolation=cv2.INTER_NEAREST)

    hu = cv2.HuMoments(cv2.moments(resized)).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    ratio = np.array([h / w], dtype=np.float32)

    final_feature = np.concatenate([hu_log.astype(np.float32), ratio])
    return final_feature

# ----------------------<유클리디안 최근접 분류>------------------------
def predict_sota_knn(sample_feature, X_train, y_train):
    dists = np.linalg.norm(X_train - sample_feature, axis=1)
    return y_train[np.argmin(dists)]

# ----------------------<유사도 테이블 시각화>------------------------
def show_similarity_table(sample_feature, X_train, y_train, class_map):
    dists = np.linalg.norm(X_train - sample_feature, axis=1)

    class_min_dists = {}
    for cls in np.unique(y_train):
        cls_indices = np.where(y_train == cls)[0]
        cls_dists = dists[cls_indices]
        min_dist = np.min(cls_dists)
        class_min_dists[cls] = min_dist

    sorted_items = sorted(class_min_dists.items(), key=lambda x: x[1])
    labels = [class_map[cls] for cls, _ in sorted_items]
    values = [dist for _, dist in sorted_items]

    fig, ax = plt.subplots(figsize=(4, len(labels) * 0.5 + 1))
    ax.axis('off')
    table_data = [["Result values"]] + [[f"{lbl} - {val:.14f}"] for lbl, val in zip(labels, values)]
    table = ax.table(cellText=table_data, colWidths=[1.8], loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    table[1, 0].set_facecolor('#d9d9d9')
    plt.tight_layout()
    plt.show()

# ----------------------<영상 처리>-----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[오류] 영상을 열 수 없습니다: {VIDEO_PATH}")
    exit()

print(f"[예측 시작] {VIDEO_PATH} | {FPS_LIMIT}fps")
recent_preds = deque(maxlen=ENSEMBLE_SIZE)


last_feat = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    feat = extract_sota_feature_from_frame(frame)
    if feat is not None:
        last_feat = feat  # 마지막 유효한 특징 저장
        pred = predict_sota_knn(feat, X_train, y_train)
        recent_preds.append(pred)
        mode_pred = Counter(recent_preds).most_common(1)[0][0]
        cls_name = IDX_TO_CLASS.get(mode_pred, '?')

        cv2.putText(frame, f"Pred: {cls_name}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("SOTA Prediction", frame)
    if cv2.waitKey(DELAY) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

