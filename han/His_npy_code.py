import cv2
import numpy as np

# === (1) 대표 이미지 경로 (A/B/C 각각 1장) ===
image_paths = [
    'C:/Users/User/SeoulTech/SeoulTech/Term_project/ASL_label/subdir/A/A4_jpg.rf.9369f1aa5b6108a0759d030730700afa.jpg',
    'C:/Users/User/SeoulTech/SeoulTech/Term_project/ASL_label/subdir/B/B9_jpg.rf.ecd94e9f60f2a00d60b4536cfd75abc5.jpg',
    'C:/Users/User/SeoulTech/SeoulTech/Term_project/ASL_label/subdir/C/C7_jpg.rf.153d4f0a86e41d21494e749ec943ff30.jpg'
]

accum_hist = np.zeros((180, 256), dtype=np.float32)

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"[오류] 이미지를 불러올 수 없습니다: {path}")
        continue

    print(f"🖐️ {path} → 손 영역을 마우스로 드래그한 뒤 Enter (ESC: 취소)")
    roi = cv2.selectROI("Select Hand Region", img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w == 0 or h == 0:
        print("[건너뜀] ROI가 선택되지 않음")
        continue

    roi_img = img[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    accum_hist += hist

# === (2) 정규화 및 저장 ===
cv2.normalize(accum_hist, accum_hist, 0, 255, cv2.NORM_MINMAX)
np.save('hand_histogram.npy', accum_hist)
print("✅ 손 색상 히스토그램 저장 완료: hand_histogram.npy")
