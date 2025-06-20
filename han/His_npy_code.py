import cv2
import numpy as np

# === (1) 여러 손 이미지 경로 ===
image_paths = [
    'C:/Users/User/SeoulTech/SeoulTech/Term_project/resized_data/A/A (3).jpg',
    'C:/Users/User/SeoulTech/SeoulTech/Term_project/resized_data/B/B (3).jpg',
    'C:/Users/User/SeoulTech/SeoulTech/Term_project/resized_data/C/C (3).jpg',
    'C:/Users/User/SeoulTech/SeoulTech/Term_project/resized_data/A/A (20).jpg',
    'C:/Users/User/SeoulTech/SeoulTech/Term_project/resized_data/B/B (20).jpg',
    'C:/Users/User/SeoulTech/SeoulTech/Term_project/resized_data/C/C (20).jpg',
    # 필요 시 더 추가 가능
]

accum_hist = np.zeros((180, 256), dtype=np.float32)
valid_count = 0  # 유효히 선택된 ROI 수

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"[❌ 오류] 이미지를 불러올 수 없습니다: {path}")
        continue

    print(f"\n🖐️ {path} → 손 영역을 마우스로 드래그한 뒤 Enter (ESC: 취소)")
    roi = cv2.selectROI("Select Hand Region", img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w == 0 or h == 0:
        print("[⚠️ 건너뜀] ROI가 선택되지 않음")
        continue

    roi_img = img[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    accum_hist += hist
    valid_count += 1
    print("✅ 히스토그램 누적 완료")

# === (2) 평균화 및 정규화 ===
if valid_count > 0:
    avg_hist = accum_hist / valid_count
    cv2.normalize(avg_hist, avg_hist, 0, 255, cv2.NORM_MINMAX)
    np.save('hand_histogram.npy', avg_hist)
    print(f"\n🎯 최종 손 색상 히스토그램 저장 완료 (총 {valid_count}개 ROI 사용)")
    print("📁 저장 파일: hand_histogram.npy")
else:
    print("\n❌ 저장 실패: 유효한 손 ROI가 선택되지 않았습니다")
