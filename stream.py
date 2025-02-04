import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# 猴子補丁：解決 np.int 別名問題
np.int = int

# 初始化 FaceAnalysis，這裡使用 CoreMLExecutionProvider 和 CPUExecutionProvidee
app = FaceAnalysis(providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
# 設定檢測尺寸（根據需要可以調整）
app.prepare(ctx_id=0, det_size=(640, 640))

# 開啟攝影機
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 翻轉影像，便於鏡像顯示
    frame = cv2.flip(frame, 1)
    # 將 BGR 影像轉換為 RGB（insightface 要求RGB格式）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 進行人臉檢測
    faces = app.get(rgb_frame)

    # 在 RGB 影像上繪製人臉邊框
    # app.draw_on() 會將邊框、關鍵點等信息畫在影像上
    rimg = app.draw_on(rgb_frame, faces)
    # 轉回 BGR 格式以便 OpenCV 顯示
    rimg = cv2.cvtColor(rimg, cv2.COLOR_RGB2BGR)

    cv2.imshow("Face Recognition", rimg)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
