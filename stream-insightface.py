import cv2
import time
import numpy as np
import os
from flask import Flask, Response, render_template_string
import insightface
from insightface.app import FaceAnalysis
from face_db_sqlite import FaceDatabase

# 開啟攝像頭
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise Exception("攝像頭打開失敗！")

# 初始化 insightface 模型 (FaceAnalysis)
fa = FaceAnalysis()
fa.prepare(ctx_id=0, det_size=(640, 640))

app = Flask(__name__)

############################
# 建立人臉資料庫 (利用資料夾中的圖片)
############################

# 從 SQLite 資料庫載入人臉資料庫 (使用 face_db_sqlite.py)
face_db = FaceDatabase()
records = face_db.get_all_records()
# 轉換成 { name: embedding } 格式供後續比對使用
face_database = {name: embedding for name, embedding, count in records}


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# 產生串流影像的產生器
def gen_frames():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # 使用 insightface 偵測人臉，輸入影像使用 BGR 格式
        faces = fa.get(frame)
        for face in faces:
            bbox = face.bbox  # 假設格式為 [x1, y1, x2, y2]
            left = int(bbox[0])
            top = int(bbox[1])
            right = int(bbox[2])
            bottom = int(bbox[3])

            # 取得人臉的嵌入向量，並進行正規化
            embedding = face.embedding
            norm = np.linalg.norm(embedding)
            if norm == 0:
                continue
            embedding = embedding / norm

            # 與 SQLite 資料庫比對 (使用 preloaded face_database)
            best_match = "Unknown"
            best_score = -1
            for person, db_embedding in face_database.items():
                score = np.dot(embedding, db_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(db_embedding)
                )
                if score > best_score:
                    best_score = score
                    best_match = person
            threshold = 0.5  # 閾值，可根據需求調整
            if best_score < threshold:
                name = "Unknown"
            else:
                name = best_match

            # 在影像上標記偵測到的人臉框與姓名
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1
            )

        # 將處理後的影像編碼成 JPEG 格式
        ret2, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


# 主頁面，嵌入串流畫面
@app.route("/")
def index():
    html = """
    <html>
        <head>
            <title>Face Recognition Stream (insightface)</title>
            <style>
                /* 確保圖片保持原始比例，不會被拉伸 */
                img {
                    max-width: 640px;
                    width: 100%;
                    height: auto;
                    display: block;
                    margin: 0 auto;
                }
            </style>
        </head>
        <body>
            <h1>Face Recognition Stream (insightface)</h1>
            <img src="/video_feed">
        </body>
    </html>
    """
    return render_template_string(html)


# 串流路由
@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
