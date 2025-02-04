import cv2
import time
import numpy as np
import face_recognition
import os
from flask import Flask, Response, render_template_string
from who_lib import (
    save_encodings_to_file,
    load_encodings_from_file,
    get_knowns_encodings_multi,
)


# 載入已知人臉編碼
def load_known_encodings():
    t0 = time.time()
    if os.path.exists("encodings.pkl"):
        known_encodings, known_names = load_encodings_from_file()
    else:
        known_encodings, known_names = get_knowns_encodings_multi("known-multiple")
        save_encodings_to_file(known_encodings, known_names, filename="encodings.pkl")
    t1 = time.time()
    print(f"Time to load encodings: {t1 - t0} seconds")
    return known_encodings, known_names


known_encodings, known_names = load_known_encodings()

# 開啟攝像頭
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise Exception("攝像頭打開失敗！")

app = Flask(__name__)


# 產生串流影像的產生器
def gen_frames():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # 縮小畫面以加速人臉辨識運算
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # 偵測畫面中的人臉位置與編碼
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings
        ):
            threshold = 0.5
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(distances)
            name = "Unknown"
            if distances[best_match_index] < threshold:
                name = known_names[best_match_index]

            # 調整人臉座標至原始畫面尺寸
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # 在畫面上繪製人臉框與姓名
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1
            )

        # 將處理後的畫面編碼成 JPEG 格式
        ret2, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


# 主頁面，嵌入串流畫面
@app.route("/")
def index():
    html = """
    <html>
        <head>
            <title>Face Recognition Stream</title>
        </head>
        <body>
            <h1>Face Recognition Stream</h1>
            <img src="/video_feed" width="640" height="480">
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

