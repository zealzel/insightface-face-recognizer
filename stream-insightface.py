import cv2
import time
import numpy as np
import os
from flask import Flask, Response, render_template_string, request, jsonify
import insightface
from insightface.app import FaceAnalysis
from face_db_sqlite import FaceDatabase
import base64

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


@app.route("/register")
def register():
    html = """
    <html>
        <head>
            <title>新增人臉註冊</title>
            <script>
                let video;
                let canvas;
                let context;
                let capturedImages = [];
                const maxCaptures = 12;
                // 約10秒內捕捉12張照片，每張間隔約833毫秒
                const frameInterval = 833;
                let captureCount = 0;

                function startRegistration() {
                    capturedImages = [];
                    captureCount = 0;
                    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
                        .then(function(stream) {
                            video.srcObject = stream;
                            video.play();
                            captureFrames();
                        })
                        .catch(function(err) {
                            console.log("Error: " + err);
                        });
                }

                function captureFrames() {
                    if (captureCount < maxCaptures) {
                        setTimeout(function() {
                            context.drawImage(video, 0, 0, canvas.width, canvas.height);
                            let dataURL = canvas.toDataURL('image/jpeg');
                            capturedImages.push(dataURL);
                            captureCount++;
                            captureFrames();
                        }, frameInterval);
                    } else {
                        // 停止影像串流
                        let stream = video.srcObject;
                        let tracks = stream.getTracks();
                        tracks.forEach(track => track.stop());
                        video.srcObject = null;
                        uploadRegistration();
                    }
                }

                function uploadRegistration() {
                    const name = document.getElementById('username').value;
                    if (!name) {
                        alert("請輸入姓名！");
                        return;
                    }
                    fetch('/upload_registration', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ name: name, images: capturedImages })
                    })
                    .then(response => response.json())
                    .then(data => {
                        alert(data.message);
                    })
                    .catch(error => {
                        console.error('Error:', error);
                    });
                }

                window.onload = function() {
                    video = document.getElementById('video');
                    canvas = document.getElementById('canvas');
                    context = canvas.getContext('2d');
                }
            </script>
        </head>
        <body>
            <h1>新增人臉註冊</h1>
            <p>請輸入您的姓名，並在10秒內轉動頭部以捕捉12張照片。</p>
            姓名：<input type="text" id="username">
            <button onclick="startRegistration()">開始註冊</button>
            <br>
            <video id="video" width="640" height="480" autoplay style="display:none;"></video>
            <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
        </body>
    </html>
    """
    return render_template_string(html)


@app.route("/upload_registration", methods=["POST"])
def upload_registration():
    data = request.get_json()
    if not data or "name" not in data or "images" not in data:
        return jsonify({"message": "資料不完整"}), 400

    name = data["name"]
    images_data = data["images"]
    embeddings = []

    for img_data in images_data:
        try:
            header, encoded = img_data.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"轉換圖片失敗: {e}")
            continue

        if img is None:
            continue

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = fa.get(rgb_img)
        if faces:
            emb = faces[0].embedding
            norm_val = np.linalg.norm(emb)
            if norm_val == 0:
                continue
            embeddings.append(emb / norm_val)

    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
        norm_val = np.linalg.norm(avg_embedding)
        if norm_val != 0:
            avg_embedding = avg_embedding / norm_val
        # 使用 face_db 物件更新至 SQLite 資料庫 (已在全域載入)
        face_db.insert_or_update_person(name, avg_embedding, count=len(embeddings))

        # 更新 preloaded face_database 字典，便於後續即時比對
        records = face_db.get_all_records()
        global face_database
        face_database = {n: emb for n, emb, c in records}

        return jsonify({"message": "註冊成功"}), 200
    else:
        return jsonify({"message": "未偵測到人臉"}), 400


@app.route("/list_persons", methods=["GET"])
def list_persons_api():
    persons = face_db.list_persons()
    return jsonify({"persons": persons})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
