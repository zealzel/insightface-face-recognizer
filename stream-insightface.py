import cv2
import time
import numpy as np
import os
from flask import Flask, Response, render_template_string
import insightface
from insightface.app import FaceAnalysis
import pickle

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
face_database = {}

# 資料庫檔案名稱，可依需求修改
FACE_DATABASE_FILE = "face_database.pkl"


def save_face_database(db, filename=FACE_DATABASE_FILE):
    """將人臉資料庫透過 pickle 儲存"""
    with open(filename, "wb") as f:
        pickle.dump(db, f)
    print(f"人臉資料庫已儲存至 {filename}")


def load_face_database(filename=FACE_DATABASE_FILE):
    """嘗試從檔案載入人臉資料庫，若不存在則回傳 None"""
    try:
        with open(filename, "rb") as f:
            db = pickle.load(f)
        print(f"從 {filename} 載入人臉資料庫成功")
        return db
    except FileNotFoundError:
        print(f"找不到 {filename}，將建立新的資料庫")
        return None


def build_face_database(database_dir):
    """遍歷資料夾建立人臉資料庫 (每個子資料夾名稱視為人名)"""
    db = {}
    for person_name in os.listdir(database_dir):
        person_folder = os.path.join(database_dir, person_name)
        if not os.path.isdir(person_folder):
            continue
        embeddings = []
        for filename in os.listdir(person_folder):
            img_path = os.path.join(person_folder, filename)
            emb = extract_face_embedding(img_path)
            if emb is not None:
                embeddings.append(emb)
        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(avg_embedding)
            if norm != 0:
                avg_embedding = avg_embedding / norm
            db[person_name] = avg_embedding
            print(f"建立 {person_name} 資料成功，樣本數：{len(embeddings)}")
    return db


def extract_face_embedding(img_path):
    """
    載入圖片、檢測人臉並提取人臉向量。
    假設圖片中只包含一張人臉。
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"讀取失敗: {img_path}")
        return None
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = fa.get(rgb_img)
    if faces:
        emb = faces[0].embedding
        norm = np.linalg.norm(emb)
        if norm == 0:
            return None
        return emb / norm
    else:
        print(f"未檢測到人臉: {img_path}")
        return None


database_dir = (
    "/Users/zealzel/Documents/Codes/Current/ai/machine-vision/face_database_images"
)

face_database = load_face_database()
if face_database is None:
    face_database = build_face_database(database_dir)
    save_face_database(face_database)


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

            # 與資料庫比對
            best_match = "Unknown"
            best_score = -1
            for person, db_embedding in face_database.items():
                score = cosine_similarity(embedding, db_embedding)
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
