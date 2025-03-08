import sys
import os

# 將專案根目錄加入 sys.path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from flask import Flask, request, jsonify
from flask_cors import CORS
from face_db_sqlite import FaceDatabase
import base64

# 初始化 insightface 模型
# fa = FaceAnalysis()
# fa = FaceAnalysis(providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
fa = FaceAnalysis(
    name="buffalo_s", providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
)
fa.prepare(ctx_id=0, det_size=(640, 640))


app = Flask(__name__)
CORS(app)
print("cors!")

# 從資料庫中動態取得最新的已註冊人臉向量 (參考 face_db_sqlite.py)
face_db = FaceDatabase()
records = face_db.get_all_records()  # 預期格式：[(name, embedding, count), ...]

# 假設存入的 embedding 已是 numpy array 格式
face_database = {name: embedding for name, embedding, count in records}


@app.route("/recognize", methods=["POST"])
def recognize():
    print("recgg")
    data = request.get_json()
    image_data = data["image"]
    width = data["width"]
    height = data["height"]

    # 解碼圖片
    image_bytes = base64.b64decode(image_data.split(",")[1])
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 調整圖片大小
    img = cv2.resize(img, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        return jsonify({"error": "Image decoding failed"}), 400

    # 使用 insightface 進行人臉偵測
    faces = fa.get(img)
    recognized_data = []
    threshold = 0.5

    for face in faces:
        # 取得 bounding box，格式為 [x1, y1, x2, y2]
        bbox = face.bbox
        left, top, right, bottom = (
            int(bbox[0]),
            int(bbox[1]),
            int(bbox[2]),
            int(bbox[3]),
        )
        face_embedding = face.embedding
        norm = np.linalg.norm(face_embedding)
        if norm == 0:
            continue
        embedding = face_embedding / norm

        # 與 SQLite 資料庫比對 (使用從資料庫取得的 vector)
        best_match = "Unknown"
        best_score = -1
        for person, db_embedding in face_database.items():
            score = np.dot(embedding, db_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(db_embedding)
            )
            if score > best_score:
                best_score = score
                best_match = person
        if best_score < threshold:
            best_name = "Unknown"
        else:
            best_name = best_match

        # 當前將辨識結果存入 list
        recognized_data.append(
            {
                "name": best_name,
                "location": {
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "left": left,
                },
            }
        )

    # 回傳辨識結果及標記後的圖片（含 data URI 前置字串）
    return jsonify({"recognizedData": recognized_data})


# 新增註冊路由
@app.route("/upload_registration", methods=["POST"])
def upload_registration():
    data = request.get_json()
    if not data or "name" not in data or "images" not in data:
        return jsonify({"message": "資料不完整"}), 400

    name = data["name"]
    images_data = data["images"]
    embeddings = []

    print("name", name)
    print("images_data", images_data)

    print("len(images_data)", len(images_data))
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
            print("img is Empty!")
            continue

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = fa.get(rgb_img)
        if faces:
            emb = faces[0].embedding
            norm_val = np.linalg.norm(emb)
            if norm_val == 0:
                continue
            embeddings.append(emb / norm_val)

    print("embeddings", embeddings)
    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
        norm_val = np.linalg.norm(avg_embedding)
        if norm_val != 0:
            avg_embedding = avg_embedding / norm_val
        # 將更新寫入 SQLite 資料庫（假設 face_db 有此方法）
        face_db.insert_or_update_person(name, avg_embedding, count=len(embeddings))
        # 更新 preloaded face_database 字典，方便後續辨識
        records = face_db.get_all_records()
        global face_database
        face_database = {n: emb for n, emb, c in records}
        return jsonify({"message": "註冊成功"}), 200
    else:
        return jsonify({"message": "未偵測到人臉"}), 400


if __name__ == "__main__":
    print("start")
    app.run(host="0.0.0.0", port=5000)
