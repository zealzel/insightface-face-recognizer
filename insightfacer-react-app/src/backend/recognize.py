import cv2
import numpy as np
from flask import Flask, request, jsonify
import insightface
from insightface.app import FaceAnalysis
from face_db_sqlite import FaceDatabase

# 初始化 insightface 模型
fa = FaceAnalysis()
fa.prepare(ctx_id=0, det_size=(640, 640))

app = Flask(__name__)


@app.route("/recognize", methods=["POST"])
def recognize():
    # 檢查是否有上傳 frame 檔案（以 FormData 傳送）
    if "frame" not in request.files:
        return jsonify({"error": "No frame provided"}), 400

    # 讀取上傳的影像
    file = request.files["frame"]
    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Image decoding failed"}), 400

    # 使用 insightface 進行人臉偵測
    faces = fa.get(img)
    recognized_data = []
    threshold = 0.5

    face_db = FaceDatabase()
    records = face_db.get_all_records()  # 預期格式：[(name, embedding, count), ...]
    # face_database = {name: np.array(embedding) for name, embedding, count in records}
    face_database = {name: embedding for name, embedding, count in records}

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

    return jsonify({"recognizedData": recognized_data})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
