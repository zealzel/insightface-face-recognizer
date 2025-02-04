import cv2
import numpy as np
from flask import Flask, request, jsonify
import insightface
from insightface.app import FaceAnalysis

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

    for face in faces:
        bbox = face.bbox  # bbox 格式為 [x1, y1, x2, y2]
        left = int(bbox[0])
        top = int(bbox[1])
        right = int(bbox[2])
        bottom = int(bbox[3])

        # 若有比對資料庫，可在此加入比較邏輯，目前直接回傳 "Unknown"
        recognized_data.append(
            {
                "name": "Unknown",
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

