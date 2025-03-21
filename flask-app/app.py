import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import numpy as np
from flask import Flask, Response, render_template, request, jsonify
from insightface.app import FaceAnalysis
import base64

from PIL import Image, ImageDraw, ImageFont

from face_common import find_face, face_db

try:
    # 請依你的系統修改中文字型路徑，以下是 MacOS 的範例
    chinese_font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 20)
except Exception as e:
    print("PIL load font failed:", e)
    chinese_font = None

# 開啟攝像頭
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise Exception("攝像頭打開失敗！")

# 初始化 insightface 模型 (FaceAnalysis)
# fa = FaceAnalysis()
# fa = FaceAnalysis(providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
fa = FaceAnalysis(
    name="buffalo_s", providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
)
fa.prepare(ctx_id=0, det_size=(640, 640))

app = Flask(__name__)


# 產生串流影像的產生器
def gen_frames():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # 儲存所有要繪製的文字資料，格式：(文字, (x,y)座標)
        texts = []

        # 使用 insightface 偵測人臉，輸入影像使用 BGR 格式
        faces = fa.get(frame)
        for face in faces:
            bbox = face.bbox  # 假設格式為 [x1, y1, x2, y2]
            left = int(bbox[0])
            top = int(bbox[1])
            right = int(bbox[2])
            bottom = int(bbox[3])
            name = find_face(face)
            print("left", left)
            print("top", top)
            print("right", right)
            print("bottom", bottom)
            print("name", name)

            # 在影像上標記偵測到的人臉框 (使用 cv2 畫矩形)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
            )

            # 儲存文字與座標，待會用 PIL 畫中文
            texts.append((name, (left + 6, bottom - 30)))

        # 如果有中文字型，轉換整張影像到 PIL，並用 PIL 畫文字，再轉回 cv2 格式
        if chinese_font is not None and texts:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            for t, pos in texts:
                draw.text(pos, t, font=chinese_font, fill=(255, 255, 255))
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            # 無中文字型則回退使用 cv2.putText() (可能無法正確顯示中文)
            for t, pos in texts:
                cv2.putText(
                    frame,
                    t,
                    pos,
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0,
                    (255, 255, 255),
                    1,
                )

        # 將處理後的影像編碼成 JPEG 格式
        ret2, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


# 主頁面，嵌入串流畫面
@app.route("/")
def index():
    return render_template("index.html")


# 串流路由
@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/register")
def register():
    return render_template("register.html")


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
        return jsonify({"message": "註冊成功"}), 200
    else:
        return jsonify({"message": "未偵測到人臉"}), 400


@app.route("/list_persons", methods=["GET"])
def list_persons_api():
    persons = face_db.list_persons()
    return jsonify({"persons": persons})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
