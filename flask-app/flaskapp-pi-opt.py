import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import numpy as np
from flask import Flask, Response, render_template, request, jsonify
from insightface.app import FaceAnalysis
import base64

from PIL import Image, ImageDraw, ImageFont

from picamera2.previews.null_preview import NullPreview
from face_common import find_face, face_db

# Import Picamera2
from picamera2 import Picamera2

try:
    # Adjust the Chinese font path as needed. This is a macOS example.
    chinese_font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 20)
except Exception as e:
    print("PIL load font failed:", e)
    chinese_font = None

# Initialize Picamera2
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(video_config)
picam2.set_controls({"FrameRate": 10})
picam2.start()

# Initialize insightface model (FaceAnalysis)
fa = FaceAnalysis(providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
# fa.prepare(ctx_id=0, det_size=(640, 640))
# fa.prepare(ctx_id=0, det_size=(320, 320))
fa.prepare(ctx_id=0, det_size=(160, 160))

app = Flask(__name__)


# Generator to yield processed frames
def gen_frames():
    # scale_factor = 0.5  # Resize factor for face detection (adjust as needed)
    scale_factor = 0.25  # Resize factor for face detection (adjust as needed)
    while True:
        # Capture frame using Picamera2
        frame = picam2.capture_array()
        if frame is None:
            continue

        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Create a resized copy for detection
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

        texts = []

        # Use insightface to detect faces in the smaller frame (BGR)
        faces = fa.get(small_frame)
        for face in faces:
            bbox = face.bbox  # Expected format: [x1, y1, x2, y2]
            # Scale bounding box coordinates back to the original frame size
            left = int(bbox[0] / scale_factor)
            top = int(bbox[1] / scale_factor)
            right = int(bbox[2] / scale_factor)
            bottom = int(bbox[3] / scale_factor)
            name = find_face(face)
            # print("left", left)
            # print("top", top)
            # print("right", right)
            # print("bottom", bottom)
            print("name", name)

            # Draw the face bounding box using cv2.rectangle on the original frame
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
            )
            texts.append((name, (left + 6, bottom - 30)))

        # If a Chinese font is available, convert the image to PIL and draw text
        if chinese_font is not None and texts:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            for t, pos in texts:
                draw.text(pos, t, font=chinese_font, fill=(255, 255, 255))
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            # Fallback: use cv2.putText() (may not correctly render Chinese)
            for t, pos in texts:
                cv2.putText(
                    frame, t, pos, cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1
                )

        # Encode the processed frame in JPEG format
        ret2, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


# Main page that embeds the video stream
@app.route("/")
def index():
    return render_template("index.html")


# Video stream route
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
        # Update the SQLite database using face_db (loaded globally)
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

