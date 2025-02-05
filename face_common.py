import cv2
import numpy as np
from insightface.app import FaceAnalysis
from face_db_sqlite import FaceDatabase

# 初始化 insightface 模型 (FaceAnalysis)
fa = FaceAnalysis(providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
fa.prepare(ctx_id=0, det_size=(640, 640))

############################
# 建立人臉資料庫 (利用資料夾中的圖片)
############################

# 從 SQLite 資料庫載入人臉資料庫 (使用 face_db_sqlite.py)
face_db = FaceDatabase()
records = face_db.get_all_records()
print("records", records)
# 轉換成 { name: embedding } 格式供後續比對使用
assert len(records) > 0
# face_database = {name: embedding for name, embedding, count in records}


def get_face_database():
    records = face_db.get_all_records()
    return {name: emb for name, emb, c in records}


def find_face_from_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"讀取失敗: {img_path}")
        return "Unknown"
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    names = []
    faces = fa.get(rgb_img)
    for face in faces:
        bbox = face.bbox  # 假設格式為 [x1, y1, x2, y2]
        left = int(bbox[0])
        top = int(bbox[1])
        right = int(bbox[2])
        bottom = int(bbox[3])
        name = find_face(face)
        # 儲存文字與座標，待會用 PIL 畫中文
        names.append((name, [left, top, right, bottom]))
    return names


def find_face(face):
    # 取得人臉的嵌入向量，並進行正規化
    name = "Unknown"
    embedding = face.embedding
    norm = np.linalg.norm(embedding)
    if norm == 0:
        print("未檢測到人臉")
        return name
    embedding = embedding / norm

    # 與 SQLite 資料庫比對 (使用 preloaded face_database)
    best_match = "Unknown"
    best_score = -1

    current_face_database = get_face_database()

    # for person, db_embedding in face_database.items():
    for person, db_embedding in current_face_database.items():
        score = np.dot(embedding, db_embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(db_embedding)
        )
        if score > best_score:
            best_score = score
            best_match = person
    threshold = 0.5  # 閾值，可根據需求調整
    if best_score > threshold:
        name = best_match
    return name
