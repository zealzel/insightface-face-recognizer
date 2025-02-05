#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用這支腳本讀取指定影片，
對每個 frame 執行 insightface 人臉偵測與辨識，
並在畫面上標示出偵測到的人臉與辨識姓名。
使用方式：
    python stream.py <video_path>
按 q 鍵可退出影片視窗。
"""

import sys
import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis
from face_db_sqlite import FaceDatabase


def load_face_database():
    """
    從資料庫中取得已註冊人臉記錄，並轉換成 dictionary 格式：
        { name: embedding (numpy array), ... }
    資料庫中的 embedding 以 bytes 形式儲存，採用 np.frombuffer 轉回 numpy array。
    """
    face_db = FaceDatabase()
    records = face_db.get_all_records()
    face_database = {}
    if len(records) > 0:
        for name, emb_bytes, count in records:
            # 這裡假設 embedding 為 np.float32 且長度固定（例如 512、128 或依您模型而定）
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            face_database[name] = emb
    return face_database


def main():
    if len(sys.argv) < 2:
        print("Usage: python stream.py <video_path>")
        sys.exit(1)
    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: 無法開啟影片檔案:", video_path)
        sys.exit(1)

    # 初始化 insightface 模型（使用與 recognize.py 同樣的 providers 設定）
    fa = FaceAnalysis(providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
    fa.prepare(ctx_id=0, det_size=(640, 640))

    # 取得已註冊人臉資料庫
    face_database = load_face_database()
    if not face_database:
        print("Warning: 資料庫中沒有註冊人臉，所有偵測皆會標記為 'Unknown'。")
    threshold = 0.5  # 辨識分數門檻，可依需求調整

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 轉換成 RGB 以供模型分析
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = fa.get(rgb_frame)
        for face in faces:
            # 取得 bounding box
            bbox = face.bbox  # [x1, y1, x2, y2]
            left, top, right, bottom = map(int, bbox)
            # 取得人臉 embedding 並正規化
            face_embedding = face.embedding
            norm = np.linalg.norm(face_embedding)
            if norm != 0:
                embedding = face_embedding / norm
            else:
                embedding = face_embedding

            best_score = -1
            best_name = "Unknown"
            # 對照資料庫中每筆紀錄，比對 cosine 相似度
            for person, db_embedding in face_database.items():
                # 如果 embedding 的 shape 與資料庫中的不一致，則跳過該筆比對
                if embedding.shape != db_embedding.shape:
                    print(
                        f"Warning: embedding shape mismatch for {person}, expected {embedding.shape} but got {db_embedding.shape}"
                    )
                    continue
                # 計算 cosine 相似度
                score = np.dot(embedding, db_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(db_embedding)
                )
                if score > best_score:
                    best_score = score
                    best_name = person
            if best_score < threshold:
                best_name = "Unknown"

            # 畫出 bounding box 與名稱
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame,
                best_name,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
