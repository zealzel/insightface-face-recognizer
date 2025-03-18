#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用這支腳本讀取即時攝影機影像，
對每個 frame 執行 insightface 人臉偵測與辨識，
並在畫面上標示出偵測到的人臉與辨識姓名。
使用方式：
    python stream_realtime.py
按 q 鍵可退出影片視窗。
"""

import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis
from face_db_sqlite import FaceDatabase
from concurrent.futures import ThreadPoolExecutor


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
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            face_database[name] = emb
    return face_database


def main():
    cap = cv2.VideoCapture(0)  # 使用攝影機作為影片來源
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # 預設值
    delay = int(1000 / fps)

    executor = ThreadPoolExecutor(max_workers=1)
    detection_future = None
    last_detection_time = time.time()
    detection_interval = 0.2
    faces_detected = []

    # fa = FaceAnalysis(providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
    fa = FaceAnalysis(
        name="buffalo_s", providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
    )
    fa.prepare(ctx_id=0, det_size=(640, 640))

    face_database = load_face_database()
    if not face_database:
        print("Warning: 資料庫中沒有註冊人臉，所有偵測皆會標記為 'Unknown'。")
    threshold = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        if current_time - last_detection_time >= detection_interval:
            if detection_future is None or detection_future.done():
                detection_future = executor.submit(
                    fa.get, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
                last_detection_time = current_time

        if detection_future is not None and detection_future.done():
            try:
                faces_detected = detection_future.result()
            except Exception as e:
                print("Detection error:", e)
                faces_detected = []

        for face in faces_detected:
            bbox = face.bbox
            left, top, right, bottom = map(int, bbox)
            face_embedding = face.embedding
            norm = np.linalg.norm(face_embedding)
            if norm != 0:
                embedding = face_embedding / norm
            else:
                embedding = face_embedding

            best_score = -1
            best_name = "Unknown"
            for person, db_embedding in face_database.items():
                if embedding.shape != db_embedding.shape:
                    continue
                score = np.dot(embedding, db_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(db_embedding)
                )
                if score > best_score:
                    best_score = score
                    best_name = person
            if best_score < threshold:
                best_name = "Unknown"
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
            if best_name != "Unknown":
                print("person", best_name)

        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
