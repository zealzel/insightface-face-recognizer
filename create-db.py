import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os

# 猴子補丁（若有 np.int 相關問題）
np.int = int

# 初始化 InsightFace 模型
app = FaceAnalysis(providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# 定義存儲資料庫的資料結構
# 例如使用字典，key 為身份 (name)，value 為人臉向量
face_database = {}


def extract_face_embedding(img_path):
    """
    載入圖片、檢測人臉並提取人臉向量。
    假設圖片中只包含一張人臉。
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"讀取失敗: {img_path}")
        return None
    # 將 BGR 轉為 RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb_img)
    if faces:
        # 取得第一張人臉的識別向量
        return faces[0].embedding
    else:
        print(f"未檢測到人臉: {img_path}")
        return None


# 假設你有一個資料夾，每個子資料夾名稱為該人的姓名，內含多張圖片
database_dir = "./face_database_images"

# 遍歷資料夾建立資料庫
for person_name in os.listdir(database_dir):
    print("person_name:", person_name)
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
        # 可以取平均向量作為該人的代表向量
        avg_embedding = np.mean(embeddings, axis=0)
        face_database[person_name] = avg_embedding
        print(f"建立 {person_name} 資料成功，樣本數：{len(embeddings)}")


# 接下來示範如何對一張新圖片進行比對
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


if __name__ == "__main__":
    # 載入一張待識別的圖片
    test_img_path = "/Users/zealzel/Documents/Codes/Current/ai/image-app/find-face/face_recognition/example-known-multiple/Jensen_Huang/Jensen_Huang4.jpg"
    test_embedding = extract_face_embedding(test_img_path)
    if test_embedding is not None:
        best_match = None
        best_score = -1
        for name, db_embedding in face_database.items():
            score = cosine_similarity(test_embedding, db_embedding)
            print(f"{name} 相似度: {score:.3f}")
            if score > best_score:
                best_score = score
                best_match = name
        # 設定閾值，假設 0.5 為閾值
        if best_score > 0.5:
            print(f"識別結果: {best_match}, 相似度: {best_score:.3f}")
        else:
            print("無匹配結果")
