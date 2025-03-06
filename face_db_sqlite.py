import sqlite3
import pickle
import numpy as np
import cv2
import insightface
import os

# DB_PATH = os.path.join(os.path.dirname(__file__), "face_database.db")
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "face_database.db"))
# DB_PATH = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "face_database_buffalo_s.db")
# )
print("DB_PATH", DB_PATH)


class FaceDatabase:
    def __init__(self):
        """
        初始化資料庫連線，並建立資料表（若不存在則自動建立）。
        資料表包含欄位：name (人名, 作為 PRIMARY KEY)、embedding (人臉向量, BLOB 儲存) 與 count (累計張數)
        使用 check_same_thread=False 允許跨執行緒使用該連線。
        """
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.create_table()

    def create_table(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS face_database (
                name TEXT PRIMARY KEY,
                embedding BLOB,
                count INTEGER
            )
        """)
        self.conn.commit()

    def insert_or_update_person(self, name, embedding, count=1):
        """
        新增或更新資料庫中的人臉紀錄
        如果該人已存在，則以增量更新的方式計算新的平均向量：
            new_avg = (old_avg * old_count + new_embedding * new_count) / (old_count + new_count)
        否則直接插入新資料。
        :param name: 人名 (字串)
        :param embedding: numpy array 格式的人臉向量 (假設已正規化)
        :param count: 本次更新包含的樣本數 (預設 1)
        """
        cur = self.conn.cursor()
        cur.execute(
            "SELECT embedding, count FROM face_database WHERE name = ?", (name,)
        )
        row = cur.fetchone()
        if row:
            # 已存在，更新平均向量與 count
            old_embedding_blob, old_count = row
            old_embedding = pickle.loads(old_embedding_blob)
            new_count = old_count + count
            new_embedding = (old_embedding * old_count + embedding * count) / new_count
            cur.execute(
                "UPDATE face_database SET embedding = ?, count = ? WHERE name = ?",
                (pickle.dumps(new_embedding), new_count, name),
            )
        else:
            # 新增記錄
            cur.execute(
                "INSERT INTO face_database (name, embedding, count) VALUES (?, ?, ?)",
                (name, pickle.dumps(embedding), count),
            )
        self.conn.commit()

    def get_person_embedding(self, name):
        """
        根據人名取得對應的嵌入向量，若不存在則回傳 None
        """
        cur = self.conn.cursor()
        cur.execute("SELECT embedding FROM face_database WHERE name = ?", (name,))
        row = cur.fetchone()
        if row:
            embedding_blob = row[0]
            return pickle.loads(embedding_blob)
        return None

    def get_all_records(self):
        """
        取得資料庫中所有紀錄，每筆紀錄包含 (name, embedding, count)
        """
        cur = self.conn.cursor()
        cur.execute("SELECT name, embedding, count FROM face_database")
        records = []
        for row in cur.fetchall():
            name, embedding_blob, count = row
            embedding = pickle.loads(embedding_blob)
            records.append((name, embedding, count))
        return records

    def add_person_from_folder(self, name, folder_path):
        """
        遍歷指定的資料夾，提取所有圖片的人臉特徵，
        並計算平均向量後以增量更新方式儲存到資料庫
        :param name: 該人的姓名
        :param folder_path: 包含該人照片的資料夾路徑
        """
        # 初始化 insightface 模型，這裡僅做一次初始化
        # fa = insightface.app.FaceAnalysis()
        fa = insightface.app.FaceAnalysis(
            name="buffalo_s",
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        fa.prepare(ctx_id=0, det_size=(640, 640))

        embeddings = []
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"讀取失敗: {img_path}")
                continue
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = fa.get(rgb_img)
            if faces:
                # 假設每張圖片只取第一個人臉
                emb = faces[0].embedding
                norm = np.linalg.norm(emb)
                if norm == 0:
                    continue
                embeddings.append(emb / norm)
            else:
                print(f"未檢測到人臉: {img_path}")

        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(avg_embedding)
            if norm != 0:
                avg_embedding = avg_embedding / norm
            # 以該人的樣本數作為更新的 count
            self.insert_or_update_person(name, avg_embedding, count=len(embeddings))
            print(f"已成功新增/更新 {name} 的資料，樣本數：{len(embeddings)}")
        else:
            print(f"{name} 的資料夾中未找到有效的人臉圖片")

    def add_multiple_person_from_folder(self, root_folder):
        """
        遍歷指定的根資料夾，針對每個子資料夾（以子資料夾名稱視為人名）呼叫 add_person_from_folder，
        實現批次新增/更新多筆人臉資料。
        :param root_folder: 根資料夾，內含多個子資料夾，每個子資料夾皆代表一個人
        """
        for person in os.listdir(root_folder):
            person_folder = os.path.join(root_folder, person)
            if os.path.isdir(person_folder):
                print(f"處理 {person} 的資料...")
                self.add_person_from_folder(person, person_folder)

    def list_persons(self):
        """
        取得資料庫中所有人員的名稱清單。
        :return: 一個包含所有人名的列表。
        """
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM face_database")
        names = [row[0] for row in cur.fetchall()]
        return names

    def close(self):
        self.conn.close()


# 測試用例
if __name__ == "__main__":
    db = FaceDatabase()

    # 模擬一個隨機人臉向量 (這裡使用 512 維向量，實際維度依模型而定)
    # embedding1 = np.random.rand(512)
    # embedding1 = embedding1 / np.linalg.norm(embedding1)
    # db.insert_or_update_person("Alice", embedding1)
    #
    # # 模擬新增同一個人的另一筆人臉資料
    # embedding2 = np.random.rand(512)
    # embedding2 = embedding2 / np.linalg.norm(embedding2)
    # db.insert_or_update_person("Alice", embedding2)
    #
    # # 模擬新增另一個人
    # embedding3 = np.random.rand(512)
    # embedding3 = embedding3 / np.linalg.norm(embedding3)
    # db.insert_or_update_person("Bob", embedding3)
    #
    records = db.get_all_records()
    for rec in records:
        print(f"姓名: {rec[0]}, 總張數: {rec[2]}")

    root_folder = (
        "/Users/zealzel/Documents/Codes/Current/ai/machine-vision/face_database_images"
    )
    # db.add_multiple_person_from_folder(root_folder)

    # 顯示資料庫中所有人員名稱
    names = db.list_persons()
    print("資料庫中所有人員：", names)

    # db.close()
