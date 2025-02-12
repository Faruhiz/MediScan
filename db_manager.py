import sqlite3
from datetime import datetime

DB_PATH = "db/medi_db.db"  # เปลี่ยนเป็นตำแหน่งไฟล์ SQLite ของคุณ

class DatabaseManager:
    def __init__(self, db_path=DB_PATH):
        """เชื่อมต่อกับ SQLite Database"""
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
    
    def insert_model(self, project_id, name, version, model_type, file_path):
        """บันทึกโมเดลที่เทรนเสร็จลงใน DB"""
        self.cursor.execute("""
            INSERT INTO models (project_id, name, version, type, file_path)
            VALUES (?, ?, ?, ?, ?)
        """, (project_id, name, version, model_type, file_path))
        self.conn.commit()
        return self.cursor.lastrowid

    def insert_evaluation(self, model_id, project_id, metrics_json):
        """บันทึกผลการ Evaluate Model"""
        self.cursor.execute("""
            INSERT INTO evaluations (model_id, project_id, metrics)
            VALUES (?, ?, ?)
        """, (model_id, project_id, metrics_json))
        self.conn.commit()
        return self.cursor.lastrowid

    def close(self):
        """ปิดการเชื่อมต่อ DB"""
        self.conn.close()
