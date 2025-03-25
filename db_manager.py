import json
import sqlite3
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, base_project_dir, base_workspace_dir):
        """🔍 สร้าง Database Manager สำหรับ Project ID (PID)"""
        self.base_project_dir = base_project_dir
        self.base_workspace_dir = base_workspace_dir

    def get_db_path(self, pid):
        """🔍 สร้าง path ไปยัง database ของ project"""
        return os.path.join(self.base_project_dir, pid, "db.db")

    def insert_model(self, pid, mode, model_path):
        """✅ เพิ่มโมเดลใหม่ลงใน database และคืนค่า `model_id`"""
        db_path = self.get_db_path(pid)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # ✅ สร้างตารางถ้ายังไม่มี
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode TEXT NOT NULL,
            model_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # ✅ แทรกข้อมูลโมเดลใหม่
        cursor.execute("INSERT INTO models (mode, model_path) VALUES (?, ?)", (mode, model_path))
        model_id = cursor.lastrowid  # ✅ ดึง `model_id` ที่เพิ่มล่าสุด

        conn.commit()
        conn.close()

        return model_id
    def insert_prediction(self, pid, image_name, model_name, predict_result, prediction_data):
        """
        ✅ บันทึกผลการพยากรณ์ลงในตาราง predict
        """
        db_path = self.get_db_path(pid)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # ✅ สร้างตาราง predict ถ้ายังไม่มี
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predict (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            pid TEXT NOT NULL,
            model_name TEXT NOT NULL,
            predict_result TEXT NOT NULL,
            prediction_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # ✅ แปลง prediction_data เป็น JSON string
        prediction_json = json.dumps(prediction_data, ensure_ascii=False)

        # ✅ บันทึกลง database
        cursor.execute("""
            INSERT INTO predict (image_name, model_name, pid, predict_result, prediction_json)
            VALUES (?, ?, ?, ?, ?)
        """, (image_name, model_name, pid, predict_result, prediction_json))

        conn.commit()
        conn.close()