import json
import sqlite3
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, base_project_dir, base_workspace_dir):
        """🔍 สร้าง Database Manager สำหรับ Project ID (PID)"""
        self.base_project_dir = base_project_dir
        self.base_workspace_dir = base_workspace_dir

    def get_db_path(self, project_id):
        """🔍 สร้าง path ไปยัง database ของ project"""
        return os.path.join(self.base_project_dir, project_id, "db.db")

    def insert_model(self, project_id, model_name , mode, model_path , validation_metrics):
        """✅ เพิ่มโมเดลใหม่ลงใน database และคืนค่า `model_id`"""
        db_path = self.get_db_path(project_id)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # ✅ สร้างตารางถ้ายังไม่มี
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            model_id TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            mode TEXT NOT NULL,
            model_path TEXT NOT NULL,
            validation_metrics TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # ✅ สร้าง model_id ด้วย modelname_DDMMYY_HHMMSS
        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
        model_id = f"{model_name}_{timestamp}"
        print("Model id: "+model_id)
        # ✅ แปลง metrics เป็น JSON string (ถ้ามี)
        metrics_json = json.dumps(validation_metrics, ensure_ascii=False) if validation_metrics else None
        
        # ✅ แทรกข้อมูลโมเดลใหม่
        cursor.execute("""
            INSERT INTO models (model_id, model_name, mode, model_path, validation_metrics)
            VALUES (?, ?, ?, ?, ?)
        """, (model_id, model_name, mode, model_path, metrics_json))

        conn.commit()
        conn.close()

        return model_id
    def insert_prediction(self, project_id, image_name, model_name, predict_result, prediction_data):
        """
        ✅ บันทึกผลการพยากรณ์ลงในตาราง predict
        """
        db_path = self.get_db_path(project_id)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # ✅ สร้างตาราง predict ถ้ายังไม่มี
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predict (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            project_id TEXT NOT NULL,
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
            INSERT INTO predict (image_name, project_id, model_name, predict_result, prediction_json)
            VALUES (?, ?, ?, ?, ?)
        """, (image_name, project_id, model_name, predict_result, prediction_json))

        conn.commit()
        conn.close()