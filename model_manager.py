
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
from tkinter import Image

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from db_manager import DatabaseManager  # เรียกใช้ SQLite Manager


MODEL_FOLDER = './models'
IMAGE_FOLDER = './images'
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)
class MLModelManager:
    def __init__(self, model_folder=MODEL_FOLDER, image_folder=IMAGE_FOLDER):
        self.model_folder = model_folder
        self.image_folder = image_folder
        self.current_model = None

    def save_model(self, model_path):
        """บันทึกโมเดลที่ถูกเทรนแล้วไปยังโฟลเดอร์ `./models` โดยใช้ absolute path ที่ถูกต้อง"""

        # ✅ ตรวจสอบว่าไฟล์มีอยู่จริง
        model_path = Path(model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"❌ Model not found at {model_path}")

        # ✅ กำหนดชื่อไฟล์ใหม่พร้อม timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_model_name = f"model_{timestamp}.pt"

        # ✅ กำหนดโฟลเดอร์เก็บโมเดล
        models_dir = Path(self.model_folder).resolve()  # ✅ ใช้ absolute path
        models_dir.mkdir(parents=True, exist_ok=True)  # ✅ สร้างโฟลเดอร์ถ้ายังไม่มี

        # ✅ สร้าง path ใหม่ที่ถูกต้อง
        saved_path = models_dir / saved_model_name  # ✅ ใช้ Path

        # ✅ ย้ายไฟล์โมเดลไปยังตำแหน่งที่ถูกต้อง
        model_path.rename(saved_path)

        # ✅ ใช้ `.as_posix()` เพื่อให้ได้ path แบบ Unix (ใช้ `/` แทน `\`)
        final_path = saved_path.resolve().as_posix()
        
        print(f"✅ Model saved to: {final_path}")

        # ✅ คืนค่า absolute path ที่ถูกต้อง
        return final_path
        
    def load_model(self, model_name):
        """
        Load a specific model from the ./models folder.

        Args:
            model_name (str): The name of the model file to load.

        Returns:
            torch.nn.Module: The loaded PyTorch model.
        """
        model_path = os.path.join(self.model_folder, model_name).replace("\\", "/")
        
        # ✅ ตรวจสอบว่าไฟล์โมเดลมีอยู่จริง
        if not os.path.exists(model_path):
            print(f"Model {model_name} not found in {self.model_folder}.")
            raise FileNotFoundError(f"Model {model_name} not found in {self.model_folder}.")

        try:
            print(f"Loading model: {model_path}")
            # ✅ เลือกอุปกรณ์ (GPU ถ้ามี)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.current_model = YOLO(model_path).to(device)  # Use Ultralytics' YOLO to load the model
            print(f"Model loaded successfully from {model_path}.")
            return self.current_model
        except Exception as e:
            raise RuntimeError(f"Error loading model {model_name}: {str(e)}")

    def train_model(self, mode):
        """Train the YOLO model using yolo.py with specific mode (detect, segment, classify)."""
        try:
            if mode not in ["detect", "segment", "classify"]:
                return {"status": "error", "message": f"Invalid mode '{mode}'. Use: detect, segment, classify"}

            print(f"🚀 Starting YOLO {mode} training...")

            db = DatabaseManager()

            # ✅ เริ่ม Training โดยใช้ yolo.py
            result = subprocess.run(
                ['python', 'yolo.py', '--mode', mode, '--task', 'train'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8"
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            print(f"📝 Raw Output from yolo.py:\n{stdout}")

            if result.returncode != 0:
                return {"status": "error", "message": "YOLO training failed", "details": stderr}

            # ✅ ค้นหา JSON บรรทัดสุดท้าย
            json_lines = [line for line in stdout.splitlines() if line.strip().startswith('{') and line.strip().endswith('}')]
            last_json_line = json_lines[-1] if json_lines else None

            # ✅ หา "Results saved to ..." จาก stdout
            match = re.search(r"Results saved to (runs[\/\\]\S+)", stdout)
            save_dir = match.group(1) if match else None

            # ✅ ถ้า save_dir ยังหาไม่ได้ ให้ใช้ JSON
            train_results = json.loads(last_json_line) if last_json_line else {}
            if not save_dir and "result_dir" in train_results:
                save_dir = train_results["result_dir"]
            
            trained_model_path = os.path.join(save_dir, "weights", "best.pt") if save_dir else None
            
            if trained_model_path and os.path.exists(trained_model_path):
                saved_path = self.save_model(trained_model_path)
                train_results["saved_path"] = saved_path
                # ✅ บันทึกโมเดลลงใน DB
                
            else:
                train_results["saved_path"] = None

            print(f"✅ Training Results:\n{json.dumps(train_results, indent=4)}")
            print("Printing Saved Path: "+saved_path)   
            model_id = db.insert_model(
                project_id=1,
                name=f"{mode}",
                version="v1.0",
                model_type=mode,
                file_path=saved_path if trained_model_path else "N/A"
            )
            print(f"✅ Inserted Model ID: {model_id}")
            # ✅ ปิด DB
            db.close()

            return {"status": "success", "data": train_results}

        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}


    def evaluate_model(self, mode, model_name):
        """Evaluate a specific YOLO model using yolo.py."""

            # ✅ กำหนดให้ชี้ไปที่ `./models` โดยอัตโนมัติ
        model_path = os.path.join(self.model_folder, model_name).replace("\\", "/")

        print(f"Evaluating YOLO {mode} model from: {model_path}")
            
        # ตรวจสอบว่าไฟล์โมเดลมีอยู่จริง
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            raise FileNotFoundError(f"Model not found at {model_path}")

        # เรียกใช้ `yolo.py` พร้อมส่ง mode และ model path
        try:
            result = subprocess.run(
                ['python', 'yolo.py', '--mode', mode, '--task', 'evaluate', '--trained_model_path', model_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            stdout = result.stdout.decode("utf-8").strip()
            stderr = result.stderr.decode("utf-8")

            # Debug: แสดงผลลัพธ์ที่ได้จาก yolo.py
            print(f"Raw Output from yolo.py:\n{stdout}")

            # เช็คว่า subprocess รันสำเร็จหรือไม่
            if result.returncode != 0:
                raise RuntimeError(f"YOLO evaluation failed: {stderr}")

            # ✅ ค้นหา JSON Output ที่ถูกต้อง
            
            json_lines = [line for line in stdout.splitlines() if line.strip().startswith('{') and line.strip().endswith('}')]
            
            if not json_lines:
                raise RuntimeError("Evaluation completed, but could not parse JSON output from yolo.py")

            eval_results = json.loads(json_lines[-1])  # ✅ ใช้ JSON บรรทัดสุดท้ายที่ print ออกมา
            return eval_results

        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            raise RuntimeError(f"An unexpected error occurred during evaluation: {str(e)}")

    def predict_from_path(self, image_name, confidence_threshold=0.4):
        """
        Run inference on an image and filter results by confidence threshold.

        Args:
            image_path (str): Path to the image to be predicted.
            confidence_threshold (float): Minimum confidence score to consider a detection.

        Returns:
            dict: Predictions or an error message.
        """
        if not self.current_model:
            print("⚠️ No model loaded. Please load a model first.")
            raise RuntimeError("⚠️ No model loaded. Please load a model first.")
        
        # ✅ ตรวจสอบ path ให้แน่ใจว่าเป็นรูปแบบ Unix (`/`) แม้จะรันบน Windows
        image_path = os.path.join(self.image_folder, image_name).replace("\\", "/")

        if not os.path.exists(image_path):
            print(f"Image not found at {image_path}")
            raise FileNotFoundError(f"Image not found at {image_path}")

        try:
            print(f"Predicting from Image path: {image_path}")

            # Using GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            results = self.current_model.predict(source=image_path, save=True, device=device)

            # Filter predictions by confidence threshold
            filtered_predictions = []
            for box in results[0].boxes:
                bbox = box.xyxy.tolist()[0]  # Bounding box [x1, y1, x2, y2]
                confidence = box.conf.tolist()[0]  # Confidence score
                class_id = int(box.cls.tolist()[0])  # Class ID

                if confidence >= confidence_threshold:
                    filtered_predictions.append({
                        "bbox": bbox,
                        "confidence": confidence,
                        "class_id": class_id,
                    })

            # 🟢 Debug: แสดงผลลัพธ์
            if not filtered_predictions:
                print("⚠️ No detections found!")

            return {"status": "success", "predictions": filtered_predictions}

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return {"error": f"An error occurred during prediction: {str(e)}"}
    