
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from tkinter import Image

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from db_manager import DatabaseManager  # เรียกใช้ SQLite Manager

# ✅ ใช้ relative path ไปยัง `MedSight_Project`
BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "MedSight_Project"))
BASE_WORKSPACE_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "MediScan", "workspace"))

class MLModelManager:
    def __init__(self):
        self.current_model = None
        self.db = DatabaseManager(BASE_PROJECT_DIR, BASE_WORKSPACE_DIR)
    
    def prepare_workspace(self, pid):
        """🔍 เรียก `prepare_workspace.py` พร้อมส่ง `BASE_PROJECT_DIR` และ `BASE_WORKSPACE_DIR`"""
        
        project_source_path = os.path.join(BASE_PROJECT_DIR, pid)

        # ✅ **เช็คว่าโปรเจคมีอยู่จริงใน `MedSight_Project`**
        if not os.path.exists(project_source_path):
            raise FileNotFoundError(f"❌ Project '{pid}' not found at {project_source_path}")

        print(f"🚀 Running `prepare_workspace.py` for PID: {pid}")

        result = subprocess.run(
            ['python', 'prepare_workspace.py', pid],
            capture_output=True,
            text=True
        )

        # ✅ **แสดงผลลัพธ์จาก `prepare_workspace.py`**
        print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(f"❌ Failed to prepare workspace: {result.stderr}")

        return True

    def save_model(self, pid, mode, result_dir ,model_path):
        """✅ บันทึกโมเดลและอัปเดต Database"""
        
        project_path = os.path.join(BASE_PROJECT_DIR, pid)

        if not os.path.exists(project_path):
            raise FileNotFoundError(f"❌ Project '{pid}' not found at {project_path}")
        
        # ✅ สร้าง model_id จาก Database
        model_id = self.db.insert_model(pid ,mode, model_path)

        # ✅ ตั้งชื่อไฟล์โมเดลใหม่
        model_filename = f"model_{mode}_{model_id}.pt"
        model_dest_path = os.path.join(project_path, model_filename)

        # ✅ เปลี่ยนชื่อโฟลเดอร์ Training Result → `train_{mode}`
        train_output_path = os.path.join(project_path, f"train_{mode}_{model_id}")

        # ✅ ตรวจสอบว่ามีโฟลเดอร์ต้นทางให้คัดลอกไหม
        if not os.path.exists(result_dir):
            raise FileNotFoundError(f"❌ Training output directory '{result_dir}' not found")
        
        # ✅ กำหนด path ปลายทางของ Training Output
        shutil.copytree(result_dir, train_output_path, dirs_exist_ok=True)
        print(f"📂 Copied training results to: {train_output_path}")

        # ✅ คัดลอกไฟล์โมเดลไปยัง MedSight_Project
        if os.path.exists(model_path):
            shutil.copy(model_path, model_dest_path)
            print(f"📂 Copied model file to: {model_dest_path}")
        else:
            print(f"⚠️ Model file not found: {model_path} (Skipping copy)")

        return {
            "status": "success",
            "message": "Training results and model copied successfully",
            "train_output_path": train_output_path,
            "model_path": model_dest_path
        }

        
    def load_model(self, model_name):
        """
        Load a specific model from the ./models folder.

        Args:
            model_name (str): The name of the model file to load.

        Returns:
            torch.nn.Module: The loaded PyTorch model.
        """
        model_path = os.path.join(self.model_folder, model_name).replace("\\", "/")
        
        # ตรวจสอบว่าไฟล์โมเดลมีอยู่จริง
        if not os.path.exists(model_path):
            print(f"Model {model_name} not found in {self.model_folder}.")
            raise FileNotFoundError(f"Model {model_name} not found in {self.model_folder}.")

        try:
            print(f"Loading model: {model_path}")
            # เลือกอุปกรณ์ (GPU ถ้ามี)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.current_model = YOLO(model_path).to(device)  # Use Ultralytics' YOLO to load the model
            print(f"Model loaded successfully from {model_path}.")
            return self.current_model
        except Exception as e:
            raise RuntimeError(f"Error loading model {model_name}: {str(e)}")

    def train_model(self, pid ,mode):
        """Train the YOLO model using yolo.py with specific mode (detect, segment, classify)."""

        if mode not in ["detect", "segment", "classify"]:
            raise ValueError(f"Invalid mode '{mode}'. Use: detect, segment, classify")
        
        # ✅ **เรียก `prepare_workspace.py` ก่อนเทรน**
        self.prepare_workspace(pid)
        
        project_path = os.path.join(BASE_WORKSPACE_DIR, pid)
        data_yaml_path = os.path.join(project_path, "data.yaml")
        print("📂 Project Path:", project_path+"\n Data_yaml_Path: ",data_yaml_path)
        if not os.path.exists(project_path):
            raise FileNotFoundError(f"❌ Project '{pid}' not found at {project_path}")

        if not os.path.exists(data_yaml_path):
            raise FileNotFoundError(f"❌ data.yaml not found in {project_path}")

        print(f"🚀 Starting YOLO {mode} for PID: {pid}")

        try:
            # ✅ เรียก YOLO จริง
            result = subprocess.run(
                ['python', 'yolo.py', '--mode', mode, '--task', 'train','--data', data_yaml_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8"
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            print(f"📝 Raw Output from yolo.py:\n{stdout}")
            
            if result.returncode != 0:
                raise RuntimeError(f"❌ Training failed: {stderr}")

            # ✅ ค้นหา JSON Output ที่เป็นบรรทัดสุดท้าย
            json_lines = [line for line in stdout.splitlines() if line.strip().startswith('{') and line.strip().endswith('}')]
            if not json_lines:
                raise RuntimeError("❌ No valid JSON output found from yolo.py!")

            train_results = json.loads(json_lines[-1])  # ✅ **แปลง JSON String → Dict**
            
            # ✅ Debug: ตรวจสอบว่าค่าถูกต้อง
            print("✅ Parsed JSON from YOLO:", train_results)

            if train_results.get("message") != "Training completed successfully":
                raise RuntimeError(f"❌ Training failed: {train_results.get('error', 'Unknown error')}")

            result_dir = train_results.get("result_dir")
            model_path = train_results.get("model_path")

            if not result_dir or not model_path:
                raise RuntimeError("❌ Training completed but result directory or model path not found.")
            
            # ✅ บันทึกโมเดลและอัปเดต DB
            save_result = self.save_model(pid , mode, result_dir , model_path)
            
            return save_result

        except Exception as e:
            print(f"❌ An unexpected error occurred: {str(e)}")
            return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}


    def evaluate_model(self, mode, model_name,eval_type):
        """Evaluate a specific YOLO model using yolo.py."""
        # ตรวจสอบว่า eval_type ถูกต้องหรือไม่
        if eval_type not in ["val", "test"]:
            raise ValueError("Invalid evaluation type. Use 'val' or 'test'.")
        
        # กำหนดให้ชี้ไปที่ `./models` โดยอัตโนมัติ
        model_path = os.path.join(self.model_folder, model_name).replace("\\", "/")

        print(f"Evaluating YOLO {mode} model from: {model_path} using {eval_type} set")
            
        # ตรวจสอบว่าไฟล์โมเดลมีอยู่จริง
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            raise FileNotFoundError(f"Model not found at {model_path}")

        # เรียกใช้ `yolo.py` พร้อมส่ง mode และ model path
        try:
            result = subprocess.run(
                ['python', 'yolo.py', '--mode', mode, '--task', 'evaluate', '--trained_model_path', model_path,'--eval_type', eval_type],
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

            # ค้นหา JSON Output ที่ถูกต้อง
            
            json_lines = [line for line in stdout.splitlines() if line.strip().startswith('{') and line.strip().endswith('}')]
            
            if not json_lines:
                raise RuntimeError("Evaluation completed, but could not parse JSON output from yolo.py")

            eval_results = json.loads(json_lines[-1])  # ✅ ใช้ JSON บรรทัดสุดท้ายที่ print ออกมา
            return eval_results

        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            raise RuntimeError(f"An unexpected error occurred during evaluation: {str(e)}")

    def predict_from_path(self, image_name, confidence_threshold=0.1):
        """
        Run inference on an image and filter results by confidence threshold.

        Args:
            image_path (str): Path to the image to be predicted.
            confidence_threshold (float): Minimum confidence score to be included in confidence_scores.

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

            confidence_scores = {}

            # ✅ กรณี Classification
            if results[0].probs is not None:
                class_names = results[0].names
                probabilities = results[0].probs.data.cpu().numpy()  # Convert to NumPy array

                # 🔹 คัดกรองค่า confidence ที่สูงกว่า `confidence_threshold`
                confidence_scores = {
                    class_names[i]: float(probabilities[i]) 
                    for i in range(len(class_names)) 
                    if probabilities[i] >= confidence_threshold
                }

                # ถ้าไม่มีค่าที่เกิน threshold ให้ return "Not found"
                if not confidence_scores:
                    return {
                        "status": "success",
                        "message": "No significant confidence scores found.",
                        "predict_result": "Not found",
                        "confidence_scores": {}
                    }

                # เลือก predict_result ที่ confidence score มากที่สุด
                predict_result = max(confidence_scores, key=confidence_scores.get)

                return {
                    "status": "success",
                    "message": "Classification completed.",
                    "predict_result": predict_result,
                    "confidence_scores": confidence_scores
                }

            # ✅ กรณี Detection / Segmentation
            filtered_predictions = []
            
            for box in results[0].boxes:
                confidence = float(box.conf.tolist()[0])  # ใช้ค่าตรงๆ
                class_id = int(box.cls.tolist()[0])  # Class ID
                class_name = results[0].names[class_id]  # Get class name

                # ✅ คัดกรองแค่ที่ confidence > confidence_threshold เท่านั้น
                if confidence >= confidence_threshold:
                    filtered_predictions.append({
                        "class": class_name,
                        "confidence": confidence
                    })

                    # ✅ บันทึกทุก instance ที่พบลงใน confidence_scores
                    if class_name in confidence_scores:
                        confidence_scores[class_name].append(confidence)
                    else:
                        confidence_scores[class_name] = [confidence]

            # ✅ หากไม่มี Object ที่ confidence score ถึง threshold
            if not confidence_scores:
                return {
                    "status": "success",
                    "message": "No significant detections found.",
                    "predict_result": "Not found",
                    "confidence_scores": {}
                }

            # ✅ ใช้ค่าเฉลี่ยของ confidence score ของ class ที่ตรวจพบมากที่สุด
            predict_result = max(confidence_scores, key=lambda k: sum(confidence_scores[k]) / len(confidence_scores[k]))
            message = "Detection completed."

            return {
                "status": "success",
                "message": message,
                "predict_result": predict_result,
                "confidence_scores": confidence_scores
            }

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return {"error": f"An error occurred during prediction: {str(e)}"}
if __name__ == "__main__":
    # 🔥 ทดสอบ Training
    pid = "project_001"  # เปลี่ยนเป็นค่า PID ที่ต้องการ
    mode = "segment"  # เปลี่ยนเป็น mode ที่ต้องการ ('detect', 'segment', 'classify')

    model_manager = MLModelManager()

    print(f"🚀 Starting training for PID: {pid}, Mode: {mode}")
    train_results = model_manager.train_model(pid, mode)

    print(train_results)
