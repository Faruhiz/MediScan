from datetime import datetime
import glob
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
import logging

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ✅ ใช้ relative path ไปยัง `MedSight_Project`
BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "MedSight_Project"))
BASE_WORKSPACE_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "MediScan", "workspace"))

class MLModelManager:
    def __init__(self):
        self.current_model = None
        self.db = DatabaseManager(BASE_PROJECT_DIR, BASE_WORKSPACE_DIR)
        self.model_name = None
        logging.info("MLModelManager initialized.")

    def prepare_workspace(self, project_id, mode):
        """🔍 เรียก `prepare_workspace.py` พร้อมส่ง `BASE_PROJECT_DIR` และ `BASE_WORKSPACE_DIR`"""
        logging.info(f"Preparing workspace for project_id: {project_id}, mode: {mode}")
        project_source_path = os.path.join(BASE_PROJECT_DIR, project_id)

        if not os.path.exists(project_source_path):
            logging.error(f"Project '{project_id}' not found at {project_source_path}")
            raise FileNotFoundError(f"❌ Project '{project_id}' not found at {project_source_path}")

        logging.info(f"Running `prepare_workspace.py` for PID: {project_id}")
        result = subprocess.run(
            ['python', 'prepare_workspace.py', project_id, mode],
            capture_output=True,
            text=True
        )

        logging.info(f"Output from `prepare_workspace.py`: {result.stdout}")
        if result.returncode != 0:
            logging.error(f"Failed to prepare workspace: {result.stderr}")
            raise RuntimeError(f"❌ Failed to prepare workspace: {result.stderr}")

        return True

    def save_model(self, project_id, model_name, mode, result_dir, model_path, validation_metrics):
        """✅ บันทึกโมเดลและอัปเดต Database"""
        logging.info(f"Saving model: {model_name} for project_id: {project_id}, mode: {mode}")
        project_path = os.path.join(BASE_PROJECT_DIR, project_id, "models", model_name)
        os.makedirs(project_path, exist_ok=True)

        if not os.path.exists(project_path):
            logging.error(f"Project '{project_id}' not found at {project_path}")
            raise FileNotFoundError(f"❌ Project '{project_id}' not found at {project_path}")

        train_output_path = os.path.join(project_path, "training_result")
        shutil.copytree(result_dir, train_output_path, dirs_exist_ok=True)
        logging.info(f"Copied training results to: {train_output_path}")

        final_model_path = os.path.join(project_path, "model.pt")
        if os.path.exists(model_path):
            shutil.copy(model_path, final_model_path)
            logging.info(f"Copied model file to: {final_model_path}")
        else:
            logging.warning(f"Model file not found: {model_path} (Skipping copy)")

        model_id = self.db.insert_model(project_id, model_name, mode, project_path, validation_metrics)
        logging.info(f"Model saved successfully with ID: {model_id}")

        return {
            "status": "success",
            "message": "Training results and model copied successfully",
            "model_id": model_id,
            "train_output_path": train_output_path,
            "model_path": final_model_path,
            "model_name": model_name,
            "validation_metrics": validation_metrics
        }

    def load_model(self, project_id, model_name):
        """Load a specific model from the ./models folder."""
        logging.info(f"Loading model: {model_name} for project_id: {project_id}")
        model_path = os.path.join(BASE_PROJECT_DIR, project_id, "models", model_name, "model.pt").replace("\\", "/")
        self.model_name = model_name

        if not os.path.exists(model_path):
            logging.error(f"Model {model_name} not found in {model_path}.")
            raise FileNotFoundError(f"Model {model_name} not found in {model_path}.")

        try:
            logging.info(f"Loading model from path: {model_path}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.current_model = YOLO(model_path).to(device)
            load_result = {
                "status": "success",
                "message": "Model loaded successfully",
                "model_path": model_path
            }
            logging.info(f"Model loaded successfully: {load_result}")
            return load_result
        except Exception as e:
            logging.error(f"Error loading model {model_path}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error loading model {model_path}: {str(e)}")

    def train_model(self, project_id, model_name, mode):
        """Train the YOLO model using yolo.py with specific mode (detect, segment, classify)."""
        logging.info(f"Training model: {model_name} for project_id: {project_id}, mode: {mode}")
        if mode not in ["detect", "segment", "classify"]:
            logging.error(f"Invalid mode '{mode}'. Use: detect, segment, classify")
            raise ValueError(f"Invalid mode '{mode}'. Use: detect, segment, classify")

        if self.db.model_exists(project_id, model_name):
            logging.error(f"Model name '{model_name}' already exists in project '{project_id}'")
            raise ValueError(f"❌ Model name '{model_name}' already exists in project '{project_id}'")

        self.prepare_workspace(project_id, mode)
        project_path = os.path.join(BASE_WORKSPACE_DIR, project_id)

        if not os.path.exists(project_path):
            logging.error(f"Project '{project_id}' not found at {project_path}")
            raise FileNotFoundError(f"❌ Project '{project_id}' not found at {project_path}")

        data_path = os.path.join(project_path, "classification") if mode == "classify" else os.path.join(project_path, "data.yaml")
        if mode != "classify" and not os.path.exists(data_path):
            logging.error(f"data.yaml not found in {data_path}")
            raise FileNotFoundError(f"❌ data.yaml not found in {data_path}")

        logging.info(f"Starting YOLO {mode} for PID: {project_id}, Data Path: {data_path}")
        try:
            result = subprocess.run(
                ['python', 'yolo.py', '--mode', mode, '--task', 'train', '--data', data_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8"
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            logging.info(f"Raw Output from yolo.py:\n{stdout}")

            if result.returncode != 0:
                logging.error(f"Training failed: {stderr}")
                raise RuntimeError(f"❌ Training failed: {stderr}")

            json_lines = [line for line in stdout.splitlines() if line.strip().startswith('{') and line.strip().endswith('}')]
            if not json_lines:
                logging.error("No valid JSON output found from yolo.py!")
                raise RuntimeError("❌ No valid JSON output found from yolo.py!")

            train_results = json.loads(json_lines[-1])
            logging.info(f"Parsed JSON from YOLO: {train_results}")

            if train_results.get("message") != "Training completed successfully":
                logging.error(f"Training failed: {train_results.get('error', 'Unknown error')}")
                raise RuntimeError(f"❌ Training failed: {train_results.get('error', 'Unknown error')}")

            result_dir = train_results.get("result_dir")
            model_path = train_results.get("model_path")
            validation_metrics = train_results.get("validation_metrics", {})

            if not result_dir or not model_path:
                logging.error("Training completed but result directory or model path not found.")
                raise RuntimeError("❌ Training completed but result directory or model path not found.")

            save_result = self.save_model(project_id, model_name, mode, result_dir, model_path, validation_metrics)
            return save_result

        except Exception as e:
            logging.error(f"An unexpected error occurred during training: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}

    # Add similar logging for other methods like `evaluate_model` and `predict_from_path`.

if __name__ == "__main__":
    model_manager = MLModelManager()