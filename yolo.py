import os
import re
import torch
import json
import argparse
from ultralytics import YOLO

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="YOLO Model Operations")
    parser.add_argument('--mode', type=str, required=True, choices=['segment', 'detect', 'classify'], help="Model Type: segmentation, detection, or classification")
    parser.add_argument('--task', type=str, required=True, choices=['train', 'evaluate', 'test'], help="Task Type: train, evaluate, or test")
    parser.add_argument('--trained_model_path', type=str, required=False, help="Path to trained model (Only for evaluate/test)")
    return parser.parse_args()

# Define Paths
BASE_PATH = "C:/MediScan"
DATA_PATH_MAP = {
    "segment": os.path.join(BASE_PATH, "Test_segmentation-1/data.yaml"),
    "detect": os.path.join(BASE_PATH, "Test_segmentation-1/data.yaml"),
    "classify": os.path.join(BASE_PATH, "classification")  # Classification uses a directory, not a .yaml
}

MODEL_MAP = {
    "segment": "yolov8n-seg.pt",
    "detect": "yolov8n.pt",
    "classify": "yolov8n-cls.pt"
}

def train_model(mode):
    """Train YOLO model based on mode."""
    try:
        print(f"🚀 Starting YOLO {mode} training...")
        model = YOLO(MODEL_MAP[mode])

        if mode == "classify":
            results = model.train(
                data=DATA_PATH_MAP[mode],  
                epochs=50,
                imgsz=224,
                batch=16,
                amp=False
            )
        else:
            results = model.train(
                data=DATA_PATH_MAP[mode],
                epochs=10,
                imgsz=640,
                batch=8,
                amp=False
            )

        print("✅ YOLO training completed!")

        # ✅ ใช้ results.save_dir เพื่อหา path ที่ถูกต้อง
        save_dir = results.save_dir if hasattr(results, "save_dir") else None
        if save_dir:
            trained_model_path = os.path.join(str(save_dir), "weights", "best.pt")
            result_dir = str(save_dir)
        else:
            trained_model_path = None
            result_dir = None

        # ✅ ตรวจสอบว่ามีโมเดลหรือไม่
        if not trained_model_path or not os.path.exists(trained_model_path):
            print("Training completed, but no model was saved!")
            result_json = {"error": "Model training completed, but model file not found."}
        else:
            result_json = {
                "message": "Training completed successfully",
                "mode": mode,
                "model_path": trained_model_path,
                "result_dir": result_dir
            }

        # ✅ Print JSON เป็นบรรทัดสุดท้าย
        print(json.dumps(result_json))
        return result_json

    except Exception as e:
        error_json = {"error": f"Training failed: {str(e)}"}
        print(json.dumps(error_json))
        return error_json


def evaluate_or_test_model(mode, trained_model_path, task):
    """Evaluate or test YOLO model."""
    try:
        print(f"\n🔹 {task.capitalize()} {mode} Model...")
        model = YOLO(trained_model_path)

        if mode == "classify":
            metrics = model.val(data=DATA_PATH_MAP[mode])
            top1_accuracy = float(metrics.top1) if metrics.top1 is not None else 0.0
            top5_accuracy = float(metrics.top5) if metrics.top5 is not None else 0.0
            
            # ✅ ใช้ JSON format แบบเดียวกับ detect/segment
            result_json = {
                "task": task,
                "mode": mode,
                "metrics": {
                    "top1_accuracy": top1_accuracy,
                    "top5_accuracy": top5_accuracy
                }
            }
            print(json.dumps(result_json))  # Ensure output is JSON
            return result_json  # Return result to the caller
        else:
            split_type = "val" if task == "evaluate" else "test"
            results = model.val(data=DATA_PATH_MAP[mode], split=split_type, imgsz=640, conf=0.5)

            # Extract relevant metrics
            map50 = results.box.map50
            map95 = results.box.map
            precision = results.box.p.mean()
            recall = results.box.r.mean()
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            result_json = {
            "message": "YOLO evaluation completed successfully",
            "metrics": {
                "mAP50": map50,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
                }
            }
            # Print result as JSON
            print(json.dumps(result_json))  # Ensure output is JSON
            return result_json  # Return result to the caller
    except Exception as e:
        error_json = {"error": f"Evaluation failed: {str(e)}"}
        print(json.dumps(error_json))
        return error_json  # Return error details to the caller

if __name__ == "__main__":
    args = parse_arguments()

    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Run based on the chosen task
    if args.task == "train":
        train_model(args.mode)
    elif args.task in ["evaluate", "test"]:
        if not args.trained_model_path:
            print(json.dumps({"error": "trained_model_path is required for evaluate/test"}))
        else:
            evaluate_or_test_model(args.mode, args.trained_model_path, args.task)
