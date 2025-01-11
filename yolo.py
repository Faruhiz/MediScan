import os
import torch
from ultralytics import YOLO
import json
import argparse
from datetime import datetime

# from ultralytics import YOLO
# from IPython.display import Image, display
# from IPython import display
# display.clear_output()
# #!yolo mode=checks

# from roboflow import Roboflow
# rf = Roboflow(api_key="dawdO4POX8Am7AyPMvZr")
# project = rf.workspace("mediscan-1ggr5").project("mediscan-unpxf")
# version = project.version(1)
# dataset = version.download("yolov8")




# 1. Define Paths
train_yaml_path = "./MediScan-1/data.yaml"  # Path to data.yaml file
trained_model_path = "./runs/detect/train/weights/best.pt"  # Path to trained YOLO model weights
# test_images_path = "./MediScan-1/test/images"  # Path to test images


try:
    model = YOLO(trained_model_path)
    print(f"Model loaded from: {trained_model_path}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit(1)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="YOLO Model Operations")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate'], help="Mode: train or evaluate")
    parser.add_argument('--model', type=str, help="Path to the YOLO model file for evaluation")
    return parser.parse_args()

def train_model():
    """Train YOLO model."""
    try:
        print("Starting YOLO training...")
        model = YOLO("yolov8n.pt")  # Load the base YOLO model
        model.train(
            data=train_yaml_path,
            epochs=100,
            imgsz=640,
            batch=16,
            amp=False
        )
        print("YOLO training completed!")
        result = {
            "message": "YOLO training completed successfully",
            "model_path": trained_model_path
        }
        print(json.dumps(result))
        return result
    except Exception as e:
        error_result = {
            "error": f"Training failed: {str(e)}"
        }
        print(json.dumps(error_result))
        return error_result

# 3. Function to Validate Dataset
def validate_model(yaml_path, model):
    try:
        print("\nValidating Model on Validation Dataset...")
        model = YOLO(model)
        conf = 0.5
        results = model.val(data=yaml_path, split="val", imgsz=640, conf = conf)  # Validate on validation set

        # ดึงข้อมูลที่จำเป็น
        map50 = results.box.map50
        map = results.box.map
        precision = results.box.p.mean()
        recall = results.box.r.mean()
        f1_score = 2 * (precision * recall) / (precision + recall)

        # คำนวณจำนวน Instance ทั้งหมดและ True Positives โดยอิงจาก Precision และ Recall
        total_instances = results.box.nc  # จำนวนคลาสทั้งหมด
        correct_predictions = precision * total_instances * recall  # ประมาณ True Positives

        print("Evaluation completed!")

        # Prepare JSON result
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
        


# 4. Function to Test Dataset
def test_model(yaml_path, model, conf):
    print("\nEvaluating Model on Test Dataset...")
    results = model.val(data=yaml_path, split="test", imgsz=640, conf = conf)  # Use `val` with test split for evaluation

    # Extract relevant metrics
    map50 = results.box.map50
    map = results.box.map
    precision = results.box.p.mean()  # Use `.mean()` to get the average precision
    recall = results.box.r.mean()  # Use `.mean()` to get the average recall
    f1_score = 2 * (precision * recall) / (precision + recall)  # Calculate F1 Score

    total_instances = results.box.nc  # จำนวนคลาสทั้งหมด
    correct_predictions = precision * total_instances * recall  # ประมาณ True Positives
    
    print("\nTest Results:")
    print(f"  Total Instances: {total_instances}")
    print(f"  Correct Predictions (approx.): {correct_predictions:.2f}")
    print(f"  Accuracy: {correct_predictions / total_instances:.2%}")
    print(f"  mAP50: {map50:.4f}")
    print(f"  mAP50-95: {map:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")
    return results

# 5. Evaluate Validation and Test Datasets
if __name__ == "__main__":
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Using CPU.")

    args = parse_arguments()

    if args.mode == 'train':
        # Train the model
        train_model()
    elif args.mode == 'evaluate':
        # Validate the model
        if not args.model:
            error_message = {"error": "Model path is required for evaluation"}
            print(json.dumps(error_message))
        else:
            validate_model(train_yaml_path, args.model)
    # 2. Function to Load Model
    # Define training command
    # train_command = "yolo detect train data='./MediScan-1/data.yaml' epochs=100 imgsz=640 batch=16 amp=False"
    # train_results = train_model()
    # print(json.dumps(train_results, indent=4))
    # Execute training
    # print("Starting YOLO Training...")

    # os.system(train_command)  # รันคำสั่ง YOLO
    # print("Training completed!")
    

    # print(json.dumps(eval_results, indent=4))

    # print("\nValidating the trained model...")
    # validation_results = validate_model(train_yaml_path, model)
    # print("\nTesting the trained model...")
    # test_results = test_model(train_yaml_path, model)
    # print("\nEvaluation completed.")
