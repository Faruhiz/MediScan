
# from roboflow import Roboflow
# rf = Roboflow(api_key="dawdO4POX8Am7AyPMvZr")
# project = rf.workspace("mediscan-1ggr5").project("test_segmentation-wzr7l")
# version = project.version(1)
# dataset = version.download("yolov8-obb")
                

# seg_yolo.py
import torch
from ultralytics import YOLO


def train():
    model = YOLO("yolo8n-seg.pt")  # Load the YOLO segmentation model
    model.train(
        data="C:/MediScan/Test_segmentation-1/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        amp=False
    )

def evaluate_val():
    # Load a pretrained YOLO model
    model_path = "C:/MediScan/runs/segment/train2/weights/best.pt"  # Adjust the path to your model weights
    model = YOLO(model_path)

    # Evaluate the model using the data in the YAML file
    results = model.val(data="C:/MediScan/Test_segmentation-1/data.yaml", split="val", imgsz=640, conf = 0.5)  # This line is updated

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

def evaluate_test():
    # Load a pretrained YOLO model
    model_path = "C:/MediScan/runs/segment/train2/weights/best.pt"  # Adjust the path to your model weights
    model = YOLO(model_path)

    # Evaluate the model using the data in the YAML file
    results = model.val(data="C:/MediScan/Test_segmentation-1/data.yaml", split="test", imgsz=640, conf = 0.5)  # This line is updated

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

if __name__ == '__main__':
    # train()
    evaluate_val()
    # evaluate_test()