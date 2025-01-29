if __name__ == "__main__":
    from ultralytics import YOLO
    model = YOLO("yolov8n-seg.pt")
    model.train(data="C:/MediScan/seg_dataset/data.yaml", epochs=50, batch=16, imgsz=640)
