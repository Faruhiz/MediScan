import os
import shutil

# Paths
BASE_PATH = "C:/MediScan/Test_segmentation-1"
CLASSIFICATION_PATH = "C:/MediScan/classification"

# Class mapping (from data.yaml)
CLASSES = {0: "cancer", 1: "non-cancer"}

# Create classification directories
for split in ["train", "valid", "test"]:
    for class_name in CLASSES.values():
        os.makedirs(os.path.join(CLASSIFICATION_PATH, split, class_name), exist_ok=True)

# Function to move images based on labels
def process_split(split):
    labels_path = os.path.join(BASE_PATH, split, "labels")
    images_path = os.path.join(BASE_PATH, split, "images")
    
    for label_file in os.listdir(labels_path):
        image_file = label_file.replace(".txt", ".jpg")  # Change based on image format
        with open(os.path.join(labels_path, label_file), "r") as f:
            lines = f.readlines()
            if lines:
                class_index = int(lines[0].split()[0])  # Read first class from YOLO label
                class_name = CLASSES[class_index]
                src = os.path.join(images_path, image_file)
                dest = os.path.join(CLASSIFICATION_PATH, split, class_name, image_file)
                if os.path.exists(src):  # Ensure image exists before moving
                    shutil.copy(src, dest)

# Process all splits
for split in ["train", "valid", "test"]:
    process_split(split)

print("✅ Classification dataset created successfully!")
