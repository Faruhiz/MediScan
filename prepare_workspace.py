import os
import shutil
import random
import sys

def prepare_workspace(pid):
    """🔍 เตรียมโฟลเดอร์ workspace และแบ่ง dataset"""
    
    BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "MedSight_Project"))
    BASE_WORKSPACE_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "MediScan", "workspace"))
    
    project_path = os.path.join(BASE_WORKSPACE_DIR, pid)
    source_folder = os.path.join(BASE_PROJECT_DIR, pid, "annotated_images")  
    data_yaml_source = os.path.join(BASE_PROJECT_DIR, pid, "data.yaml")  
    data_yaml_dest = os.path.join(project_path, "data.yaml")

    train_images = os.path.join(project_path, "train/images")
    train_labels = os.path.join(project_path, "train/labels")
    val_images = os.path.join(project_path, "valid/images")
    val_labels = os.path.join(project_path, "valid/labels")
    test_images = os.path.join(project_path, "test/images")
    test_labels = os.path.join(project_path, "test/labels")
    
    # ✅ ลบ workspace/{pid} ถ้ามีอยู่ก่อนแล้ว
    if os.path.exists(project_path):
        shutil.rmtree(project_path)

    # ✅ สร้างโฟลเดอร์ใหม่
    for folder in [train_images, train_labels, val_images, val_labels, test_images, test_labels]:
        os.makedirs(folder, exist_ok=True)

    # ✅ ตรวจสอบโฟลเดอร์โปรเจคที่มี annotated images
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Project '{pid}' not found at {source_folder}")

    # ✅ เก็บไฟล์ภาพที่มี label เท่านั้น
    image_files = []
    valid_image_files = set()

    for folder_name in os.listdir(source_folder):
        folder_path = os.path.join(source_folder, folder_name)

        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith((".jpg", ".png", ".jpeg")):
                    image_path = os.path.join(folder_path, file_name)
                    label_name = os.path.splitext(file_name)[0] + ".txt"
                    label_path = os.path.join(folder_path, label_name)

                    if os.path.basename(image_path) in valid_image_files:
                        continue  

                    if os.path.exists(label_path):
                        image_files.append((image_path, label_path))
                        valid_image_files.add(os.path.basename(image_path))

    if not image_files:
        raise RuntimeError(f"Warning: No labeled images found in {source_folder}")

    # ✅ สุ่มข้อมูลและแบ่ง train, val, test
    total_files = len(image_files)
    random.shuffle(image_files)

    train_files = image_files.copy()
    val_test_files = random.sample(train_files, int(len(train_files) * 0.3))
    val_size = int(len(val_test_files) * 0.5)
    val_files = val_test_files[:val_size]
    test_files = val_test_files[val_size:]

    def move_files(file_list, dest_images_folder, dest_labels_folder):
        for image_path, label_path in file_list:
            shutil.copy(image_path, dest_images_folder)
            shutil.copy(label_path, dest_labels_folder)

    move_files(train_files, train_images, train_labels)
    move_files(val_files, val_images, val_labels)
    move_files(test_files, test_images, test_labels)

    # ✅ คัดลอก data.yaml
    if os.path.exists(data_yaml_source):
        shutil.copy(data_yaml_source, data_yaml_dest)
        print(f"Copied data.yaml to {project_path}")
    else:
        print("Warning: data.yaml not found in the source folder!")

    # ✅ ตรวจสอบจำนวนไฟล์จริงที่ถูกคัดลอก
    real_train_count = len(os.listdir(train_images))
    real_val_count = len(os.listdir(val_images))
    real_test_count = len(os.listdir(test_images))

    # ✅ คำนวณเปอร์เซ็นต์
    train_percent = (real_train_count / total_files) * 100 if total_files > 0 else 0
    val_percent = (real_val_count / total_files) * 100 if total_files > 0 else 0
    test_percent = (real_test_count / total_files) * 100 if total_files > 0 else 0

    print(f"Data split completed for {pid}")
    print(f"Data split summary:")
    print(f"Train: {real_train_count} images ({train_percent:.2f}%)")
    print(f"Valid: {real_val_count} images ({val_percent:.2f}%)")
    print(f"Test : {real_test_count} images ({test_percent:.2f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python prepare_workspace.py <project_id>")
        sys.exit(1)
    print("Prepare workspace for project: ", sys.argv[1])
    project_id = sys.argv[1].strip()
    prepare_workspace(project_id)
