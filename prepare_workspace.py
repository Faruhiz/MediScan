import os
import shutil
import random
import sys


def load_class_names_from_yaml(filepath):
    """อ่าน class names จาก data.yaml โดยไม่ใช้ PyYAML"""
    class_names = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    reading_names = False
    for line in lines:
        line = line.strip()

        if line.startswith("names:"):
            if "[" in line:
                # แบบ inline: names: ['cancer', 'non-cancer']
                raw = line.split(":", 1)[1].strip().strip("[]")
                for idx, name in enumerate(raw.split(",")):
                    class_names[idx] = name.strip().strip("'\"")
                break
            else:
                reading_names = True
                continue

        if reading_names:
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                if k.isdigit():  # ข้ามบรรทัดที่ไม่ใช่เลข เช่น 'nc'
                    class_id = int(k)
                    class_name = v.strip().strip("'\"")
                    class_names[class_id] = class_name
            elif not line:
                break  # stop on empty line or end of block

    return class_names



def prepare_workspace(pid, mode):
    BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "MedSight_Project"))
    BASE_WORKSPACE_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "MediScan", "workspace"))

    source_folder = os.path.join(BASE_PROJECT_DIR, pid, "annotated_images")
    data_yaml_source = os.path.join(BASE_PROJECT_DIR, pid, "data.yaml")

    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Project '{pid}' not found at {source_folder}")

    if mode == "classify":
        # === Classification Mode ===
        classification_folder = os.path.join(BASE_WORKSPACE_DIR, pid, "classification")

        # ลบเฉพาะ folder classification
        if os.path.exists(classification_folder):
            shutil.rmtree(classification_folder)

        for split in ["train", "valid", "test"]:
            os.makedirs(os.path.join(classification_folder, split), exist_ok=True)

        # โหลดชื่อคลาสจาก data.yaml
        class_names = load_class_names_from_yaml(data_yaml_source)

        image_label_pairs = []
        for folder_name in os.listdir(source_folder):
            folder_path = os.path.join(source_folder, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(folder_path, file_name)
                        label_path = os.path.join(folder_path, os.path.splitext(file_name)[0] + ".txt")
                        if os.path.exists(label_path):
                            image_label_pairs.append((image_path, label_path))

        if not image_label_pairs:
            raise RuntimeError("No labeled images found for classification.")

        # Shuffle and split
        random.shuffle(image_label_pairs)
        total = len(image_label_pairs)
        val_size = int(total * 0.15)
        test_size = int(total * 0.15)
        train_size = total - val_size - test_size

        splits = {
            "train": image_label_pairs[:train_size],
            "valid": image_label_pairs[train_size:train_size + val_size],
            "test": image_label_pairs[train_size + val_size:]
        }

        for split, items in splits.items():
            for image_path, label_path in items:
                with open(label_path, 'r') as f:
                    first_line = f.readline().strip()
                    if not first_line:
                        continue
                    class_id = int(first_line.split()[0])
                    class_name = class_names.get(class_id, f"class_{class_id}")
                    class_folder = os.path.join(classification_folder, split, class_name)
                    os.makedirs(class_folder, exist_ok=True)
                    shutil.copy(image_path, os.path.join(class_folder, os.path.basename(image_path)))

        print(f"Classification dataset prepared at: {classification_folder}")

    else:
        # === Segmentation or Detection Mode ===
        project_path = os.path.join(BASE_WORKSPACE_DIR, pid)
        train_images = os.path.join(project_path, "train/images")
        train_labels = os.path.join(project_path, "train/labels")
        val_images = os.path.join(project_path, "valid/images")
        val_labels = os.path.join(project_path, "valid/labels")
        test_images = os.path.join(project_path, "test/images")
        test_labels = os.path.join(project_path, "test/labels")

        # ลบเฉพาะ segmentation/detection part
        for path in [train_images, train_labels, val_images, val_labels, test_images, test_labels]:
            if os.path.exists(os.path.dirname(path)):
                shutil.rmtree(os.path.dirname(path))

        for folder in [train_images, train_labels, val_images, val_labels, test_images, test_labels]:
            os.makedirs(folder, exist_ok=True)

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
            raise RuntimeError("No labeled images found for segmentation/detection.")

        total_files = len(image_files)
        random.shuffle(image_files)
        val_test_files = random.sample(image_files, int(total_files * 0.3))
        val_size = int(len(val_test_files) * 0.5)
        val_files = val_test_files[:val_size]
        test_files = val_test_files[val_size:]
        train_files = [f for f in image_files if f not in val_test_files]

        def move_files(file_list, dest_img, dest_lbl):
            for img, lbl in file_list:
                shutil.copy(img, dest_img)
                shutil.copy(lbl, dest_lbl)

        move_files(train_files, train_images, train_labels)
        move_files(val_files, val_images, val_labels)
        move_files(test_files, test_images, test_labels)

        # Copy data.yaml
        if os.path.exists(data_yaml_source):
            shutil.copy(data_yaml_source, os.path.join(project_path, "data.yaml"))
            print(f"Copied data.yaml to {project_path}")
        else:
            print("data.yaml not found in project.")

        print(f"Segmentation/Detection dataset prepared at: {project_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python prepare_workspace.py <project_id> <mode>")
        sys.exit(1)

    project_id = sys.argv[1].strip()
    mode = sys.argv[2].strip().lower()
    print(f"Prepare workspace for project: {project_id} in mode: {mode}")
    prepare_workspace(project_id, mode)
