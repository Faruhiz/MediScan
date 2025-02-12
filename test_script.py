import subprocess
import argparse

# Define the script name
script_name = "yolo_2.py"

# Define test cases
test_cases = {
    "segment_train": {"mode": "segment", "task": "train"},
    "detect_train": {"mode": "detect", "task": "train"},
    "classify_train": {"mode": "classify", "task": "train"},
    
    "segment_evaluate": {"mode": "segment", "task": "evaluate", "trained_model_path": "C:/MediScan/runs/segment/train/weights/best.pt"},
    "detect_evaluate": {"mode": "detect", "task": "evaluate", "trained_model_path": "C:/MediScan/runs/detect/train/weights/best.pt"},
    "classify_evaluate": {"mode": "classify", "task": "evaluate", "trained_model_path": "C:/MediScan/runs/classify/train/weights/best.pt"},

    "segment_test": {"mode": "segment", "task": "test", "trained_model_path": "C:/MediScan/runs/segment/train/weights/best.pt"},
    "detect_test": {"mode": "detect", "task": "test", "trained_model_path": "C:/MediScan/runs/detect/train/weights/best.pt"},
    "classify_test": {"mode": "classify", "task": "test", "trained_model_path": "C:/MediScan/runs/classify/train/weights/best.pt"},
}

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run a specific YOLO test case")
parser.add_argument('--case', type=str, required=True, choices=test_cases.keys(), help="Test case to run")
args = parser.parse_args()

# Get the selected test case
case = test_cases[args.case]

print(f"\n🔹 Running test: Mode={case['mode']}, Task={case['task']}")

# Build command
command = ["python", script_name, "--mode", case["mode"], "--task", case["task"]]

if "trained_model_path" in case:
    command.extend(["--trained_model_path", case["trained_model_path"]])

# Execute command with UTF-8 encoding
try:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=True)
    print(f"✅ Test passed: {case['mode']} {case['task']}")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"❌ Test failed: {case['mode']} {case['task']}")
    print(e.stderr)