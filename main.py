import os
import socket
import threading
import redis
import pickle
import signal
import sys
import base64
import cv2
import numpy as np
import torch
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import subprocess
import json
from datetime import datetime
from ultralytics import YOLO

# Directory for storing models
MODEL_FOLDER = './models'
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# Connect to Redis
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# Global variable to control the server's running state
server_running = True


### --- MLModelManager Class ---

class MLModelManager:
    def __init__(self, model_folder=MODEL_FOLDER):
        self.model_folder = model_folder
        self.current_model = None

    def save_model(self, model_path):
        """Save the trained model to the ./models folder with a dynamic name."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dynamic_model_name = f"model_{timestamp}.pt"
        saved_path = os.path.join(self.model_folder, dynamic_model_name)

        os.makedirs(self.model_folder, exist_ok=True)
        torch.save(torch.load(model_path), saved_path)
        print(f"Model saved to {saved_path}")
        return saved_path
    
    def load_model(self, model_name):
        """
        Load a specific model from the ./models folder.

        Args:
            model_name (str): The name of the model file to load.

        Returns:
            torch.nn.Module: The loaded PyTorch model.
        """
        model_path = os.path.join(self.model_folder, model_name)
        print(f"Loading model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} not found in {self.model_folder}.")

        try:
            print(f"Loading model: {model_path}")
            self.current_model = YOLO(model_path)  # Use Ultralytics' YOLO to load the model
            print(f"Model loaded successfully from {model_path}.")
            return self.current_model
        except Exception as e:
            raise RuntimeError(f"Error loading model {model_name}: {str(e)}")

    def train_model(self):
        """Train the YOLO model using yolo.py."""
        try:
            print("Starting YOLO training...")
            result = subprocess.run(
                ['python', 'yolo.py', '--mode', 'train'],  # Pass mode as 'train' to yolo.py
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout = result.stdout.decode("utf-8").strip()
            stderr = result.stderr.decode("utf-8")

            print(f"Raw Output from yolo.py:\n{stdout}")

            if result.returncode != 0:
                return {"status": "error", "message": "YOLO training failed", "details": stderr}

            # Parse JSON output from yolo.py
            try:
                json_lines = [line for line in stdout.splitlines() if line.startswith('{') and line.endswith('}')]
                if not json_lines:
                    raise json.JSONDecodeError("No valid JSON output found", stdout, 0)

                train_results = json.loads(json_lines[-1])  # Parse the last valid JSON line

                # Check if the training results contain an error
                if "error" in train_results:
                    return {"status": "error", "message": train_results["error"], "details": train_results}

                trained_model_path = train_results.get("model_path")

                # Save the model dynamically
                if trained_model_path and os.path.exists(trained_model_path):
                    saved_path = self.save_model(trained_model_path)
                    train_results["saved_path"] = saved_path
                else:
                    train_results["saved_path"] = None

                print(f"Training Results: {json.dumps(train_results, indent=4)}")
                return {"status": "success", "data": train_results}
            except json.JSONDecodeError as e:
                return {"status": "error", "message": "Invalid JSON output from yolo.py", "raw_output": stdout}

        except Exception as e:
            return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}
        

    def evaluate_model(self, model_path="./runs/detect/train/weights/best.pt", conf=0.5):
        """Evaluate a specific YOLO model."""
        try:
            print(f"Evaluating model from: {model_path}")
            
            # Ensure the model path exists
            if not os.path.exists(model_path):
                return {"error": f"Model not found at {model_path}"}

            # Call yolo.py with the selected model path
            result = subprocess.run(
                ['python', 'yolo.py', '--mode', 'evaluate', '--model', model_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout = result.stdout.decode("utf-8").strip()
            stderr = result.stderr.decode("utf-8")

            # Debug: Print raw output
            print(f"Raw Output from yolo.py:\n{stdout}")

            # Check for errors
            if result.returncode != 0:
                return {"error": "YOLO evaluation failed", "details": stderr}

            # Parse the last valid JSON output
            try:
                json_lines = [line for line in stdout.splitlines() if line.startswith('{') and line.endswith('}')]
                if not json_lines:
                    raise json.JSONDecodeError("No valid JSON output found", stdout, 0)

                eval_results = json.loads(json_lines[-1])
                print(f"Evaluation Results: {json.dumps(eval_results, indent=4)}")
                return eval_results
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON output: {e}")
                return {"error": "Invalid JSON output from yolo.py", "raw_output": stdout}

        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}


    # def predict(self, input_data):
    #     """Make predictions using the deployed model."""
    #     if not self.current_model:
    #         return {"error": "No model deployed"}
        
    #     predictions = self.current_model.predict(input_data)
    #     return {"predictions": predictions.tolist()}
    
    def predict_image(self, image_path, confidence_threshold=0.3):
        """
        Run inference on an image and filter results by confidence threshold.

        Args:
            image_path (str): Path to the image to be predicted.
            confidence_threshold (float): Minimum confidence score to consider a detection.

        Returns:
            dict: Predictions or an error message.
        """
        if not self.current_model:
            return {"error": "No model loaded. Please load a model first."}
        
        if not os.path.exists(image_path):
            return {"error": f"Image not found at {image_path}"}

        try:
            print(f"Running inference on: {image_path}")
            results = self.current_model.predict(source=image_path, save=False)

            # Filter predictions by confidence threshold
            filtered_predictions = []
            for box in results[0].boxes:
                bbox = box.xyxy.tolist()[0]  # Bounding box [x1, y1, x2, y2]
                confidence = box.conf.tolist()[0]  # Confidence score
                class_id = int(box.cls.tolist()[0])  # Class ID

                if confidence >= confidence_threshold:
                    filtered_predictions.append({
                        "bbox": bbox,
                        "confidence": confidence,
                        "class_id": class_id,
                    })

            return {"status": "success", "predictions": filtered_predictions}

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return {"error": f"An error occurred during prediction: {str(e)}"}
    

    def predict_from_path(self, image_path):
        """Make predictions from an image path using YOLOv5."""
        if not self.current_model:
            return {"error": "Model not loaded"}

        # Load and preprocess the image
        try:
            print(f"Loading image from {image_path}")
            image = Image.open(image_path).convert('RGB')
            image = image.resize((640, 640))  # Resize to 640x640 for YOLOv5
            image = np.array(image) / 255.0  # Normalize to [0, 1]
            image_tensor = torch.tensor(image).float().unsqueeze(0).permute(0, 3, 1, 2)  # Add batch dimension and correct shape

            # If using GPU, move the tensor to CUDA
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()

            print(f"Image shape before prediction: {image_tensor.shape}")

            # Check the model type
            print(f"Model type: {type(self.current_model)}")

            # Make prediction using YOLOv5 model
            with torch.no_grad():
                print("Making prediction...")
                output = self.current_model(image_tensor)

            # Ensure the output is a dictionary (as expected in YOLOv5)
            if isinstance(output, dict):
                # Extract predictions from the 'pred' key
                predictions = output['pred'][0]  # YOLOv5 typically stores predictions in this format
            else:
                return {"error": "Model output is not in the expected format"}

            # Filter predictions (e.g., confidence threshold of 0.5)
            threshold = 0.5
            detections = predictions[predictions[:, 4] > threshold]  # Filter by confidence score

            # Process predictions (bounding boxes, class ids, and scores)
            result = []
            for det in detections:
                bbox = det[:4]  # Bounding box [x1, y1, x2, y2]
                confidence = det[4]  # Confidence score
                class_id = int(det[5])  # Class ID
                result.append({
                    "bbox": bbox.tolist(),
                    "confidence": confidence,
                    "class_id": class_id
                })

            return {"predictions": result}

        except Exception as e:
            print(f"Error in predict_from_path: {str(e)}")
            return {"error": str(e)}
        

    def draw_bounding_boxes(self, image_path, predictions, output_path="./output.jpg"):
        """
        Draw bounding boxes on the image and save it.

        Args:
            image_path (str): Path to the input image.
            predictions (list): List of predictions with bounding boxes.
            output_path (str): Path to save the output image.

        Returns:
            str: Path to the saved output image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Load the image
        image = cv2.imread(image_path)

        # Draw each bounding box
        for pred in predictions:
            bbox = pred["bbox"]
            confidence = pred["confidence"]
            class_id = pred["class_id"]

            # Convert to integers for drawing
            x_min, y_min, x_max, y_max = map(int, bbox)
            
            # Draw rectangle and label
            color = (0, 255, 0)  # Green for bounding box
            label = f"Class {class_id}: {confidence:.2f}"
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save the output image
        cv2.imwrite(output_path, image)
        print(f"Output image saved to {output_path}")
        return output_path




### --- TCPServer Class ---

class TCPServer:
    def __init__(self, host='0.0.0.0', port=5001, model_manager=None):
        self.host = host
        self.port = port
        self.model_manager = model_manager
        self.server_socket = None

    def handle_command(self, command, conn):
        """Process a single command."""
        response = {"error": "Unknown command"}
        
        try:
            if command == 'train':
                result = self.model_manager.train_model()
                response = result
            elif command.startswith('evaluate'):
                # Parse the incoming data for evaluate
                try:
                    parts = command.split('|')  # Split the command string using '|'
                    if len(parts) < 2:
                        raise ValueError("Model path is required for evaluation.")

                    model_path = parts[1].strip()  # First part is the model path
                    # Check if the model path exists
                    if not os.path.exists(model_path):
                        print(f"Model not found at {model_path}")
                        response = {"status": "error", "message": f"Model not found at {model_path}"}
                    else: 
                        print(f"Evaluating model: {model_path}")
                        result = self.model_manager.evaluate_model(model_path=model_path)
                        response = {"status": "success", "data": result}
                except Exception as e:
                    response = {"status": "error", "message": f"Error during evaluation: {str(e)}"}
            elif command.startswith('predict'):
                 # Extract image path
                parts = command.split('|')
                
                # Check if the path is provided
                if len(parts) < 2:
                    response = {"error": "Missing image path"}
                else:
                    image_path = parts[1].strip()  # Strip any extra whitespace
                    print(f"Received image path: {image_path}")
                    
                    # Check if the file exists
                    if not os.path.exists(image_path):
                        response = {"error": f"File not found: {image_path}"}
                    else:
                        # Load the image, process it, and make predictions
                        response = self.model_manager.predict_from_path(image_path)
            elif command.startswith('deploy'):
                model_name = command.split('|')[1] if '|' in command else "ml_model.pkl"
                try:
                    self.model_manager.load_model(model_name)
                    response = {"message": f"Model '{model_name}' deployed successfully"}
                except Exception as e:
                    response = {"error": f"Failed to deploy model: {str(e)}"}
            else:
                response = {"error": "Invalid command"}
        except Exception as e:
            response = {"error": f"Error processing command: {str(e)}"}

        conn.sendall((str(response) + "\n").encode('utf-8'))

    def handle_client(self, conn):
        """Handle TCP client requests."""
        try:
            with conn:
                while server_running:  # Use the global `server_running` to control the loop
                    data = conn.recv(4096)
                    if not data:
                        break

                    # Parse client command
                    command = data.decode('utf-8').strip()

                    # Process the command in a separate daemon thread
                    threading.Thread(target=self.handle_command, args=(command, conn), daemon=True).start()
        except Exception as e:
            print(f"Error handling client: {e}")

    def start_server(self):
        """Start the TCP server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.settimeout(1.0)  # Set a timeout to allow graceful shutdown checks
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"TCP server running on {self.host}:{self.port}")

        while server_running:
            try:
                conn, addr = self.server_socket.accept()
                print(f"Connected by {addr}")
                # Handle each client in a daemon thread
                threading.Thread(target=self.handle_client, args=(conn,), daemon=True).start()
            except socket.timeout:
                # This allows the loop to periodically check the `server_running` flag
                continue
            except Exception as e:
                print(f"Server error: {e}")
                break

    def stop_server(self):
        """Stop the TCP server."""
        global server_running
        server_running = False
        if self.server_socket:
            self.server_socket.close()


### --- RedisHandler Class ---

class RedisHandler:
    def __init__(self, redis_client, model_manager):
        self.redis_client = redis_client
        self.model_manager = model_manager

    def listen(self):
        """Listen for Redis messages and handle them."""
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(['train', 'evaluate', 'predict', 'deploy'])

        print("Listening for Redis messages...")
        for message in pubsub.listen():
            if not server_running:
                break
            try:
                channel = message['channel'].decode('utf-8')
                data = message['data']

                # Check if data is an integer (e.g., Redis subscription confirmation)
                if isinstance(data, int):
                    print(f"Received non-command message: {data}")
                    continue

                # Decode text data if it is in bytes
                if isinstance(data, bytes):
                    data = data.decode('utf-8')

                print(f"Redis message on {channel}: {data}")

                response = {"status": "error", "message": "Unknown command"}

                result = None

                try:
                    if channel == 'train':
                        result = self.model_manager.train_model()
                        response = {"status": "success", "data": result}
                    elif channel == 'evaluate':
                        # Parse the incoming data for evaluate
                        try:
                            parts = data.split('|')  # Split the data string using '|'
                            if len(parts) < 1:
                                raise ValueError("Model path is required for evaluation.")

                            model_path = parts[0].strip()  # First part is the model path

                            # Check if the model path exists
                            if not os.path.exists(model_path):
                                print(f"Model not found at {model_path}")
                                response = {"status": "error", "message": f"Model not found at {model_path}"}
                    
                            else:
                                print(f"Evaluating model: {model_path} ")
                                result = self.model_manager.evaluate_model(model_path=model_path)
                                response = {"status": "success", "data": result}
                        except Exception as e:
                            response = {"status": "error", "message": f"Error during evaluation: {str(e)}"}
                    elif channel == 'predict':
                        # Expect the `data` to be the path to an image file or base64 image
                        
                        image_path = 'D:/Code/Project/MediScan/test-image/1.png'
                        result = model_manager.predict_from_path(image_path)
                        print(result)  # This should give you either predictions or an error message
                        response = {"status": "success", "data": result}
                    elif channel == 'deploy':
                        model_name = data if data else "ml_model.pkl"
                        try:
                            self.model_manager.load_model(model_name)
                            response = {"status": "success", "message": f"Model '{model_name}' deployed successfully"}
                        except Exception as e:
                            response = {"status": "error", "message": f"Failed to deploy model: {str(e)}"}
                    else:
                        response = {"status": "error", "message": "Invalid command"}
                except Exception as e:
                    response = {"status": "error", "message": f"Error processing command: {str(e)}"}

                # Publish response to a dedicated response channel
                self.redis_client.publish('response', str(response))

            except UnicodeDecodeError as e:
                print(f"Error decoding message: {e}")
                continue


### --- Signal Handler for Graceful Shutdown ---

def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) for graceful shutdown."""
    print("\nTerminated servers process...")
    global server_running
    server_running = False
    tcp_server.stop_server()
    sys.exit(0)


### --- Main Execution ---

if __name__ == '__main__':
    # Set up the signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize the MLModelManager
    model_manager = MLModelManager()

    # Start the RedisHandler in a separate daemon thread

    redis_handler = RedisHandler(redis_client, model_manager)
    threading.Thread(target=redis_handler.listen, daemon=True).start()

    # Start the TCP server

    tcp_server = TCPServer(model_manager=model_manager)
    tcp_server.start_server()

    print("Loading model...")
    # Load the model
    model_name = "yolo_best.pt"  # Replace with your saved model name
    try:
        model_manager.load_model(model_name)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
    # TEST predictions
    image_path = "./3.jpg"  # Replace with your test image path
    predictions = model_manager.predict_image("./3.jpg", confidence_threshold=0.4)
    if "predictions" in predictions:
        print(json.dumps(predictions, indent=4))

        # Draw and save bounding boxes
        output_path = model_manager.draw_bounding_boxes(image_path, predictions["predictions"])
        print(f"Image with bounding boxes saved to: {output_path}")

    
