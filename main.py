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

    def save_model(self, model, model_name="ml_model.pkl"):
        """Save the trained model to a file."""
        model_path = os.path.join(self.model_folder, model_name)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return model_path

    def load_model(self, model_name="yolo11n.pt"):
        """Load a saved model."""
        model_path = os.path.join(self.model_folder, model_name)
        try:
            print(f"Loading model from {model_path}")
            self.current_model = torch.load(model_path)
            self.current_model.eval()
            print(f"Model output type: {type(self.current_model)}")

            print(f"Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        return self.current_model


    def train_model(self):
        """Train a new ML model."""
        # Example: Use Iris dataset
        data = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            data['data'], data['target'], test_size=0.2, random_state=42
        )
        
        # Train the model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Evaluate the model on the test set
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Save the trained model
        model_path = self.save_model(model)
        self.current_model = model
        
        return {
            "message": "Model trained successfully",
            "accuracy": accuracy,
            "model_path": model_path
        }

    def evaluate_model(self, test_data, test_labels):
        """Evaluate the model with test data."""
        if not self.current_model:
            return {"error": "No model loaded"}
        
        predictions = self.current_model.predict(test_data)
        accuracy = accuracy_score(test_labels, predictions)
        report = classification_report(test_labels, predictions, output_dict=True)
        
        return {
            "accuracy": accuracy,
            "report": report
        }

    def predict(self, input_data):
        """Make predictions using the deployed model."""
        if not self.current_model:
            return {"error": "No model deployed"}
        
        predictions = self.current_model.predict(input_data)
        return {"predictions": predictions.tolist()}
    
    def predict_image(self, base64_image):
        """Predict using YOLO model with base64-encoded image data."""
        if not self.current_model:
            return {"error": "No YOLO model deployed"}

        try:
            # Decode base64 image
            image_data = base64.b64decode(base64_image)
            np_image = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

            # Perform YOLO prediction
            results = self.current_model.predict(image)  # Adapt for your YOLO API
            return {"predictions": results}
        except Exception as e:
            return {"error": f"Error processing image: {str(e)}"}


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
                parts = command.split('|')
                test_data = eval(parts[1])
                test_labels = eval(parts[2])
                response = self.model_manager.evaluate_model(test_data, test_labels)
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
                        parts = data.split('|')
                        test_data = eval(parts[0])
                        test_labels = eval(parts[1])
                        result = self.model_manager.evaluate_model(test_data, test_labels)
                        response = {"status": "success", "data": result}
                    elif channel == 'predict':
                        # Expect the `data` to be the path to an image file or base64 image
                        image_path = 'D:/Code/Project/MediScan/test-image/1.png'
                        result = model_manager.predict_from_path(image_path)
                        print(result)  # This should give you either predictions or an error message

                        # parts = data.split('|')
                        # if len(parts) < 2:
                        #     response = {"status": "error", "message": "Missing image data"}
                        # else:
                        #     # Check if the data is base64 or file path
                        #     if data.startswith('data:image/'):  # Check for base64 string
                        #         base64_image = parts[1]
                        #         result = self.model_manager.predict_image(base64_image)
                        #     else:
                        #         image_path = data  # Treat as file path
                        #         if not os.path.exists(image_path):
                        #             response = {"status": "error", "message": f"File not found: {image_path}"}
                        #         else:
                        #             result = self.model_manager.predict_from_path(image_path)
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
