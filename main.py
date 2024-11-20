import os
import socket
import threading
import redis
import pickle
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

# Global variable to store the trained model
current_model = None

### --- Helper Functions ---

def save_model(model, model_name="ml_model.pkl"):
    """Save the trained model to a file."""
    model_path = os.path.join(MODEL_FOLDER, model_name)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return model_path

def load_model(model_name="ml_model.pkl"):
    """Load a saved model."""
    model_path = os.path.join(MODEL_FOLDER, model_name)
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def train_model():
    """Train a new ML model."""
    global current_model

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
    model_path = save_model(model)
    current_model = model  # Keep the model in memory for immediate use
    
    return {
        "message": "Model trained successfully",
        "accuracy": accuracy,
        "model_path": model_path
    }

def evaluate_model(test_data, test_labels):
    """Evaluate the model with test data."""
    global current_model
    if not current_model:
        return {"error": "No model loaded"}
    
    predictions = current_model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, output_dict=True)
    
    return {
        "accuracy": accuracy,
        "report": report
    }

def predict(input_data):
    """Make predictions using the deployed model."""
    global current_model
    if not current_model:
        return {"error": "No model deployed"}
    
    predictions = current_model.predict(input_data)
    return {"predictions": predictions.tolist()}

### --- Thread Handling for Commands ---
def handle_command(command, conn):
    """Process a single command in a thread."""
    response = {"error": "Unknown command"}

    try:
        if command == 'train':
            # Train the model
            result = train_model()
            response = result
        elif command.startswith('evaluate'):
            # Example: evaluate|[[5.1, 3.5, 1.4, 0.2]]|[0]
            parts = command.split('|')
            test_data = eval(parts[1])
            test_labels = eval(parts[2])
            response = evaluate_model(test_data, test_labels)
        elif command.startswith('predict'):
            # Example: predict|[[5.1, 3.5, 1.4, 0.2]]
            parts = command.split('|')
            input_data = eval(parts[1])
            response = predict(input_data)
        elif command.startswith('deploy'):
            # Deploy a saved model
            model_name = command.split('|')[1] if '|' in command else "ml_model.pkl"
            try:
                global current_model
                current_model = load_model(model_name)
                response = {"message": f"Model '{model_name}' deployed successfully"}
            except Exception as e:
                response = {"error": f"Failed to deploy model: {str(e)}"}
        else:
            response = {"error": "Invalid command"}
    except Exception as e:
        response = {"error": f"Error processing command: {str(e)}"}

    # Send response back to the client
    conn.sendall((str(response) + "\n").encode('utf-8'))

def handle_client(conn):
    """Handle TCP client requests."""
    try:
        with conn:
            while True:
                data = conn.recv(4096)
                if not data:
                    break

                # Parse client command
                command = data.decode('utf-8').strip()

                # Create a new thread for the command
                command_thread = threading.Thread(target=handle_command, args=(command, conn))
                command_thread.start()
    except Exception as e:
        print(f"Error handling client: {e}")

### --- TCP Server ---

def start_tcp_server():
    """Start the TCP server."""
    HOST = '0.0.0.0'
    PORT = 5001

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)  # Listen for up to 5 connections
    print(f"TCP server running on {HOST}:{PORT}")

    while True:
        conn, addr = server_socket.accept()
        print(f"Connected by {addr}")
        threading.Thread(target=handle_client, args=(conn,)).start()

### --- Redis Subscriber ---

def redis_subscriber():
    """Listen for Redis messages and handle them."""
    pubsub = redis_client.pubsub()
    pubsub.subscribe(['train', 'evaluate', 'predict', 'deploy'])

    print("Listening for Redis messages...")
    for message in pubsub.listen():
        if message['type'] == 'message':
            channel = message['channel'].decode('utf-8')
            data = message['data'].decode('utf-8')
            print(f"Redis message on {channel}: {data}")

            response = {"status": "error", "message": "Unknown command"}

            try:
                if channel == 'train':
                    # Train the model and send response
                    result = train_model()
                    response = {"status": "success", "data": result}
                elif channel == 'evaluate':
                    # Example: data sent as "test_data|test_labels"
                    parts = data.split('|')
                    test_data = eval(parts[0])
                    test_labels = eval(parts[1])
                    result = evaluate_model(test_data, test_labels)
                    response = {"status": "success", "data": result}
                elif channel == 'predict':
                    # Example: data sent as "input_data"
                    input_data = eval(data)
                    result = predict(input_data)
                    response = {"status": "success", "data": result}
                elif channel == 'deploy':
                    # Example: data sent as "model_name"
                    model_name = data if data else "ml_model.pkl"
                    try:
                        global current_model
                        current_model = load_model(model_name)
                        response = {"status": "success", "message": f"Model '{model_name}' deployed successfully"}
                    except Exception as e:
                        response = {"status": "error", "message": f"Failed to deploy model: {str(e)}"}
                else:
                    response = {"status": "error", "message": "Invalid command"}
            except Exception as e:
                response = {"status": "error", "message": f"Error processing command: {str(e)}"}
    
            # Publish response to a dedicated response channel
            redis_client.publish('response', str(response))


### --- Main Execution ---

if __name__ == '__main__':
    # Start the Redis subscriber in a separate thread
    threading.Thread(target=redis_subscriber, daemon=True).start()

    # Start the TCP server
    start_tcp_server()
