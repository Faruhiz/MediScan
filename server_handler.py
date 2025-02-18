import json
import os
import socket
import sys
import threading
import signal

import redis
from model_manager import MLModelManager
import model_manager

# redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
server_running = True
tcp_server = None  # เก็บ instance ของ TCPServer ไว้
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

class TCPServer:
    def __init__(self, host='0.0.0.0', port=5001, model_manager=None):
        self.host = host
        self.port = port
        self.model_manager = model_manager
        self.server_socket = None

    def handle_command(self, command, conn):
        """Process a single command for training, evaluating, predicting, and deploying a model."""
        response = {"error": "Unknown command"}
        try:
            parts = command.split('|')
            cmd_type = parts[0]  # คำสั่งหลัก เช่น train, evaluate, predict, deploy

            if cmd_type == 'train':
                print("🚀 Running Training Command...")
                if len(parts) < 2:
                    response = {"error": "Missing mode for training. Use: train|detect, train|segment, train|classify"}
                else:
                    mode = parts[1].strip()
                    result = self.model_manager.train_model(mode)
                    response = result

            elif cmd_type == 'evaluate':
                print("🔍 Running Evaluation Mode...")
                if len(parts) < 3:
                    response = {"error": "Missing required parameters. Use: evaluate|mode|model_path"}
                else:
                    mode = parts[1].strip()
                    model_name = parts[2].strip()

                    try:
                        # ✅ Call evaluate_model() and using `raise` to easily handle error
                        result = self.model_manager.evaluate_model(mode, model_name)
                        response = {"status": "success", "data": result}
                    except FileNotFoundError as e:
                        response = {"error": str(e)}
                    except RuntimeError as e:
                        response = {"error": f"Evaluation failed: {str(e)}"}

            elif cmd_type == 'predict':
                if len(parts) < 2:
                    response = {"error": "Missing image path. Use: predict|image_path"}
                else:
                    image_name = parts[1].strip()

                    try:
                        # เรียก `predict_from_path()` และใช้ `raise` ในการจัดการ error
                        result = self.model_manager.predict_from_path(image_name)
                        response = {"status": "success", "data": result}
                    except FileNotFoundError as e:
                        response = {"error": str(e)}
                    except RuntimeError as e:
                        response = {"error": f"Prediction failed: {str(e)}"}

            elif cmd_type == 'deploy':
                if len(parts) < 2:
                    response = {"error": "Missing model name. Use: deploy|model_name"}
                else:
                    model_name = parts[1].strip()
                    try:    
                        self.model_manager.load_model(model_name)
                        response = {"status": "success", "message": f"Model '{model_name}' deployed successfully"}
                        print(response)
                    except FileNotFoundError as e:
                        response = {"error": str(e)}
                    except Exception as e:
                        response = {"error": f"Failed to deploy model: {str(e)}"}
        
            else:
                response = {"error": "Invalid command"}

        except Exception as e:
            response = {"error": f"Error processing command: {str(e)}"}

        conn.sendall((json.dumps(response) + "\n").encode('utf-8'))

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
            print("✅ TCP server stopped.")


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
                parts = data.split('|')

                try:
                    if channel == 'train':
                        if len(parts) < 2:
                            response = {"status": "error", "message": "Missing mode for training. Use: train|detect, train|segment, train|classify"}
                        else:
                            mode = parts[1].strip()
                            result = self.model_manager.train_model(mode)
                            response = {"status": "success", "data": result}

                    elif channel == 'evaluate':
                        if len(parts) < 3:
                            response = {"status": "error", "message": "Missing required parameters. Use: evaluate|mode|model_path"}
                        else:
                            mode = parts[1].strip()
                            model_name = parts[2].strip()

                            try:
                                # ✅ เรียก `evaluate_model()` และใช้ `raise` ในการจัดการ error
                                result = self.model_manager.evaluate_model(mode, model_name)
                                response = {"status": "success", "data": result}
                            except FileNotFoundError as e:
                                response = {"error": str(e)}
                            except RuntimeError as e:
                                response = {"error": f"Evaluation failed: {str(e)}"}

                    elif channel == 'predict':
                        if len(parts) < 2:
                            response = {"status": "error", "message": "Missing image path. Use: predict|image_path"}
                        else:
                            image_name = parts[1].strip()
                            
                            try:
                                # เรียก `predict_from_path()` และใช้ `raise` ในการจัดการ error
                                result = self.model_manager.predict_from_path(image_name)
                                response = {"status": "success", "data": result}
                            except FileNotFoundError as e:
                                response = {"error": str(e)}
                            except RuntimeError as e:
                                response = {"error": f"Prediction failed: {str(e)}"}
                            
                    elif channel == 'deploy':
                        if len(parts) < 2:
                            response = {"status": "error", "message": "Missing model name. Use: deploy|model_name"}
                        else:
                            model_name = parts[1].strip()
                            try:    
                                self.model_manager.load_model(model_name)
                                response = {"status": "success", "message": f"Model '{model_name}' deployed successfully"}
                                print(response)
                            except FileNotFoundError as e:
                                response = {"error": str(e)}
                            except Exception as e:
                                response = {"error": f"Failed to deploy model: {str(e)}"}
                           
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
def start_servers():
    """Start TCP Server and Redis Handler in separate daemon threads."""
    model_manager = MLModelManager()

    # Start Redis Handler
    redis_handler = RedisHandler(redis_client, model_manager)
    redis_thread = threading.Thread(target=redis_handler.listen, daemon=True)
    redis_thread.start()

    # Start TCP Server
    tcp_server = TCPServer(model_manager=model_manager)
    tcp_thread = threading.Thread(target=tcp_server.start_server, daemon=True)
    tcp_thread.start()

    print("✅ Servers started successfully.")
    return tcp_server  # Return TCP Server instance for shutdown handling

def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) for graceful shutdown."""
    print("\nTerminated servers process...")
    tcp_server.stop_server()  # 🔥 เพิ่มคำสั่งให้หยุดเซิร์ฟเวอร์
    sys.exit(0)