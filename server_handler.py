import json
import socket
import sys
import threading
import signal
from model_manager import MLModelManager

# ✅ ตัวแปรควบคุมการทำงานของเซิร์ฟเวอร์
server_running = True
tcp_server = None  

class TCPServer:
    def __init__(self, host='0.0.0.0', port=5001, model_manager=None):
        self.host = host
        self.port = port
        self.model_manager = model_manager
        self.server_socket = None

    def handle_command(self, command, conn):
        """📌 ดำเนินการตามคำสั่งที่ได้รับ"""
        response = {"error": "Unknown command"}

        try:
            # ✅ แปลงข้อความเป็น JSON
            try:
                data = json.loads(command)
            except json.JSONDecodeError:
                response = {"error": "Invalid JSON format"}
                conn.sendall((json.dumps(response) + "\n").encode('utf-8'))
                return
            
            cmd_type = data.get("command", "").lower()
            p_id = data.get("p_id", "").strip()  # รับค่า project_id
            
            # ✅ ตรวจสอบว่า p_id ถูกส่งมาหรือไม่
            if not p_id:
                response = {"error": "Missing project ID (p_id). Use: {'command': 'train', 'p_id': 'proj_001'}"}
                conn.sendall((json.dumps(response) + "\n").encode('utf-8'))
                return

            if cmd_type == 'train':
                print(f"🚀 Running Training Command for project: {p_id}")
                mode = data.get("mode")

                if not mode:
                    response = {"error": "Missing mode for training. Use: {'command': 'train', 'mode': 'detect/segment/classify', 'p_id': 'proj_001'}"}
                else:
                    try:
                        result = self.model_manager.train_model(p_id, mode)  # ✅ ส่ง `p_id` ไป train_model
                        response = {"status": "success", "data": result}
                    except (ValueError, RuntimeError) as e:
                        response = {"error": str(e)}
                    except Exception as e:
                        response = {"error": f"Unexpected error: {str(e)}"}

                print(f"📌 Training Response: {response}")

            elif cmd_type == 'evaluate':
                print(f"🔍 Running Evaluation for project: {p_id}")
                eval_type = data.get("eval_type", "test").lower()
                mode = data.get("mode")
                model_name = data.get("model_name")

                if not mode or not model_name:
                    response = {"error": "Missing required parameters. Use: {'command': 'evaluate', 'eval_type': 'val/test', 'mode': 'classify', 'model_name': 'model.pt'}"}
                elif eval_type not in ["val", "test"]:
                    response = {"error": "Invalid evaluation type. Use 'val' or 'test'."}
                else:
                    try:
                        result = self.model_manager.evaluate_model(p_id, mode, model_name, eval_type)  # ✅ ส่ง `p_id`
                        response = {"status": "success", "data": result}
                    except (ValueError, FileNotFoundError, RuntimeError) as e:
                        response = {"error": str(e)}

                print(f"📌 Evaluation Response: {response}")

            elif cmd_type == 'predict':
                print(f"🔍 Running Predict Command for project: {p_id}")
                image_path = data.get("image_path", "").strip()
                if not image_path:
                    response = {"error": "Missing image path. Use: {'command': 'predict', 'p_id': 'proj_001', 'image_path': 'path/to/image.jpg'}"}
                else:
                    try:
                        result = self.model_manager.predict_from_path(p_id, image_path)  # ✅ ส่ง `p_id`
                        response = {"status": "success", "data": result}
                    except FileNotFoundError as e:
                        response = {"error": str(e)}
                    except RuntimeError as e:
                        response = {"error": f"Prediction failed: {str(e)}"}
                    except Exception as e:
                        response = {"error": f"An unexpected error occurred: {str(e)}"}

                print(f"📌 Prediction Response: {response}")

            elif cmd_type == 'deploy':
                model_name = data.get("model_name")

                if not model_name:
                    response = {"error": "Missing model name. Use: {'command': 'deploy', 'p_id': 'proj_001', 'model_name': 'model.pt'}"}
                else:
                    try:
                        self.model_manager.load_model(p_id, model_name)  # ✅ ส่ง `p_id`
                        response = {"status": "success", "message": f"Model '{model_name}' deployed successfully"}
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
        """📌 รับคำสั่งจาก Client และส่งไป `handle_command()`"""
        try:
            with conn:
                while server_running:
                    data = conn.recv(4096)
                    if not data:
                        break
                    command = data.decode('utf-8').strip()
                    self.handle_command(command, conn)
        except Exception as e:
            print(f"Error handling client: {e}")

    def start_server(self):
        """📌 เริ่ม TCP Server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"✅ TCP server running on {self.host}:{self.port}")

        while server_running:
            try:
                conn, addr = self.server_socket.accept()
                print(f"📌 Connected by {addr}")
                threading.Thread(target=self.handle_client, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Server error: {e}")
                break

    def stop_server(self):
        """📌 หยุด TCP Server"""
        global server_running
        server_running = False
        if self.server_socket:
            self.server_socket.close()
            print("✅ TCP server stopped.")

### --- Start Servers ---
def start_servers():
    """📌 เริ่ม TCP Server"""
    global tcp_server
    model_manager = MLModelManager()
    tcp_server = TCPServer(model_manager=model_manager)
    tcp_thread = threading.Thread(target=tcp_server.start_server, daemon=True)
    tcp_thread.start()

    print("✅ Server started successfully.")
    return tcp_server

### --- Handle Exit ---
def signal_handler(sig, frame):
    """📌 Handle SIGINT (Ctrl+C) for graceful shutdown."""
    print("\n🔴 Terminating server...")
    tcp_server.stop_server()
    sys.exit(0)

# ✅ Register Signal Handler
signal.signal(signal.SIGINT, signal_handler)
