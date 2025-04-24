from fastapi import FastAPI, Body, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import socket
import json
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI(title="MedSight TCP Interface (via Swagger)", 
              description="Wrapper to test TCP commands via HTTP (Swagger)"
              )

TCP_HOST = "127.0.0.1"
TCP_PORT = 5001

# run api
# uvicorn swagger_api:app --reload --port 8000
# http://127.0.0.1:8000/docs

# ✅ Persistent TCP Client Class
class PersistentTCPClient:
    def __init__(self, host="127.0.0.1", port=5001):
        self.host = host
        self.port = port
        self.socket = None
        self._connect()

    def _connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            print(f"✅ Connected to TCP server at {self.host}:{self.port}")
        except Exception as e:
            print(f"❌ Failed to connect to TCP server: {e}")
            self.socket = None

    def send_command(self, command_dict):
        if not self.socket:
            return {"error": "TCP socket is not connected"}

        try:
            message = json.dumps(command_dict) + "\n"
            self.socket.sendall(message.encode("utf-8"))

            response_data = self.socket.recv(8192).decode("utf-8")
            return json.loads(response_data)
        except Exception as e:
            return {"error": f"Exception in send_command: {str(e)}"}

    def close(self):
        try:
            if self.socket:
                self.socket.close()
                self.socket = None
                print("🛑 TCP socket closed.")
        except Exception as e:
            print(f"⚠️ Error closing socket: {e}")

# ✅ Global persistent client instance
tcp_client = PersistentTCPClient()

# 📌 Pydantic Models for Request
class TrainRequest(BaseModel):
    project_id: str
    model_name: str
    mode: str  # detect / segment / classify

class EvaluateRequest(BaseModel):
    project_id: str
    model_name: str

class DeployRequest(BaseModel):
    project_id: str
    model_name: str

class PredictRequest(BaseModel):
    project_id: str
    image_name: str

# Global dictionary to track API call counts
api_call_counts = {
    "/train": 0,
    "/evaluate": 0,
    "/deploy": 0,
    "/predict": 0,
}

def track_api_usage(endpoint: str):
    """Increment the call count for the given endpoint."""
    if endpoint in api_call_counts:
        api_call_counts[endpoint] += 1
    else:
        api_call_counts[endpoint] = 1

# ✅ Swagger Routes
@app.post("/train")
def train_model(req: TrainRequest):
    track_api_usage("/train")
    cmd = {
        "command": "train",
        "project_id": req.project_id,
        "model_name": req.model_name,
        "mode": req.mode
    }
    return tcp_client.send_command(cmd)

@app.post("/evaluate")
def evaluate_model(req: EvaluateRequest):
    track_api_usage("/evaluate")
    cmd = {
        "command": "evaluate",
        "project_id": req.project_id,
        "model_name": req.model_name,
    }
    return tcp_client.send_command(cmd)

@app.post("/deploy")
def deploy_model(req: DeployRequest):
    track_api_usage("/deploy")
    cmd = {
        "command": "deploy",
        "project_id": req.project_id,
        "model_name": req.model_name
    }
    return tcp_client.send_command(cmd)

@app.post("/predict")
def predict_image(req: PredictRequest):
    track_api_usage("/predict")
    cmd = {
        "command": "predict",
        "project_id": req.project_id,
        "image_name": req.image_name
    }
    return tcp_client.send_command(cmd)

@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    """Generate a dashboard with API usage statistics."""
    # Generate a bar chart using Matplotlib
    endpoints = list(api_call_counts.keys())  # Ensure these are strings
    counts = list(api_call_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(endpoints, counts, color='skyblue')
    plt.xlabel("API Endpoints")
    plt.ylabel("Call Counts")
    plt.title("API Usage Dashboard")
    plt.xticks(rotation=45, ha="right")  # Ensure proper rotation for readability

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    # Encode the image to base64
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()

    # Return an HTML page with the embedded image
    html_content = f"""
    <html>
        <head>
            <title>API Usage Dashboard</title>
        </head>
        <body>
            <h1>API Usage Dashboard</h1>
            <img src="data:image/png;base64,{image_base64}" alt="API Usage Dashboard">
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/close")
def close_connection():
    tcp_client.close()
    return {"message": "TCP connection closed"}