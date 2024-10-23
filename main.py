from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
import os
from datetime import datetime
from configs import Config
from db.models import db, ImageMetadata

app = Flask(__name__)
socketio = SocketIO(app)

# Directory for storing uploaded files
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config.from_object(Config)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Configure the SQLite database
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///metadata.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# Create the database table(s)
with app.app_context():
    db.create_all()

# Directory to save uploaded files
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Upload
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    image = request.files['file']

    if image.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    
    # if image.mimetype not in ['image/jpeg', 'image/png']:
    #     return "Invalid file type", 400


    # Save file to uploads folder
    image_name = image.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    image.save(file_path)

    # Metadata
    metadata = ImageMetadata(
        filename=image_name,
        upload_time=datetime.now(),
        # mime_type=image.mimetype,
        file_size=len(image.read())
    )

    db.session.add(metadata)
    db.session.commit()

    # Reset file pointer after reading
    image.seek(0)

    socketio.emit('upload_status', {'status': 'Image uploaded', 'image_name': image_name})

    return jsonify({
        "message": "File uploaded successfully",
        "metadata": {
            "filename": image_name,
            # "mime_type": image.mime_type,
            "file_size": len(image.read()),
            "upload_time": metadata.upload_time
        }
    }), 200

# --------- WebSocket -----------
@socketio.on('connect')
def handle_connect():
    print("Client connected!")
    socketio.emit('server_response', {'message': 'Connected to WebSocket'})

@socketio.on('process_image')
def handle_image_processing(data):
    classification_result = data['classification_result']

    # data use to be added to DB later
    feedback = data['feedback']
    image_id = data['image_id']

    socketio.emit('classification_result', {'result': classification_result})

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected!")

if __name__ == '__main__':
    # Use eventlet for async WebSocket handling
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
