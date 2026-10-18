import json
import base64
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Import your high-level dispatcher
from src.vision_engine import VisionManager

app = FastAPI()

# Mount the static directory to serve your HTML/JS/CSS without external CDNs
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the two-tier routing manager
manager = VisionManager()

@app.get("/")
async def get_dashboard():
    with open("static/index.html", "r") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def vision_stream(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to edge stream.")
    
    try:
        while True:
            # Receive incoming data from the JavaScript canvas
            payload = await websocket.receive_text()
            
            # 1. Handle JSON Control Messages (Model/Task Switching)
            if payload.startswith('{'):
                command = json.loads(payload)
                
                # The two-tier pipeline switch
                if command.get("type") == "switch_pipeline":
                    task = command.get("task")
                    model = command.get("model")
                    manager.switch_pipeline(task, model)
                    
                # Zero-shot class text updates
                elif command.get("type") == "update_classes":
                    custom_classes = command.get("classes", [])
                    manager.get_active_engine().set_custom_classes(custom_classes)
                    
            # 2. Handle Base64 Image Frames
            else:
                # Strip the data URI header and decode the raw JPEG frame
                encoded_data = payload.split(',')[1]
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Fetch the active isolated task engine and process the frame
                active_engine = manager.get_active_engine()
                predictions = active_engine.process_frame(frame)
                
                # Return the structured JSON to the frontend rendering loop
                await websocket.send_json({"boxes": predictions})
                
    except WebSocketDisconnect:
        print("Client disconnected from edge stream.")
    except Exception as e:
        print(f"WebSocket Error: {e}")