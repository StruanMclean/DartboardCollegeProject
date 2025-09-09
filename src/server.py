from typing import List, Tuple
from fastapi import FastAPI, WebSocket
from camrea import Camera
from helpers.logger import logger
from model.download import download_model

app = FastAPI()

cam1 = Camera(0)
cam2 = Camera(1)

MODEL_LOADED: Tuple[bool, str] = download_model()

@app.get("/")
async def root():
    return {"message": "Running"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):    
    await websocket.accept()
    logger.info("WebSocket client connected")

    scores: List[Tuple[int, str]] = []

    while True:
        await websocket.receive_text()
        await websocket.send_text(f"Message text was:")