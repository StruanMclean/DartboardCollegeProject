from typing import List, Tuple
from fastapi import FastAPI, WebSocket
from src.camrea import Camera
from src.helpers.logger import logger

app = FastAPI()

cam1 = Camera(0)
cam2 = Camera(1)

@app.get("/")
async def root():
    return {"message": "Running"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")

    global scores
    scores: List[Tuple[int, str]] = []

    while True:
        await websocket.receive_text()
        await websocket.send_text(f"Message text was:")