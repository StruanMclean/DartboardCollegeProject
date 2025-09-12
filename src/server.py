from typing import List, Tuple
from fastapi import FastAPI, WebSocket
from camrea import Camera
from helpers.filter import filter_reasonable_scores
from helpers.logger import logger
from model.download import download_model
from prediction import Predict
from helpers.filter import compare_duo_cam

app = FastAPI()

MODEL_LOADED, MODEL_DIR = download_model()

predicter = Predict(str(MODEL_DIR))
cam1 = Camera(0)
cam2 = Camera(1)
last_validated_scores = []
consecutive_empty_readings = 0

def get_dart_count_from_cameras() -> Tuple[int, int]:
    try:
        count1 = predicter.get_dart_count(cam1.cam_num) if hasattr(predicter, 'get_dart_count') else 0
        count2 = predicter.get_dart_count(cam2.cam_num) if hasattr(predicter, 'get_dart_count') else 0
        return count1, count2
    except Exception as e:
        logger.error(f"Error getting dart counts: {e}")
        return 0, 0


@app.get("/")
async def root():
    return {"message": "Running"}
    
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket): 
    global last_validated_scores, consecutive_empty_readings  

    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                scores1 = predicter.main(cam1)
                scores2 = predicter.main(cam2)

                scores1_filtered = filter_reasonable_scores(scores1) if scores1 else []
                scores2_filtered = filter_reasonable_scores(scores2) if scores2 else []

                validated_scores = compare_duo_cam(scores1_filtered, scores2_filtered)

                count1, count2 = get_dart_count_from_cameras()
                max_dart_count = max(count1, count2)

                if (
                    last_validated_scores and 
                    not validated_scores and 
                    max_dart_count == 0
                ):
                    consecutive_empty_readings += 1
                    
                    if consecutive_empty_readings >= 2:
                        logger.info("Darts removed from board")
                        validated_scores = []
                        consecutive_empty_readings = 0
                    else:
                        validated_scores = last_validated_scores
                else:
                    consecutive_empty_readings = 0

                if validated_scores:
                    last_validated_scores - validated_scores

                response = {
                    "scores": validated_scores,
                    "dart_count": max_dart_count,
                    "camera1_count": count1,
                    "camera2_count": count2,
                    "camera1_scores": scores1_filtered,
                    "camera2_scores": scores2_filtered,
                }
                
                await websocket.send_json(response)

            except Exception as e:
                logger.error(f"Error processing camera data: {e}")
                # Send error response
                await websocket.send_json({
                    "scores": [],
                    "dart_count": 0,
                    "error": str(e)
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket client disconnected")
