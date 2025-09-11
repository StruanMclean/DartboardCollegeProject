import math
from typing import List
import cv2
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from camrea import Camera
from darttracker import DartTracker
from dataclass.classes import Classes
from dataclass.dartboard import DartboardScorer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from helpers.logger import logger
from helpers.vect import Vector
import numpy as np

class Predict:
    def __init__(self, model_path):
        setup_logger()

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.007
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
        cfg.INPUT.MIN_SIZE_TEST = 800
        cfg.INPUT.MAX_SIZE_TEST = 1333

        self.predictor = DefaultPredictor(cfg)
        self.metadata = MetadataCatalog.get("darts_train")

        self.scorer = DartboardScorer()
        self.camera_calibrations = {}
        
        self.dart_trackers = {}

    @staticmethod
    def __get_radius(calib_points):
        return int(abs(calib_points["left"].x - calib_points["right"].x) / 2)
    
    @staticmethod
    def __get_points(output, darts_only=False):
        instances = output["instances"]
        
        if len(instances) == 0:
            print("No instances detected")
            return [], {}, []
            
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()
        scores = instances.scores.cpu().numpy()

        calibration_points = {}
        darts = []
        dart_scores = []

        print(f"Found {len(instances)} detections:")
        for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            
            print(f"  Detection {i}: class={cls}, score={score:.3f}, center=({center_x:.1f}, {center_y:.1f})")
            
            if cls == Classes.objects:
                continue
            elif cls == Classes.top:
                calibration_points["top"] = Vector(center_x, center_y)
                print(f"    -> TOP point")
            elif cls == Classes.left:
                calibration_points["left"] = Vector(center_x, center_y)
                print(f"    -> LEFT point")
            elif cls == Classes.right:
                calibration_points["right"] = Vector(center_x, center_y)
                print(f"    -> RIGHT point")
            elif cls == Classes.bottom:
                calibration_points["bottom"] = Vector(center_x, center_y)
                print(f"    -> BOTTOM point")
            elif cls == Classes.dart:
                darts.append(Vector(center_x, center_y))
                dart_scores.append(float(score))
                print(f"    -> DART at: ({center_x:.1f}, {center_y:.1f})")

        if not darts_only:
            calibration_points["center"] = Vector(
                (calibration_points["left"].x - calibration_points["right"].x) / 2,
                (calibration_points["top"].y - calibration_points["bottom"].y) / 2
            )

        if darts_only:
            return darts, {}, dart_scores
        else:
            return darts, calibration_points, dart_scores
        
    def get_score(self, cam_num, darts: list[Vector]):
        if not darts:
            return []
        
        if cam_num not in self.camera_calibrations:
            print(f"No calibration found for camera {cam_num}")
            return []
            
        calibration = self.camera_calibrations[cam_num]
        homography = calibration['homography_matrix']

        scores = []

        if homography is not None:
            for dart in darts:
                transformed_dart = self.__get_point_on_dartboard(dart, homography)

                distance = math.sqrt(
                    transformed_dart.x * transformed_dart.x +
                    transformed_dart.y * transformed_dart.y
                )

                angle_rad = math.atan2(-transformed_dart.y, transformed_dart.x)
                angle_deg = math.degrees(angle_rad)
                
                dartboard_angle = (90 - angle_deg) % 360
                
                segment_angle = 18.0
                segment_index = int(dartboard_angle / segment_angle) % 20
                
                dartboard_numbers = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
                number = dartboard_numbers[segment_index]

                if distance <= self.scorer.double_bull_radius:
                    scores.append((50, "Double Bulls Eye"))
                elif distance <= self.scorer.bull_radius:
                    scores.append((25, "Bulls Eye"))
                elif distance <= self.scorer.treble_inner_radius:
                    scores.append((number, str(number)))
                elif distance <= self.scorer.treble_outer_radius:
                    scores.append((number * 3, f"T{number}"))
                elif distance <= self.scorer.double_inner_radius:
                    scores.append((number, str(number)))
                elif distance <= self.scorer.double_outer_radius:
                    scores.append((number * 2, f"D{number}"))
                else:
                    scores.append((0, "Miss"))

            return scores
        else:
            print("No homography matrix available")
            return []
        
    def get_dart_tracker(self, cam_num: int) -> DartTracker:
        if cam_num not in self.dart_trackers:
            self.dart_trackers[cam_num] = DartTracker()

        return self.dart_trackers[cam_num]

    def compute_homography(self, calib_points):
        if len(calib_points) < 4:
            logger.warning(f"Need exactly 4 calibration points for homography, got {len(calib_points)}")
            return None
        
        outer_radius = self.scorer.double_outer_radius 

        logger.info(f"Using outer radius: {outer_radius} for homography computation")

        dartboard_points = np.array([
            [0, -outer_radius],
            [-outer_radius, 0],
            [outer_radius, 0],
            [0, outer_radius]
        ], dtype=np.float32)

        try:
            image_points = np.array([
                [calib_points['top'].x, calib_points['top'].y],
                [calib_points['left'].x, calib_points['left'].y],
                [calib_points['right'].x, calib_points['right'].y],
                [calib_points['bottom'].x, calib_points['bottom'].y]
            ], dtype=np.float32)
        except KeyError as e:
            logger.warning(f"Missing calibration point: {e}")
            return None
        
        try:
            H, _ = cv2.findHomography(image_points, dartboard_points, cv2.RANSAC, 5.0)

            if H is not None:
                return H
            else:
                print("Failed to compute homography - points may be collinear")
                return None
        except Exception as e:
            print(f"Failed to compute homography: {e}")
            return None
        
    def calabrate(self, cam: Camera):
        output = self.predictor(cam.start_frame)
        _, calib_points, _ = self.__get_points(output, darts_only=False)

        print("Calibration points found:", calib_points)

        if len(calib_points) < 4:
            print("Can't find all required points. Need: top, bottom, left, right, center")
            print(f"Found only: {list(calib_points.keys())}")
            return False

        required_points = ["top", "bottom", "left", "right", "center"]
        missing_points = [p for p in required_points if p not in calib_points]
        if missing_points:
            print(f"Missing required calibration points: {missing_points}")
            return False

        homography = self.compute_homography(calib_points, calib_points["center"])

        if homography is None:
            print("Failed to compute homography matrix")
            return False

        self.camera_calibrations[cam.cam_num] = {
            "dartboard_center": calib_points["center"],
            "calib_points": calib_points,
            "homography_matrix": homography,
            "dartboard_radius": self.__get_radius(calib_points)
        }

        print("Calibration successful:", self.camera_calibrations[cam.cam_num])
        return True
    
    def apply_nms_to_darts(self, darts: List[Vector], scores: List[float], nms_threshold: float = 0.3) -> List[Vector]:
        if len(darts) <= 1:
            return darts
        
        boxes = []
        for dart in darts:
            box_size = 20
            x = dart.x - box_size // 2
            y = dart.y - box_size // 2
            boxes.append([x, y, box_size, box_size])
        
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.0, nms_threshold)
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [darts[i] for i in indices]
        else:
            return []
    
    def reset_dart_tracker(self, cam_num: int):
        if cam_num in self.dart_trackers:
            del self.dart_trackers[cam_num]

    def reset_all_dart_trackers(self):
        self.dart_trackers.clear()
    
    def main(self, cam: Camera):
        frame = cam.get_cur_frame()
        outputs = self.predictor(frame)

        all_darts, _, dart_confidence_scores = self.__get_points(outputs, darts_only=True)
        
        if not all_darts:
            self.reset_dart_tracker(cam.cam_num)
            return []
        
        if all_darts and dart_confidence_scores:
            filtered_darts = self.apply_nms_to_darts(all_darts, dart_confidence_scores, nms_threshold=0.3)
        else:
            filtered_darts = all_darts

        dart_tracker = self.get_dart_tracker(cam.cam_num)
        new_darts = dart_tracker.update(filtered_darts)

        scores = []

        if new_darts:
            print(f"Found {len(new_darts)}")

            scores = self.get_score(cam.cam_num, new_darts)

            for dart in new_darts:
                dart_tracker.mark_as_scored(dart)

            total_score = 0
            for i, (score, description) in enumerate(scores):
                print(f"New Dart {i + 1}: {score} points ({description})")
                total_score += score

            if len(scores) > 0:
                print(f"Total new score: {total_score}")

        # Some detectron shit
        v = Visualizer(frame[:, :, ::-1], 
                      metadata=self.metadata, 
                      scale=0.8)
        
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        image = out.get_image()[:, :, ::-1]
        image = np.ascontiguousarray(image, dtype=np.uint8)

        if cam.cam_num in self.camera_calibrations:
            center = self.camera_calibrations[cam.cam_num]['dartboard_center']
            cv2.circle(image, (int(center.x), int(center.y)), 5, (255, 0, 255), -1)
            cv2.putText(image, "CENTER", 
                       (int(center.x + 10), int(center.y - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        if new_darts and scores:
            for i, (dart, (score, description)) in enumerate(zip(new_darts, scores)):
                cv2.circle(image, (int(dart.x), int(dart.y)), 8, (0, 255, 0), 3)
                cv2.putText(image, f"NEW: {score}", 
                           (int(dart.x), int(dart.y - 20)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        all_tracked = dart_tracker.get_unscored_darts()
        for dart in all_tracked:
            if dart not in new_darts:
                cv2.circle(image, (int(dart.x), int(dart.y)), 5, (255, 255, 0), 2)
                cv2.putText(image, "TRACKED", 
                           (int(dart.x), int(dart.y + 15)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        cv2.imwrite(f"current{cam.cam_num}_scored.jpg", image)

        return scores
