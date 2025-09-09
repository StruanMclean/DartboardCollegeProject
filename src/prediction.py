from detectron2.utils.logger import setup_logger
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

    # Implmented based on https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
    # I dont understand the math im too spesh i guess
    def find_homography(src_points, dist_points):
        if src_points.shape != dist_points.shape:
            logger.error("Src points do not match Dist points shape")
            return None
        
        n_points = src_points.shape[0]
        if n_points >= 4:
            logger.warning(f"Need exactly 4 calibration points for homography, got {len(n_points)}")
            return None
        
        A = [] # Matrix

        for i in range(n_points):
            x, y = src_points[i]
            u, v = dist_points[i]
            A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
            A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])

        A = np.array(A)
        U, S, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        H = H / H[2, 2]

        return H

    # Implmented based on https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
    # Same again :(
    def warp_image(self, img, H, dist_shape):
        height, width = dist_shape
        warped_img = np.zeros((height, width, img.shape[2]), dtype=img.dtype)

        H_inv = np.linalg.inv(H)

        for i in range(height):
            for j in range(width):
                dist_pt = np.array([j, i, 1])
                src_pt = H_inv @ dist_pt
                src_pt /= src_pt[2]
                x, y = src_pt[:2]

        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            warped_img[i, j] = img[int(y), int(x)]

        return warped_img

    def compute_homography(self, calib_points, image, dartboard_center: Vector):
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
        
        homography = self.find_homography(image_points, dartboard_points)
        warped = self.warp_image(image, homography, (640, 480))
        
