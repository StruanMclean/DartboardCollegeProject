from typing import List
#from ast import List
import math
from helpers.vect import Vector
import time

class DartTracker:
    def __init__(self, distance_threshold=50, time_threshold=2.0):
        self.tracked_darts = []
        self.distance_threshold = distance_threshold
        self.time_threshold = time_threshold

    def mark_as_scored(self, dart: Vector):
        for i, (tracked_dart, timestamp, scored) in enumerate(self.tracked_darts):
            distance = math.sqrt(
                (dart.x - tracked_dart.x)**2 + 
                (dart.y - tracked_dart.y)**2
            )
            if distance <= self.distance_threshold:
                self.tracked_darts[i] = (tracked_dart, timestamp, True)
                break

    def get_unscored_darts(self) -> List[Vector]:
        return [
            dart for dart, _, scored in self.tracked_darts 
            if not scored
        ]

    def update(self, new_darts: List[Vector]) -> List[Vector]:
        current_time = time.time()
        
        # For filtering out darts since
        # there can be multiple boxes
        new_unique_darts = []

        self.tracked_darts = [
            (dart, timestamp, scored) 
            for dart, timestamp, scored in self.tracked_darts 
            if current_time - timestamp <= self.time_threshold
        ]

        for new_dart in new_darts:
            is_new = True
            
            for tracked_dart, _, _ in self.tracked_darts:
                distance = math.sqrt(
                    (new_dart.x - tracked_dart.x)**2 + 
                    (new_dart.y - tracked_dart.y)**2
                )
                
                if distance <= self.distance_threshold:
                    is_new = False
                    break
            
            if is_new:
                self.tracked_darts.append((new_dart, current_time, False))
                new_unique_darts.append(new_dart)
        
        return new_unique_darts
