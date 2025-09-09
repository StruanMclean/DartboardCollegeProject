from dataclasses import dataclass

@dataclass
class DartboardScorer:
    dartboard_numbers = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
    
    double_bull_radius = 90 
    bull_radius = 159
    treble_inner_radius = 990
    treble_outer_radius = 1070
    double_inner_radius = 1620
    double_outer_radius = 1700 

    segment_angle = 18

    calibration_angles = [
        351,  # Between 5 and 20 (20 is at 0°, 5 is at 342°, intersection at 351°)
        171,  # Between 13 and 6 (13 is at 162°, 6 is at 180°, intersection at 171°)
        45,   # Between 17 and 3 (3 is at 36°, 17 is at 54°, intersection at 45°)
        189   # Between 8 and 11 (11 is at 198°, 8 is at 180°, intersection at 189°)
    ]