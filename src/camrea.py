import cv2

class Camera:
    def __init__(self, cam_num):
        self.cam_num = cam_num
        self.cam = cv2.VideoCapture(self.cam_num)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.width = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Reruns to make sure camera is not blured
        for i in range(10):
            self.cam.read()
            
        _, self.start_frame = self.cam.read()
        _, self.cur_frame = self.cam.read()

    def get_cur_frame(self):
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        _, self.cur_frame = self.cam.read()
        return self.cur_frame