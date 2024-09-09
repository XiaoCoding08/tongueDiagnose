from ultralytics import YOLO
from PIL import Image
from common import config

class Detector:
    def __init__(self):
        self.modelName = config.DETECT_MODEL
        self.model = YOLO(self.modelName)

    # 返回舌头坐标
    def detect(self,imgPath): 
        result = self.model(imgPath)
        xyxy = (result[0].boxes.xyxy).tolist()
        if not xyxy:
            return []
        box = xyxy[0]
        box = list(map(int, box))
        return box

