from ultralytics import YOLO

model = YOLO('models/yolov5su.pt')
model.train(data="tongue.yaml", epochs=1, imgsz=640)
# model.val(data="tongue.yaml")
