from ultralytics import YOLO

model = YOLO('models/yolov5su.pt')
model.train(data="tongue.yaml", epochs=15, imgsz=640)
# model.val(data="tongue.yaml")
