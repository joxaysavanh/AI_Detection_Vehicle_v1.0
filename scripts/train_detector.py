
from ultralytics import YOLO

model = YOLO("../weights/yolo12x.pt")

train_results = model.train(
    data="datasets/data.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    workers=0,
)

metrics = model.val()
