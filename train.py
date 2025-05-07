from ultralytics import YOLO

# Load a model
model = YOLO("Model Path")

# Train the model
train_results = model.train(
    data="Data Path",
    epochs=300,  
    imgsz=640,  
    device="",  
    batch=32,
)

metrics = model.val()
