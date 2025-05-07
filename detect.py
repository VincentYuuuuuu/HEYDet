import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('Weight Path') 
    model.predict(source='Source Path',
                  imgsz=640,
                  project='runs',
                  name='exp',
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                  # agnostic_nms=True,
                  # visualize=True,
                  # show_conf=False, 
                  # show_labels=False, 
                  # save_txt=True,
                )