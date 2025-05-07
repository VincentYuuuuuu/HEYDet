import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('Wright Path') # 
    model.val(data='Data Path',
              split='val', 
              imgsz=640,
              batch=16,
              # iou=0.7,
              # rect=False,
              # save_json=True,
              project='runs/val',
              name='exp',
              )