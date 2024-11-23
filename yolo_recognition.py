from ultralytics import YOLO
import os

yaml_path = 'data.yaml'
# yaml_path = 'data_processed.yaml'
assert os.path.exists(yaml_path), f'File not found {yaml_path}'
print(f'-----------------> Found {yaml_path}')


model = YOLO('yolo11n')
model.train(
    data=yaml_path,
    epochs=55, 
    batch=4,
    imgsz=800, 
    workers=8, 
    device='cuda', 
    dropout=0.1, 
    plots=True, 
    augment=True,
    hsv_h=0.1,
    hsv_v=0.5,
    degree=(-20, 20),
    translate=0.1,
    shear=0.2,
    perspective=0.0005,
    mosaic=1.0,)

'''
----------> raw2
model.train(
    data=yaml_path,
    epochs=55, 
    batch=4,
    imgsz=800, 
    workers=8, 
    device='cuda', 
    dropout=0.1, 
    plots=True, 
    augment=True,
    hsv_h=0.1,
    hsv_v=0.5,
    shear=0.2,
    perspective=0.0005,
    mosaic=0.5,)
'''