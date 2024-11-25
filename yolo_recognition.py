from ultralytics import YOLO
import os
import yaml

yaml_path = 'data.yaml'
# yaml_path = 'data_processed.yaml'
assert os.path.exists(yaml_path), f'File not found {yaml_path}'
print(f'-----------------> Found {yaml_path}')

with open('best_hyperparameters.yaml', 'r') as f:
    hyperparameters = yaml.safe_load(f)

model = YOLO('yolo11n')

model.train(
    data=yaml_path,
    epochs=110, 
    batch=4,
    imgsz=800, 
    workers=8, 
    device='cuda', 
    dropout=0.1, 
    plots=True, 
    augment=True,
    lr0=0.00758,
    lrf=0.00955,
    momentum=0.78372,
    weight_decay=0.00059,
    warmup_epochs=3.4495,
    warmup_momentum=0.95,
    box=7.9401,
    cls=0.63256,
    dfl=1.60972,
    hsv_h=0.08584,
    hsv_s=0.67105,
    hsv_v=0.40281,
    degrees=9.62981,
    translate=0.09428,
    scale=0.32995,
    shear=0.17692,
    perspective=0.00045,
    flipud=0.0,
    fliplr=0.48023,
    bgr=0.0,
    mosaic=0.70733,
    mixup=0.0,
    copy_paste=0.0,)

'''
model.tune(
    data=yaml_path,
    epochs=20,
    iterations=25,
    optimizer='adam', 
    batch=4,
    imgsz=800, 
    workers=8, 
    device='cuda', 
    dropout=0.1, 
    plots=True, 
    augment=True,
    hsv_h=0.1,
    hsv_v=0.5,
    degrees=10.0,
    translate=0.1,
    shear=0.2,
    perspective=0.0005,
    mosaic=1.0,)'''


'''# ----------> raw2
model.train(
    data=yaml_path,
    epochs=100, 
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
    mosaic=0.5,)'''
