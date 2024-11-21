from ultralytics import YOLO
import os

yaml_path = 'data.yaml'
assert os.path.exists(yaml_path), f'File not found {yaml_path}'
print(f'-----------------> Found {yaml_path}')

model = YOLO('yolo11n')
model.train(data=yaml_path, epochs=10)