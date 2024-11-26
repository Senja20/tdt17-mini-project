from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

from utils import get_device

model = YOLO(model='pole_detection\\train66\\weights\\best.pt')


# Initialize the SAHI AutoDetectionModel with the YOLO11 model
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",  # SAHI uses 'yolov8' as the model type for YOLOv8 and above
    model_path='pole_detection\\train66\\weights\\best.pt',
    confidence_threshold=0.3,
    device=get_device(),
)

img_path = 'C:\\Users\\Yauhen\\tdt17\data\\Poles\\test\\images\\combined_image_5_png.rf.9372598b5abf9cfff473ec530fdbf7be.jpg'

# Perform sliced inference on an image
result = get_sliced_prediction(
    img_path,
    detection_model,
    slice_height=800 - 32 * 2,
    slice_width=800 - 32 * 2,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

# Export and visualize the predicted bounding boxes
result.export_visuals(export_dir=".")
