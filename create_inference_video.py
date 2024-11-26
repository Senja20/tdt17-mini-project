import cv2 as cv
import os
from glob import glob
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
import re
import numpy as np
from dotenv import load_dotenv

load_dotenv()

MODEL_VERSION = "train_tuned2"
USE_SAHI = True
FRAME_RATE = 4
CREATE_GROUND_TRUTH_VIDEO = False

PATH_train = os.getenv('PATH_TO_TRAIN_DATA')
PATH_val = os.getenv('PATH_TO_VAL_DATA')
assert os.path.exists(PATH_train), 'PATH_TO_TRAIN_DATA is not valid'
assert os.path.exists(PATH_val), 'PATH_TO_VAL_DATA is not valid'

def load_model():
    # Load the model
    if USE_SAHI:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",  # SAHI uses 'yolov8' as the model type for YOLOv8 and above
            model_path=f'runs/detect/{MODEL_VERSION}/weights/best.pt',
            confidence_threshold=0.3,
            device='cuda',
        )
        print('---------> Model loaded')
        return detection_model
    else:
        model = YOLO(f'runs/detect/{MODEL_VERSION}/weights/best.pt')
        print('---------> Model loaded')
        return model

def infere_and_draw(frame, model):

    results = model(frame)
    predictions = results[0].boxes
    overlay = frame.copy()

    for box in predictions:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0].item()
        class_id = int(box.cls[0])
        label = f"{model.names[class_id]} ({confidence:.2f})"

        roi = overlay[y1:y2, x1:x2]  # Region of interest
        brighter_roi = cv.convertScaleAbs(roi, alpha=1.5, beta=50)  # Adjust brightness
        overlay[y1:y2, x1:x2] = brighter_roi  # Apply the brighter region back

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def infere_and_draw_sahi(frame, model):

    results = get_sliced_prediction(
        frame,
        model,
        slice_height=800 - 32 * 2,
        slice_width=800 - 32 * 2,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )
    predictions = results.object_prediction_list
    overlay = frame.copy()

    for box in predictions:
        bbox = box.bbox
        score = box.score.value
        name = box.category.name

        x1 = int(bbox.minx)
        y1 = int(bbox.miny)
        x2 = int(bbox.maxx)
        y2 = int(bbox.maxy)
        label = f"{name} ({score:.2f})"

        roi = overlay[y1:y2, x1:x2]  # Region of interest
        brighter_roi = cv.convertScaleAbs(roi, alpha=1.5, beta=50)
        overlay[y1:y2, x1:x2] = brighter_roi

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def extract_base_and_number(filename):
    # Extract the base name (e.g., "combined_image") and numeric part
    match = re.match(r"(.*?)(\d+)", filename.split("/")[-1])  # Adjust for full path
    base = match.group(1) if match else ""
    number = int(match.group(2)) if match else float('inf')
    return (base, number)

def extract_bounding_box_from_data_set(img_path):

    img_width = 1024
    img_height = 128

    label_path = img_path.replace('.jpg', '.txt')
    label_path = label_path.replace('images', 'labels')

    with open(label_path, 'r') as file:
        contents = file.read()
    contents = contents.strip().split('\n')

    # Extract bounding box coordinates without the class label
    bounding_boxes = []
    for line in contents:
        parts = line.split()  # Split the line into parts
        coords = list(map(float, parts[1:]))  # Skip the first element (class label) and convert the rest to float
        
        # Convert YOLO percentages to real coordinates
        x_center = coords[0] * img_width
        y_center = coords[1] * img_height
        width = coords[2] * img_width
        height = coords[3] * img_height

        # Calculate real bounding box coordinates
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # Append as [x_min, y_min, x_max, y_max]
        bounding_boxes.append([x_min, y_min, x_max, y_max])
    bounding_boxes_array = np.array(bounding_boxes).astype(int)

    return bounding_boxes_array

def main():

    print('---------> Starting video creation, using of SAHI:', USE_SAHI)

    if USE_SAHI:
        output_path = f'output_imgs/output_{MODEL_VERSION}_sahi'
    else:    
        output_path = f'output_imgs/output_{MODEL_VERSION}_simple'

    frame_files = sorted(glob(f'{PATH_val}/images/*.jpg'), key=extract_base_and_number)

    model = load_model()

    frame = cv.imread(frame_files[0])
    hight, width, _ = frame.shape

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video_writer = cv.VideoWriter(f'{output_path}.mp4', fourcc, FRAME_RATE, (width, hight))

    i = 0
    for frame_file in frame_files:
        if i % 10 == 0:
            print(f'Processing frame {i} of {len(frame_files)}')
        frame = cv.imread(frame_file)
        if USE_SAHI:
            frame_pred = infere_and_draw_sahi(frame, model)
        else:
            frame_pred = infere_and_draw(frame, model)
        video_writer.write(frame_pred)
        i += 1

    video_writer.release()
    print('---------> Created video at:', output_path)

    if CREATE_GROUND_TRUTH_VIDEO:
        print('----------> Continueing because of CREATE_GROUND_TRUTH_VIDEO:', CREATE_GROUND_TRUTH_VIDEO)

        video_writer = cv.VideoWriter(f'output_imgs/ground_truth.mp4', fourcc, FRAME_RATE, (width, hight))
        i = 0
        for frame_file in frame_files:
            if i % 10 == 0:
                print(f'Processing frame {i} of {len(frame_files)}')
            frame = cv.imread(frame_file)       
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            bounding_boxes = extract_bounding_box_from_data_set(frame_file)
            if len(bounding_boxes) > 0:
                for box in bounding_boxes:
                    frame = cv.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

            video_writer.write(frame)
            i += 1

        video_writer.release()
        print('---------> Created video with ground truth bounding boxes')
    

if __name__ == '__main__':
    main()