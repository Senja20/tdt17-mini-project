import cv2 as cv
import os
from glob import glob
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
from dotenv import load_dotenv

load_dotenv()

MODEL_VERSION = "train_raw2"

USE_SAHI = False

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


def main():

    print('---------> Starting video creation, using of SAHI:', USE_SAHI)

    if USE_SAHI:
        output_path = f'output_imgs/output_{MODEL_VERSION}_sahi'
    else:    
        output_path = f'output_imgs/output_{MODEL_VERSION}_simple'
    frame_rate = 3

    frame_files = sorted(glob(f'{PATH_val}/images/*.jpg'))

    model = load_model()

    frame = cv.imread(frame_files[0])
    hight, width, _ = frame.shape

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video_writer = cv.VideoWriter(f'{output_path}.mp4', fourcc, frame_rate, (width, hight))

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

if __name__ == '__main__':
    main()