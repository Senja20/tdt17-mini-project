import os
import cv2
from utils_functions.img_pre_processing import preprocess_image
from dotenv import load_dotenv

load_dotenv()

PATH_train = os.getenv('PATH_TO_TRAIN_DATA')
PATH_val = os.getenv('PATH_TO_VAL_DATA')

assert os.path.exists(PATH_train), 'PATH_TO_TRAIN_DATA is not valid'
assert os.path.exists(PATH_val), 'PATH_TO_VAL_DATA is not valid'

def preprocess_dataset(input_dir, output_dir):
    i = 0
    for filename in os.listdir(input_dir):
        if i % 10 == 0:
            print(f'---------> Processing image {i} of {len(os.listdir(input_dir))}')

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Preprocess the image
        preprocessed_img = preprocess_image(input_path)

        # Save the preprocessed image
        cv2.imwrite(output_path, preprocessed_img)
        i += 1

# Input and output directories
train_img_dir = PATH_train + '/images'
val_img_dir = PATH_val + '/images'
preprocessed_train_dir = '../Poles/train/images'
preprocessed_val_dir = '../Poles/test/images'

# Preprocess training and validation images
preprocess_dataset(train_img_dir, preprocessed_train_dir)
# preprocess_dataset(val_img_dir, preprocessed_val_dir)