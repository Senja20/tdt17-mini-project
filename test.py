from os import environ

import cv2
from ultralytics import YOLO

# Set the environment variable for the model
environ["CUDA_LAUNCH_BLOCKING"] = "1"
environ["TORCH_USE_CUDA_DSA"] = "1"


# Load the model
model = YOLO(model='pole_detection\\train66\\weights\\best.pt')

# Load the image
img_path = 'C:\\Users\\Yauhen\\tdt17\data\\Poles\\test\\images\\combined_image_5_png.rf.9372598b5abf9cfff473ec530fdbf7be.jpg'
img = cv2.imread(img_path)

# Run inference
results = model(img, device='cpu')

# Plot the results
annotated_img = results[0].plot()

# Display the annotated image
cv2.imshow('Detections', annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
