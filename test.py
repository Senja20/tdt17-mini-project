from os import environ

import cv2
from ultralytics import YOLO

# Set the environment variable for the model
environ["CUDA_LAUNCH_BLOCKING"] = "1"
environ["TORCH_USE_CUDA_DSA"] = "1"


# Load the model
model = YOLO('yolov9c_trained.pt')

# Load the image
img_path = 'C:\\Users\\Yauhen\\tdt17\\img.jpg'
img = cv2.imread(img_path)

# Run inference
results = model(img, device='cpu')

# Plot the results
annotated_img = results[0].plot()

# Display the annotated image
cv2.imshow('Detections', annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Save the annotated image
save_path = 'C:\\Users\\Yauhen\\vortex-image-processing\\runs\\detect\\predict\\output_with_detections.jpg'
cv2.imwrite(save_path, annotated_img)
