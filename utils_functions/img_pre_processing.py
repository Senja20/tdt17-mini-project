import cv2 as cv
import numpy as np

# Define a preprocessing function
def preprocess_image(img_path):
    img = cv.imread(img_path)

    # alpha = 1.8  # Contrast control (1.0-3.0)
    # beta = 20     # Brightness control (0-100)
    # img = cv.convertScaleAbs(img, alpha=alpha, beta=beta)

    # img = cv.GaussianBlur(img, (3, 3), 0)
    height, width, _ = img.shape

    # First iteration of drawing the mask
    ignore_bottom = 30
    corner_width = int(width // 8)
    left_mask_end = int(width // 4)
    right_mask_start = int(3 * width // 4)

    img[0 : height - ignore_bottom, corner_width:left_mask_end] = 0
    img[0 : height - ignore_bottom, right_mask_start:width - corner_width] = 0

    # Second iteration of drawing the mask
    ignore_bottom = 50
    corner_width = int(width // 16)
    left_mask_end = int(width // 3)
    right_mask_start = int(2 * width // 3)

    # Apply the second mask on the original image
    img[0 : height - ignore_bottom, corner_width:left_mask_end] = 0
    img[0 : height - ignore_bottom, right_mask_start:width - corner_width] = 0
    
    mask = np.zeros_like(img)
    mask[int(height // 2.5) :, :] = img[int(height // 2.5) :, :]
    img = mask

    return img
