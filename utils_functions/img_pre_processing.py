import cv2 as cv

# Define a preprocessing function
def preprocess_image(img_path):
    img = cv.imread(img_path)
    # Resize, normalize, or apply filters
    img = cv.GaussianBlur(img, (3, 3), 0)

    alpha = 1.8  # Contrast control (1.0-3.0)
    beta = 20     # Brightness control (0-100)
    img = cv.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img
