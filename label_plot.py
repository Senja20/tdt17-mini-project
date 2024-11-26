import os

import matplotlib.pyplot as plt
import numpy as np


# Function to read YOLO labels and extract attributes
def read_yolo_labels(directory):
    x_centers = []
    y_centers = []
    widths = []
    heights = []

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):  # Ensure it's a YOLO label file
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # Ensure valid YOLO label format
                        x_center = float(parts[1])  # Normalized x_center
                        y_center = float(parts[2])  # Normalized y_center
                        width = float(parts[3])  # Normalized width
                        height = float(parts[4])  # Normalized height
                        x_centers.append(x_center)
                        y_centers.append(y_center)
                        widths.append(width)
                        heights.append(height)

    return np.array(x_centers), np.array(y_centers), np.array(widths), np.array(heights)


# Directory containing YOLO label files
label_directory = "C:\\Users\\Yauhen\\tdt17\\data\\Poles\\train\\labels"

# Read the YOLO label files
x_centers, y_centers, widths, heights = read_yolo_labels(label_directory)

# Prepare data for box plot
data = [x_centers, y_centers, widths, heights]
labels = ['X Center', 'Y Center', 'Width', 'Height']

# Create the box plot with colors
plt.figure(figsize=(10, 8))
box = plt.boxplot(data, vert=True, patch_artist=True, labels=labels)

# Add colors to the boxes
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFD966']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Customize plot appearance
plt.title('Distribution of Bounding Box Attributes')
plt.ylabel('Label Values')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
