"""
This module is for the dataset class that is used in the project.
"""

from dataclasses import dataclass
from os import listdir, path
from typing import List

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class Annotation:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class DataSample:
    image: Image.Image
    annotations: List[Annotation]


class PoleDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = [
            f for f in listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.image_files.sort()
        self.label_files = [path.splitext(f)[0] + '.txt' for f in self.image_files]

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.image_files)

    def __getitem__(self, idx) -> DataSample:
        img_path = path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        label_path = path.join(self.labels_dir, self.label_files[idx])
        annotations = []
        if path.exists(label_path):
            with open(label_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    annotations.append(
                        Annotation(class_id, x_center, y_center, width, height)
                    )

        return DataSample(image, annotations)
