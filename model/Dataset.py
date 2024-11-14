"""
This module is for the dataset class that is used in the project.
"""

from os import listdir, path
from typing import Dict

from PIL import Image
from torch.utils.data import Dataset

from torch import tensor
from torchvision import transforms

to_tensor = transforms.ToTensor()

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

    def __getitem__(self, idx) -> Dict[str, tensor]:
        img_path = path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        image_tensor = to_tensor(image)  # Convert the image to a tensor

        label_path = path.join(self.labels_dir, self.label_files[idx])
        annotations = []
        if path.exists(label_path):
            with open(label_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    annotations.append([class_id, x_center, y_center, width, height])

        return {
            'image': image_tensor,
            'annotations': tensor(annotations) 
        }

