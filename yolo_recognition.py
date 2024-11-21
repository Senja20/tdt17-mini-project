from ultralytics import YOLO
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

path_data_set_train = '/cluster/home/tristanw/Poles/train'

class CustomImageDataset(Dataset):
    def __init__(self, data_set_path):
        self.data_set_path = data_set_path
        self.images_list = os.read_dir(data_set_path + '/images')
        self.labels_list = os.read_dir(data_set_path + '/labels')

        print('Found {} images and {} labels'.format(len(self.images_list), len(self.labels_list)))

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        labels_name = self.labels_list[idx]

        image = read_image(self.images_list[idx])
        image 

        return read_image(self.images_list[idx]), self.data_set_path + '/labels/' + labels_name
    

# dataloader = CustomImageDataset()

yaml_path = 'data.yaml'
assert os.path.exists(yaml_path), f'File not found {yaml_path}'
print(f'-----------------> Found {yaml_path}')

model = YOLO('yolo11s')


model.train(data=yaml_path, epochs=55)