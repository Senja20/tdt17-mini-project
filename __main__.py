
print('Hello, World!')
from model import PoleDataset
from dotenv import load_dotenv
from os import getenv
from torch.utils.data import DataLoader
from utils import collate_fn, get_device

load_dotenv()

if __name__ == '__main__':
    device = get_device()

    path_to_train_images: str = getenv('PATH_TO_TRAIN_DATA')
    path_to_val_images: str = getenv('PATH_TO_VAL_DATA')

    train_dataset = PoleDataset(
        path_to_train_images + '/images',
        path_to_train_images + '/labels',
    )
    val_dataset = PoleDataset(
        path_to_val_images + '/images',
        path_to_val_images + '/labels',
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)