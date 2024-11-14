from typing import Dict, List
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch import stack

def collate_fn(batch: List[Dict[str, Tensor]]):
    images = [item['image'] for item in batch]
    annotations = [item['annotations'] for item in batch]

    # Stack images into a batch
    images = stack(images, dim=0)

    # Pad annotations to match the maximum number of annotations in the batch
    annotations_padded = pad_sequence(
        annotations, batch_first=True, padding_value=-1
    )

    mask = (annotations_padded[:, :, 0] != -1)  # Assume class_id is the first element

    return {
        'image': images,
        'annotations': annotations_padded,
        'mask': mask 
    }