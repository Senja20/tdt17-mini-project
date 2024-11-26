"""
get device for model training and classification
cuda or  cpi
"""

from torch import device
from torch.cuda import is_available


def get_device() -> device:
    """
    desc: Function used to get the device.
    :return: The device.
    """
    return device("cuda" if is_available() else "cpu")
