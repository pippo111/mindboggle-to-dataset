import numpy as np
import os
import nibabel as nib
from scipy.ndimage import zoom, rotate, shift, distance_transform_edt as distance

""" Extract scan image from niftii data file
"""
def niftii_to_images(filename: str, path: str = '.') -> np.ndarray:
    full_path = os.path.join(path, filename)
    img = nib.load(full_path)
    img_data = img.get_fdata(dtype=np.float32)

    return img_data

""" Divide background and structure by labels
    It is possible to invert mask, for example to select every label
    by selecting background and then invert.

    Returns binarized image as 0|255.
"""
def labels_to_mask(data: np.ndarray, labels: list, invert: bool = False) -> np.ndarray:
    mask = np.isin(data, labels, invert=invert).astype(np.uint8) * 255

    return mask

""" Resizes image to desired shape 2d/3d
"""
def resize(data: np.ndarray, new_shape: tuple) -> np.ndarray:
    old_shape = data.shape
    ratios = tuple(new_dim / old_dim for new_dim, old_dim in zip(new_shape, old_shape))

    resized_data = zoom(data, ratios, order=0)

    return resized_data

""" Slices cuboid into smaller pieces
    Can be undone by 'uncubify' fn
"""
def slice_image(inputs: np.ndarray, new_shape: tuple) -> np.ndarray:
    W, H, D = inputs.shape
    w, h, d = new_shape

    outputs = inputs.reshape(W//w, w, H//h, h, D//d, d)
    outputs = outputs.transpose(0,2,4,1,3,5)

    outputs = outputs.reshape(-1, w, h, d)

    return outputs

""" Normalizes input image to range 0-255
"""
def norm_to_uint8(data: np.ndarray) -> np.ndarray:
    max_value = data.max()
    if not max_value == 0:
        data = data / max_value

    data = 255 * data
    img = data.astype(np.uint8)
    return img
