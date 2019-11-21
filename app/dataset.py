import os
import shutil
import numpy as np
import glob
from PIL import Image
from sklearn.model_selection import train_test_split

from inn_pipeline.dataset import NiftiDataset
from inn_pipeline.utils.volume_ops import resize_volume
from utils.image import slice_image, norm_to_uint8, niftii_to_images, labels_to_mask, resize, swap_axes

class MyDataset():
    def __init__(
            self,
            niftii_dir = '~/datasets/niftii',
            niftii_labels = 'aseg-in-t1weighted_2std.nii.gz',
            niftii_images = 't1weighted_2std.nii.gz',
            dataset_dir = 't1weighted_2std.nii.gz',
            collection_name = 'mindboggle',
            scan_shape = (192, 256, 256),
            input_shape = (48, 64, 64),
            labels = [255.0],
            invert = False,
            scans = [],
        ):
        self.niftii_dir = niftii_dir
        self.niftii_labels = niftii_labels
        self.niftii_images = niftii_images
        self.dataset_dir = dataset_dir
        self.collection_name = collection_name
        self.scan_shape = scan_shape
        self.input_shape = input_shape
        self.labels = labels
        self.invert = invert
        self.scans = scans

        self.collection_dir = os.path.join(dataset_dir, collection_name)

        self.dims = '2d' if 1 in input_shape else '3d'

    """ Public API
        Creates splitted dataset for 2d/3d models
        Ouputs are saved numpy arrays if 3d or png image files if 2d
        2d / 3d is obtained by desired input size
        e.g. (48, 64, 64) will be 3d npy files
        (1, 256, 256) will be 2d png files by axis=0
    """
    def create_dataset(self):
        if os.path.exists(self.collection_dir):
            shutil.rmtree(self.collection_dir)
        os.makedirs(self.collection_dir)

        for group in ['train', 'valid', 'test']:
            for scan_name in self.scans[group]:
                full_path = os.path.join(self.niftii_dir, scan_name)
                X_data, y_data = self.prepare_images_labels(full_path)

                zero_mask = 0

                for i, (X, y) in enumerate(zip(X_data, y_data)):
                    if y.max() > 0.0 or group == 'test':
                        self.save_by_type(X, f'{scan_name}_{i:03d}', f'{group}/images')
                        self.save_by_type(y, f'{scan_name}_{i:03d}', f'{group}/labels')
                    else:
                        if zero_mask % 10 == 0:
                            self.save_by_type(X, f'{scan_name}_{i:03d}', f'{group}/images')
                            self.save_by_type(y, f'{scan_name}_{i:03d}', f'{group}/labels')
                        zero_mask += 1


    """ Saves image / labels files
    """
    def save_by_type(self, data, name, types):
        data_full_path = os.path.join(self.collection_dir, types)
        data_full_name = os.path.join(data_full_path, name)

        if not os.path.exists(data_full_path):
            os.makedirs(data_full_path)

        norm_data = norm_to_uint8(data)

        if self.dims == '2d':
            print(f'Saving {types} as {data_full_name}.png')
            im = Image.fromarray(norm_data.squeeze())
            im.save(f'{data_full_name}.png')
        else:
            print(f'Saving {types} as {data_full_name}.npy')
            np.save(data_full_name, norm_data)

        print('Done.')

    """ Returns images and labels with requested format
        Images and labels are resized to desired standard size with shape
        adjusted to requested size
        Labels are also binarized based on labels
    """
    def prepare_images_labels(self, path) -> (np.ndarray, np.ndarray):
        print(f'Loading from {path}/{self.niftii_labels}')
        prepared_labels = NiftiDataset(os.path.join(path, self.niftii_labels), grayscale_conversion = False).coronal_view
        prepared_labels = labels_to_mask(prepared_labels, self.labels, invert=self.invert)
        prepared_labels = resize_volume(prepared_labels, self.scan_shape, bspline_order=0)

        print(f'Loading from {path}/{self.niftii_images}...')
        prepared_images = NiftiDataset(os.path.join(path, self.niftii_images), grayscale_conversion = True).coronal_view
        prepared_images = resize_volume(prepared_images, self.scan_shape)

        return prepared_images, prepared_labels
