import os
import shutil
import numpy as np
import glob
from PIL import Image
from sklearn.model_selection import train_test_split

from utils.image import slice_image, norm_to_uint8, niftii_to_images, labels_to_mask, resize

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
            only_masks = True,
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
        self.only_masks = only_masks
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

        to_slice = self.scan_shape != self.input_shape

        for scan_name in self.scans:
            full_path = os.path.join(self.niftii_dir, scan_name)
            X_data, y_data = self.prepare_images_labels(scan_name, full_path)

            # We are able to chose if scan image should be sliced
            if to_slice:
                sliced_images = slice_image(X_data, self.input_shape)
                sliced_labels = slice_image(y_data, self.input_shape)

                for i, labels in enumerate(sliced_labels):
                    self.save_dataset(sliced_images[i], labels, f'{scan_name}_{i:03d}')

            else:
                self.save_dataset(X_data, y_data, scan_name)

        self.split_dataset()

    """ Saves datasets by splitting files into images and labels
        As an option can ommit empty mask files
    """
    def save_dataset(self, X, y, scan_name):
        if y.max() > 0.0 or not self.only_masks:
            self.save_by_type(X, scan_name, 'images')
            self.save_by_type(y, scan_name, 'labels')

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
        Images and labels are resized to desired standard size
        Labels are also binarized based on labels
    """
    def prepare_images_labels(self, scan_name, path) -> (np.ndarray, np.ndarray):
        print(f'Loading from {path}/{self.niftii_labels}')
        label_data = niftii_to_images(self.niftii_labels, path)
        prepared_labels = labels_to_mask(label_data, self.labels, invert=self.invert)
        prepared_labels = resize(prepared_labels, self.scan_shape)

        print(f'Loading from {path}/{self.niftii_images}...')
        image_data = niftii_to_images(self.niftii_images, path)
        prepared_images = resize(image_data, self.scan_shape)

        return prepared_images, prepared_labels

    """ Splits files into separate datasets for training, validation and tests
        Files are grabbed from processed directory and moved into
        train / valid / test directory.

        Old structure is then removed.
    """
    def split_dataset(self):
        X_files = glob.glob(os.path.join(self.collection_dir, 'images', '*.???'))
        y_files = glob.glob(os.path.join(self.collection_dir, 'labels', '*.???'))

        X_train, X_valid, y_train, y_valid = train_test_split(X_files, y_files, test_size=0.2, random_state=1)
        X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size=0.25, random_state=1)

        files = {
            'images': { 'train': X_train, 'valid': X_valid, 'test': X_test },
            'labels': { 'train': y_train, 'valid': y_valid, 'test': y_test }
        }

        for dataset in ('train', 'valid', 'test'):
            for types in ('images', 'labels'):
                types_dir = os.path.join(self.collection_dir, dataset, types)
                os.makedirs(types_dir)
                for images in files[types][dataset]:
                    shutil.move(images, os.path.join(self.collection_dir, dataset, types))

        shutil.rmtree(os.path.join(self.collection_dir, 'images'))
        shutil.rmtree(os.path.join(self.collection_dir, 'labels'))
