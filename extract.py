import json

from app.dataset import MyDataset

with open('config.json', 'r') as cfg_file:
    config = json.load(cfg_file)

dataset = config['dataset']
scans = config['scans']

my_dataset = MyDataset(
    collection_source=dataset['collection_source'],
    niftii_dir=dataset['niftii_dir'],
    niftii_labels=dataset['niftii_labels'],
    niftii_images=dataset['niftii_images'],
    dataset_dir=dataset['dataset_dir'],
    collection_name=dataset['collection_name'],
    scan_shape=tuple(dataset['scan_shape']),
    view=dataset['view'],
    labels=dataset['labels'],
    invert=dataset['invert'],
    scans=scans
)

my_dataset.create_dataset()
my_dataset.create_summary()
