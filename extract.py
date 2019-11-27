import config as cfg

from app.dataset import MyDataset

my_dataset = MyDataset(
    niftii_dir=cfg.dataset['niftii_dir'],
    niftii_labels=cfg.dataset['niftii_labels'],
    niftii_images=cfg.dataset['niftii_images'],
    dataset_dir=cfg.dataset['dataset_dir'],
    collection_name=cfg.dataset['collection_name'],
    scan_shape=cfg.dataset['scan_shape'],
    view=cfg.dataset['view'],
    labels=cfg.dataset['labels'],
    invert=cfg.dataset['invert'],
    scans=cfg.scans
)
my_dataset.create_dataset()
