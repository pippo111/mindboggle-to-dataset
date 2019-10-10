import config as cfg

from app.dataset import MyDataset

my_dataset = MyDataset(
  niftii_dir = cfg.dataset['niftii_dir'],
  niftii_labels = cfg.dataset['niftii_labels'],
  niftii_images = cfg.dataset['niftii_images'],
  dataset_dir = cfg.dataset['dataset_dir'],
  collection_name = cfg.dataset['collection_name'],
  scan_shape = cfg.dataset['scan_shape'],
  input_shape = cfg.dataset['input_shape'],
  labels = cfg.dataset['labels'],
  only_masks=cfg.dataset['only_masks'],
  invert=cfg.dataset['invert'],
  split=cfg.dataset['split'],
  scans = cfg.scans
)
my_dataset.create_dataset()
