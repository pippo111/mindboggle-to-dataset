#
# Example config file for creating white matter hyperintensities mask
# from wmh.isi.uu.nl datasets
#
# Crated datasets will be saved as images collection with image / mask
# 

# Dataset common setup
dataset = {
    'niftii_dir': '/home/filip/Projekty/ML/datasets/niftii/wmh/GE3T',
    'niftii_labels': 'wmh.nii.gz',
    'niftii_images': 'orig/3DT1.nii.gz',
    'dataset_dir': '/home/filip/Projekty/ML/datasets/processed',
    'collection_name': 'GE3T_20_256x256x192_wmh',
    'scan_shape': (256, 256, 192),
    'input_shape': (256, 1, 192),
    'labels': [1.0],
    'only_masks': True,
    'invert': False
}

scans = [
    # These we will use for train and validation:
    '100',
    '101',
    '102',
    '103',
    '104',
    '105',
    '106',
    '107',
    '108',
    '109',
    '110',
    '112',
    '113',
    '114',
    '115',
    '116',
    '126',
    '132',
    '137',
    # This we will use only for test:
    # '144'
  ]
  