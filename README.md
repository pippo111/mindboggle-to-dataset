# Mindboggle to dataset
Tool for converting mindboggle niftii files into ready to use 2d/3d datasets.

## Installation

For your convinience you can recreate environment by running:
```bash
conda env create -f environment.yml
conda activate datasets
```

## Usage

1. Put somewhere on your mindboggle niftii collection files.
2. Create `config.py` with setup.
3. Run `python extract.py`

## Config file

Two examplary files are available: `config.brain.py` and `config.lateral_ventricle.py` 

Detailed explanation on `config.brain.py` example:

```python
# Dataset common setup
dataset = {
    # Input directory with niftii collections from mindboggle
    'niftii_dir': '/home/filip/Projekty/ML/datasets/niftii',
    
    # Filename for labeled niftii file
    'niftii_labels': 'aseg-in-t1weighted_2std.nii.gz',
    
    # Filename for niftii file with scan image
    'niftii_images': 't1weighted_2std.nii.gz',
    
    # Output root directory for processed files
    'dataset_dir': '/home/filip/Projekty/ML/datasets/processed',
    
    # Name of processed dataset. It will be saved under root directory.
    # Warning: if the directory already exists it will be wiped out
    'collection_name': 'mindboggle_84_Nx192x256_brain',
    
    # Input scan size. If the input will be different then it will be resized
    'scan_shape': (192, 256, 256),
    
    # Output dataset item size.
    # For 2d datasets set one of the axis to 1
    # Otherwise dataset will be saved as 3d numpy cuboids.
    # In the example below we are creating 2d dataset on axis=2 saved as png slices
    'input_shape': (192, 256, 1),
    
    # Labels that will be converted to binary mask
    'labels': [0.0],
    
    # If set to true, ommit empty masks i.e. masks with only background
    'only_masks': True,
    
    # If set to true, invert selected labels to set them as background
    # In this example we are taking background label (0.0) and then inverting it.
    # This way we are selecting every label exept of the background.
    'invert': True
}

scans = [
    # Directory names with some of mindboggle collection files
    'Afterthought-1',
    'MMRR-3T7T-2-1',
    'MMRR-3T7T-2-2',
    'MMRR-21-1',
    'MMRR-21-2'
  ]
  
```
