# Image-to-image translation to remove skin colour bias

## Prerequisites
Code is intended to work with **Python 3.6.x** or later.

### [Pytorch 1.13.0](https://pytorch.org/blog/PyTorch-1.13-release/) and [torchvision 0.14.0](https://pypi.org/project/torchvision/)
Follow the instructions in [pytorch.org](https://pytorch.org/) for your current setup.

## Training $ Testing
### 1. Setup the dataset
First, you need to build the dataset by setting up the following directory structure:
```
.
├── datasets                   
|   ├── <dataset_name>         # e.g. TLA4, SWET
|   |   ├── white              # contains domain A images (i.e. white)
|   |   |   ├── train          # Training
|   |   |   └── test           # Testing
|   |   └── non-white          # contains domain B images (i.e. non-white)
|   |   |   ├── train          # Training
|   |   |   └── test           # Testing
```

### 2. Training
