# Image-to-image translation to remove skin colour bias

## Prerequisites
Code is intended to work with **Python 3.6.x** or later.

### [Pytorch 1.13.0](https://pytorch.org/blog/PyTorch-1.13-release/) and [torchvision 0.14.0](https://pypi.org/project/torchvision/)
Follow the instructions in [pytorch.org](https://pytorch.org/) for your current setup.

## Training & Testing
### 1. Setup the dataset
First, you need to build the dataset by setting up the following directory structure:
```
.
├── datasets                   
|   ├── <dataset_name>         # e.g. TLA4, SWET
|   |   ├── white              # contains domain A images (i.e. white)
|   |   |   ├── train          # Training
|   |   |   └── test           # Testing
|   |   └── non_white          # contains domain B images (i.e. non-white)
|   |   |   ├── train          # Training
|   |   |   └── test           # Testing
```

### 2. Training
```
python train.py --model ag_cut --normG batch --normD batch --init_type normal --name <save_name>
```
This command will start a training session using the images under the _dataroot/white/train_ and _dataroot/non_white/train_ with the given hyperparameters. 
**normG, normD** and **init_type** must be set as specified for training AGCUT. You can change other hyperparameters; see ```./train --help``` for a description.

The generator, patch sampler and discriminator weights will be saved under the output directory.

### 3. Testing
```
python test.py --model ag_cut --normG batch --normD batch --init_type normal --load_path <path of generator weights> --name <save_name>
```
This command will take the images under the _dataroot/white/test and _dataroot/non_white/test_, run them through the selected generator and save the output 
under the output directory.

## Code Structure
- ```train.py```, ```test.py```: the entry point for training and testing.
- ```models```: defines the architectures of all models
- ```options```: defines the options can be used for training
- ```utils```: defines all utility functions used. For example, ```create_dataset.py```.

## Acknowledgements
This source code borrows heavily from [CycleGAN](https://github.com/junyanz/CycleGAN.git), [CUT](https://github.com/taesungp/contrastive-unpaired-translation.git) and [AGGAN](https://github.com/sarathknv/AGGAN.git). 
We acknowledge the Imperial College London for supporting this research.
