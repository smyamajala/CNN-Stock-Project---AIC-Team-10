
#def(
#transform.sensor

import os # standard library
from datetime import datetime # from this section import this feature

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

DATA_DIR = "sample_candlesticks" # directory of generated images 

#1 Configuration: Sets image size, batch size, and split percentages

#2 Transform images into format our CNN expects
transform = transform.

#3 Loading images from UP/DOWN folders
set = datasets.ImageFolder( 
  root = DATA_DIR,
  transform = transform
)

#4 Sort the images chronogically by their start date 
#5 Calculate split boundaries 
#6 Create Dataloaders

#Print the dataset information --> return all the deliverables from the file 
