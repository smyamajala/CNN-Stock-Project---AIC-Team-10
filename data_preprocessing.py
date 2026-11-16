from sample_candlesticks import DOWN
@DOWN
class DataConfig:
  img_size: tuple[int, int] = [64, 64]
  batch_size: int = 2
  train_split = float = 0.7
  val_split = float = 0.15
  test_split = float = 0.15

transform.rbg
set = datasets.ImageFolder(sample_candlesticks
  root = DATA_DIR, DOWN
  transform = rbg 

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
