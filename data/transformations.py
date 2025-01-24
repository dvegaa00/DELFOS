from torchvision.transforms.transforms import Resize
from torchvision.transforms import transforms
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import random
import numpy as np
import torch

# Define data transformation


transform_train = transforms.Compose([
    transforms.Lambda(lambda x: x[:, 65:, :]), 
    transforms.ToPILImage(), 
    transforms.Resize((224, 224), antialias=True), 
    torchvision.transforms.AugMix(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

"""
transform_train = transforms.Compose([
    transforms.Lambda(lambda x: x[:, 65:, :]),  # Crop top 65 pixels
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5]),
])
"""
transform_test = transforms.Compose([
    transforms.Lambda(lambda x: x[:, 65:, :]),
    transforms.ToPILImage(),
    transforms.Resize((224, 224), antialias=True), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

