import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import os 
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

def create_dataloaders(dataset_dir, json_root, dataset_class, transform_train, transform_test, batch_size, fold, args, multimodal=False):
    """
    Creates dataloaders for training, validation, and testing with weighted sampling for class imbalance.

    Args:
        train_dir (str): Path to training dataset directory.
        val_dir (str): Path to validation dataset directory.
        test_dir (str): Path to testing dataset directory.
        dataset_class (Dataset): Custom dataset class.
        transform_train (callable): Transformations for training dataset.
        transform_test (callable): Transformations for validation and testing datasets.
        batch_size (int): Batch size for the dataloaders.

    Returns:
        tuple: train_loader, val_loader, test_loader
    """
    # Training dataset
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    test_dir = os.path.join(dataset_dir, "test")
    
    if multimodal:
        dataset_train = dataset_class(root=train_dir, json_root=json_root, transform=transform_train)
    else:
        dataset_train = dataset_class(root=train_dir,  transform=transform_train)
        
    # Calculate class weights for weighted sampling
    if args.sampling:
        print("Implementing sampling")
        num_negatives = sum(1 for _, label in dataset_train if label == 0) #2789 
        num_positives = sum(1 for _, label in dataset_train if label == 1) #470 
        
        class_sample_counts = [num_negatives, num_positives]
        class_weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)

        # Calculate sample weights
        sample_weights = [class_weights[label] for _, label in dataset_train]

        # Create WeightedRandomSampler
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        # Train DataLoader
        train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=8, pin_memory=True)
    else:
        print("No sampling implemented")
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        num_negatives = 1
        num_positives = 1

    # Validation dataset and DataLoader
    if os.path.exists(val_dir):
        if multimodal:
            dataset_val = dataset_class(root=val_dir, json_root=json_root, transform=transform_train)
        else:
            dataset_val = dataset_class(root=val_dir, transform=transform_test)
            
        val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    else: 
        val_loader = None

    # Test dataset and DataLoader
    if multimodal:
        dataset_test = dataset_class(root=test_dir, json_root=json_root, transform=transform_train)
    else:
        dataset_test = dataset_class(root=test_dir, transform=transform_test)
        
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader, test_loader, num_negatives, num_positives
