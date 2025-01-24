import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import os 

def create_dataloaders(dataset_dir, dataset_class, transform_train, transform_test, batch_size):
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
    
    dataset_train = dataset_class(root=train_dir, transform=transform_train)

    # Calculate class weights for weighted sampling
    num_negatives = sum(1 for _, label in dataset_train if label == 0)
    num_positives = sum(1 for _, label in dataset_train if label == 1)
    class_sample_counts = [num_negatives, num_positives]
    class_weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)

    # Calculate sample weights
    sample_weights = [class_weights[label] for _, label in dataset_train]

    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Train DataLoader
    train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, shuffle=False)

    # Validation dataset and DataLoader
    dataset_val = dataset_class(root=val_dir, transform=transform_test)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    # Test dataset and DataLoader
    dataset_test = dataset_class(root=test_dir, transform=transform_test)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_negatives, num_positives
