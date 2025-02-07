import os
import shutil
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import torch
import numpy as np
import random
import os
import shutil
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict


def create_k_folds(data_dir, output_dir, k=5):
    """
    Split dataset into k folds at the patient level and save each fold's train and test data into separate folders.
    Ensures no patient overlap between splits.

    Args:
        data_dir (str): Path to the dataset directory with subfolders representing classes.
        output_dir (str): Path to save the k-fold data.
        k (int): Number of folds.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect patient IDs, their labels, and associated image paths
    patient_data = []  # Stores (patient_id, class_label, [image_paths])
    labels = []        # Class labels for stratification
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_dirs)}

    for class_name in class_dirs:
        class_path = os.path.join(data_dir, class_name)
        for patient_id in os.listdir(class_path):
            patient_path = os.path.join(class_path, patient_id)
            if os.path.isdir(patient_path):
                images = [
                    os.path.join(patient_path, img)
                    for img in os.listdir(patient_path)
                    if img.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
                if images:
                    patient_data.append((patient_id, class_to_idx[class_name], images))
                    labels.append(class_to_idx[class_name])  # Class label for stratification

    # Stratified split by patient
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(patient_data, labels), 1):
        # Create directories for this fold
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        train_dir = os.path.join(fold_dir, "train")
        test_dir = os.path.join(fold_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Copy train and test data
        for idx in train_idx:
            patient_id, class_label, images = patient_data[idx]
            class_name = list(class_to_idx.keys())[class_label]
            patient_folder = os.path.join(train_dir, class_name, patient_id)
            os.makedirs(patient_folder, exist_ok=True)
            for image_path in images:
                shutil.copy(image_path, os.path.join(patient_folder, os.path.basename(image_path)))

        for idx in test_idx:
            patient_id, class_label, images = patient_data[idx]
            class_name = list(class_to_idx.keys())[class_label]
            patient_folder = os.path.join(test_dir, class_name, patient_id)
            os.makedirs(patient_folder, exist_ok=True)
            for image_path in images:
                shutil.copy(image_path, os.path.join(patient_folder, os.path.basename(image_path)))

        print(f"Fold {fold_idx} created: Train and test data saved in {fold_dir}")


data_dir = "/home/dvegaa/DELFOS/delfos_final_dataset/delfos_images"  # Dataset directory with subfolders for each class
output_dir = "/home/dvegaa/DELFOS/delfos_final_dataset/delfos_images_4kfold"  # Where to save the folds
create_k_folds(data_dir, output_dir, k=4)


