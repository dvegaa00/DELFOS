# Count patients and images in train/test for each fold
import os
from collections import Counter


def validate_k_folds(data_dir):
    """
    Validates the k-fold dataset structure and checks for:
        - Total image and patient counts in each split.
        - No patient overlap between train and test splits.
        - No image overlap between train and test splits.

    Args:
        data_dir (str): Path to the dataset directory with fold subfolders.
    """
    for fold in os.listdir(data_dir):
        fold_path = os.path.join(data_dir, fold)
        if not os.path.isdir(fold_path):
            continue

        print(f"\nValidating {fold}...")

        splits = ["train", "test"]
        patient_sets = {split: set() for split in splits}
        image_sets = {split: set() for split in splits}
        split_stats = {split: {"patients": Counter(), "images": Counter()} for split in splits}

        for split in splits:
            split_path = os.path.join(fold_path, split)
            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if not os.path.isdir(class_path):
                    continue

                for patient_id in os.listdir(class_path):
                    patient_path = os.path.join(class_path, patient_id)
                    if not os.path.isdir(patient_path):
                        continue

                    patient_sets[split].add(patient_id)
                    split_stats[split]["patients"][class_name] += 1

                    for image_name in os.listdir(patient_path):
                        image_path = os.path.join(patient_path, image_name)
                        if image_name.lower().endswith((".png", ".jpg", ".jpeg")):
                            image_sets[split].add(image_path)
                            split_stats[split]["images"][class_name] += 1

        # Validate patient overlap
        patient_overlap = patient_sets["train"].intersection(patient_sets["test"])
        if patient_overlap:
            raise ValueError(f"Error: Patient overlap detected in {fold}: {patient_overlap}")
        else:
            print(f"No patient overlap detected in {fold}.")

        # Validate image overlap
        image_overlap = image_sets["train"].intersection(image_sets["test"])
        if image_overlap:
            raise ValueError(f"Error: Image overlap detected in {fold}: {image_overlap}")
        else:
            print(f"No image overlap detected in {fold}.")

        # Print split stats
        for split in splits:
            print(f"{fold} - {split.capitalize()} Distribution (Patients): {dict(split_stats[split]['patients'])}")
            print(f"{fold} - {split.capitalize()} Distribution (Images): {dict(split_stats[split]['images'])}")



data_dir = "/home/dvegaa/DELFOS/delfos_final_dataset/delfos_images_kfold"
validate_k_folds(data_dir)





