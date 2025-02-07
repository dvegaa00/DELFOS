import os

# Path to the main directory
base_path = "/home/dvegaa/DELFOS/delfos_final_dataset/delfos_images_kfold"

# Store patient IDs in sets for each fold
folds = {}
for fold in os.listdir(base_path):
    fold_path = os.path.join(base_path, fold)
    if os.path.isdir(fold_path):
        train_patients = set()
        test_patients = set()
        
        # Check train folder
        train_path = os.path.join(fold_path, "train")
        if os.path.exists(train_path):
            for category in os.listdir(train_path):  # Cardiopatia, No Cardiopatia
                category_path = os.path.join(train_path, category)
                if os.path.isdir(category_path):
                    train_patients.update(os.listdir(category_path))

        # Check test folder
        test_path = os.path.join(fold_path, "test")
        if os.path.exists(test_path):
            for category in os.listdir(test_path):  # Cardiopatia, No Cardiopatia
                category_path = os.path.join(test_path, category)
                if os.path.isdir(category_path):
                    test_patients.update(os.listdir(category_path))

        folds[fold] = {"train": train_patients, "test": test_patients}

# Check for overlaps between train and test sets
for fold, data in folds.items():
    train_set = data["train"]
    test_set = data["test"]
    overlap = train_set.intersection(test_set)

    if overlap:
        print(f"⚠️ Overlap found in {fold}: {overlap}")
    else:
        print(f"✅ No overlap in {fold}")

# Check for overlaps between folds
all_train_sets = [data["train"] for data in folds.values()]
all_test_sets = [data["test"] for data in folds.values()]

for i in range(len(all_train_sets)):
    for j in range(i + 1, len(all_train_sets)):
        fold_i = list(folds.keys())[i]
        fold_j = list(folds.keys())[j]

        train_overlap = all_train_sets[i].intersection(all_train_sets[j])
        test_overlap = all_test_sets[i].intersection(all_test_sets[j])

        #if train_overlap:
        #    print(f"⚠️ Overlap in training sets between {fold_i} and {fold_j}: {train_overlap}")
        if test_overlap:
            print(f"⚠️ Overlap in test sets between {fold_i} and {fold_j}: {test_overlap}")

print("Check completed!")
