import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
from datetime import datetime
import os
import pathlib
import sys
import numpy as np
import matplotlib.pyplot as plt

delfos_path = pathlib.Path(__name__).resolve().parent.parent
sys.path.append(str(delfos_path))
from utils import *
from data.transformations import transform_train, transform_test
from data.load_data import create_dataloaders
from data.multimodal_dataloader import DelfosDataset
from img_script.get_img_model import ImageModel
from tabular_script.get_tab_model import TabularModel
from multimodal_script.get_multimodal_model import MultimodalModel
from multimodal_script.train_multimodal import train_one_epoch
from multimodal_script.evaluate_multimodal import evaluate
from utils import *
import random

# Parse the arguments
args = get_main_parser()

exp_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
wandb.init(project="MultiModal", 
           entity="spared_v2", 
           name=exp_name )

config = wandb.config
wandb.log({"args": vars(args),
           "model": "multimodal"})

# Define parameters for pr curve
pr_curve = {"precision":[],
            "recall":[],
            "thresholds":[]}

folds = args.folds
best_thresholds = []

test_loader_all = []
models = []
for fold in range(folds):
    set_seed(42)
    train_loader, _, test_loader, num_negatives, num_positives = create_dataloaders(
        dataset_dir = f"/home/dvegaa/DELFOS/delfos_final_dataset/delfos_images_kfold/fold_{fold+1}",
        json_root = "/home/dvegaa/DELFOS/delfos_final_dataset/delfos_clinical_data_wnm_standarized_woe.json",
        dataset_class=DelfosDataset,
        transform_train=transform_train,
        transform_test=transform_test,
        batch_size=args.batch_size,
        multimodal=True
    )
    
    test_loader_all.append(test_loader)
    
    # Load multimodal model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #Get image model
    #"/home/dvegaa/DELFOS/MedViT/MedViT_models/2025-01-22-14-26-21/best_model.pth"
    img_model = ImageModel(args)
    img_model = img_model.build_model()
    if args.img_checkpoint != None:
        print("image model pretrained weights loaded")
        image_checkpoint = os.path.join("/home/dvegaa/DELFOS/DELFOS/img_script/image_checkpoints", args.img_checkpoint, f"fold{fold}_best_model.pth")
        checkpoint = torch.load(image_checkpoint, map_location = torch.device("cuda"), weights_only=True)
        img_model.load_state_dict(checkpoint, strict=False)

    if args.img_model == "medvit":
        img_model.proj_head = torch.nn.Identity()
    else:
        img_model.head = torch.nn.Identity()
    img_model = img_model.to(device)

    #Get tabular model
    #"/home/dvegaa/DELFOS/MedViT/tabular_models/model_2025-01-20_13-42-37.pth"
    tab_model = TabularModel(args)
    tab_model = tab_model.build_model()
    if args.tab_checkpoint != None:
        print("tabular model pretrained weights loaded")
        tabular_checkpoint = "/home/dvegaa/DELFOS/DELFOS/tabular_script/tabular_checkpoints/model_2025-01-20_13-42-37.pth"
        checkpoint = torch.load(tabular_checkpoint, map_location = torch.device("cuda"), weights_only=True)
        tab_model.load_state_dict(checkpoint, strict=False)

    tab_model.mlp = torch.nn.Identity()
    tab_model = tab_model.to(device)

    #Get multimodal Model
    multimodal_model = MultimodalModel(img_model=img_model, tab_model=tab_model, args=args)
    multimodal_model = multimodal_model.build_model()
    multimodal_model = multimodal_model.to(device)

    exp_name = "2025-01-29-02-51-36"
    multimodal_checkpoint = os.path.join("/home/dvegaa/DELFOS/DELFOS/multimodal_script/multimodal_checkpoints", exp_name, f"fold{fold}_best_model.pth")
    multimodal_checkpoint = torch.load(multimodal_checkpoint, map_location = torch.device("cuda"), weights_only=True)
    multimodal_model.load_state_dict(multimodal_checkpoint, strict=True)
    multimodal_model = multimodal_model.to(device)
    models.append(multimodal_model)
    
    # Assuming model, val_loader, test_loader, and device are defined
    best_threshold, precision, recall, thresholds = find_best_threshold_multimodal(multimodal_model, test_loader, device)
    print(f"Best Threshold (Validation): {best_threshold:.4f}")
    best_thresholds.append(best_threshold)
    
    pr_curve["precision"].append(precision)
    pr_curve["recall"].append(recall)
    pr_curve["thresholds"].append(thresholds)

# Plot pr curve
plt.figure(figsize=(8, 6))
for i, (precision, recall) in enumerate(zip(pr_curve["precision"], pr_curve["recall"]), 1):
    plt.plot(recall, precision, label=f'Fold {i}')

# Add labels and legend
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve per Fold')
plt.legend()
plt.grid()
plt.savefig(f"pr_curve/image_{args.multimodal_model}.png")

# Evaluate on the test set using the best threshold
mean_threshold = np.mean(best_thresholds)
metrics_folds ={"precision":[],
                "recall": [],
                "f1-score": [],
                "roc auc": []}
print(f"Mean Threshold: {mean_threshold:.4f}")


for fold in range(folds):
    precision_test, recall_test, accuracy_test, f1_test, roc_auc_test, y_score_test, y_true_test = evaluate_threshold_multimodal(models[fold], test_loader_all[fold], device, 0.5)
    #print metrics for folds
    print(f"Test Metrics_{fold}:")
    print(f"Precision: {precision_test:.4f}")
    print(f"Recall: {recall_test:.4f}")
    print(f"F1-Score: {f1_test:.4f}")
    print(f"ROC AUC: {roc_auc_test:.4f}")
    
    metrics_folds["precision"].append(precision_test)
    metrics_folds["recall"].append(recall_test)
    metrics_folds["f1-score"].append(f1_test)
    metrics_folds["roc auc"].append(roc_auc_test)
    
# Compute mean and standard deviation for each metric
metrics_summary = {}
for key, values in metrics_folds.items():
    mean_value = np.mean(values)
    std_value = np.std(values)
    metrics_summary[key] = {"mean": mean_value, "std": std_value}

# Print the results
for metric, summary in metrics_summary.items():
    print(f"{metric}: Mean = {summary['mean']:.4f}, Std = {summary['std']:.4f}")
