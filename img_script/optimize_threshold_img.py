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
from data.img_dataloader import DelfosDataset
from img_script.get_img_model import ImageModel
from tabular_script.get_tab_model import TabularModel
from multimodal_script.get_multimodal_model import MultimodalModel
from multimodal_script.train_multimodal import train_one_epoch
from multimodal_script.evaluate_multimodal import evaluate
from utils import *

# Parse the arguments
args = get_main_parser()

exp_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
wandb.init(project="MedVit", 
           entity="spared_v2", 
           name=exp_name )

config = wandb.config
wandb.log({"args": vars(args),
           "model": args.img_model})

# Define parameters for pr curve
pr_curve = {"precision_train":[],
            "recall_train":[],
            "thresholds_train":[],
            "precision_test":[],
            "recall_test":[],
            "thresholds_test":[],}

folds = args.folds
best_thresholds = []

test_loader_all = []
models = []
for fold in range(folds):
    set_seed(42)
    
    if args.folds == 3:
        dataset_path = f"/home/dvegaa/DELFOS/delfos_final_dataset/delfos_images_kfold/fold_{fold+1}"
    elif args.folds == 4:
        dataset_path = f"/home/dvegaa/DELFOS/delfos_final_dataset/delfos_images_4kfold/fold_{fold+1}"
     
    train_loader, _, test_loader, num_negatives, num_positives = create_dataloaders(
        dataset_dir = dataset_path,
        json_root = "",
        dataset_class=DelfosDataset,
        transform_train=transform_train,
        transform_test=transform_test,
        batch_size=args.batch_size,
        fold=fold,
        args=args,
        multimodal=False
    )
    
    test_loader_all.append(test_loader)
    
    # Load multimodal model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #Get image model
    #"/home/dvegaa/DELFOS/MedViT/MedViT_models/2025-01-22-14-26-21/best_model.pth"
    img_model = ImageModel(args)
    img_model = img_model.build_model()
    img_model = img_model.to(device)

    exp_name = args.exp_name
    img_checkpoint = os.path.join("/home/dvegaa/DELFOS/DELFOS/img_script/image_checkpoints", exp_name, f"fold{fold}_best_model.pth")
    img_checkpoint = torch.load(img_checkpoint, map_location = torch.device("cuda"), weights_only=True)
    img_model.load_state_dict(img_checkpoint, strict=False)
    img_model = img_model.to(device)
    models.append(img_model)
    
    # Get best threshold for val split
    best_threshold, precision, recall, thresholds = find_best_threshold_img(img_model, test_loader, device)
    print(f"Best Threshold (Validation): {best_threshold:.4f}")
    best_thresholds.append(best_threshold)
    
    pr_curve["precision_test"].append(precision)
    pr_curve["recall_test"].append(recall)
    pr_curve["thresholds_test"].append(thresholds)

    # Get best threshold for train split for visualization of pr curve
    _, precision, recall, thresholds = find_best_threshold_img(img_model, train_loader, device)
    
    pr_curve["precision_train"].append(precision)
    pr_curve["recall_train"].append(recall)
    pr_curve["thresholds_train"].append(thresholds)

plot_prcurve(pr_curve=pr_curve, path=f"pr_curve/image_{args.img_model}_{args.exp_name}.png")

# Evaluate on the test set using the best threshold
mean_threshold = np.mean(best_thresholds)
print(f"Mean Threshold (Validation): {mean_threshold:.4f}")

metrics_folds ={"precision":[],
                "recall": [],
                "f1-score": [],
                "roc auc": []}

for fold in range(folds):
    mean_threshold = 0.5
    precision_test, recall_test, accuracy_test, f1_test, roc_auc_test, report_test, y_score_test, y_true_test = evaluate_threshold_img(models[fold], test_loader_all[fold], device, mean_threshold)
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
