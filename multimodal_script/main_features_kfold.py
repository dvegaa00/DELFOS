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
import random

delfos_path = pathlib.Path(__name__).resolve().parent.parent
sys.path.append(str(delfos_path))
from utils import *
from data.transformations import transform_train, transform_test
from data.load_data import create_dataloaders
from data.multimodal_dataloader import DelfosDataset
from img_script.get_img_model import ImageModel
from tabular_script.get_tab_model import TabularModel
from multimodal_script.get_multimodal_model import MultimodalModel
from multimodal_script.train_multimodal import train_one_epoch_features
from multimodal_script.evaluate_multimodal import evaluate_features
from utils import *

# Parse the arguments
args = get_main_parser()

exp_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
wandb.init(project="MultiModal", 
           entity="spared_v2", 
           name=exp_name )

config = wandb.config
wandb.log({"args": vars(args),
           "model": "multimodal_kfold"})

# K-Fold Cross Validation
folds = args.folds
best_f1_folds = []

for fold in range(folds):
    set_seed(42)
    
    if args.folds == 3:
        dataset_path = f"/home/dvegaa/DELFOS/delfos_final_dataset/delfos_images_kfold/fold_{fold+1}"
    elif args.folds == 4:
        dataset_path = f"/home/dvegaa/DELFOS/delfos_final_dataset/delfos_images_4kfold/fold_{fold+1}"
    
    # Load pre-extracted features
    ## TRAIN
    train_data = []
    feature_dir = f"/home/dvegaa/DELFOS/DELFOS/multimodal_features/fold_{fold}/train"
    
    img_features = torch.load(os.path.join(feature_dir, "img_features.pt"))
    train_data.append(img_features)
    
    tab_features = torch.load(os.path.join(feature_dir, "tab_features.pt"))
    train_data.append(tab_features)
    
    targets = torch.load(os.path.join(feature_dir, "targets.pt"))
    train_data.append(targets)
    
    ## TEST
    test_data = []
    feature_dir = f"/home/dvegaa/DELFOS/DELFOS/multimodal_features/fold_{fold}/test"
    
    img_features = torch.load(os.path.join(feature_dir, "img_features.pt"))
    test_data.append(img_features)
    
    tab_features = torch.load(os.path.join(feature_dir, "tab_features.pt"))
    test_data.append(tab_features)
    
    targets = torch.load(os.path.join(feature_dir, "targets.pt"))
    test_data.append(targets)

    # Load multimodal model
    device = "cuda" 
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
    elif args.img_model == "resnet18" or args.img_model == "resnet50":
        img_model.fc = torch.nn.Identity()
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
    
    # Define loss function
    loss_weights = torch.tensor(args.loss_factor).to(device)

    print(loss_weights)
    criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights).to(device)

    # Define optimizer
    optimizer = optim.AdamW(multimodal_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    # Run training, validation and evaluation loop
    best_f1 = 0.0
    best_f1_count = 0.0
    patience = 10
    counter = 0
    #Save best model
    save_path = os.path.join("features_checkpoints", exp_name, f"fold{fold}_best_model.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for epoch in range(args.num_epochs):
        print(f"Epoch [{epoch+1}/{args.num_epochs}]")

        # Training phase
        train_one_epoch_features(multimodal_model, train_data, criterion, optimizer, epoch, device, fold, args)

        #Validation phase
        best_f1, current_f1, val_loss = evaluate_features(multimodal_model, test_data, criterion, device, fold, args, mode="val", save_path=save_path, best_f1=best_f1)
        
        print(val_loss)
        if current_f1 >= best_f1_count:
            best_f1_count = current_f1
            counter = 0
        else:
            print("counter + 1")
            counter += 1
            print(counter)
            print(best_f1_count)
        
        if counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break
        
        print(f"best f1-score: {best_f1}")
    # Test phase
    print("Loading the best model for test evaluation...")
    multimodal_model.load_state_dict(torch.load(save_path))
    evaluate_features(multimodal_model, test_data, criterion, device, fold, args, mode="test")

    best_f1_folds.append(best_f1)
    mean_f1 = np.mean(best_f1_folds)
    std_f1 = np.std(best_f1_folds)
    
    wandb.log({"average_f1_score": mean_f1})
    wandb.log({"std_f1_score": std_f1})
# Finish wandb logging
wandb.finish()