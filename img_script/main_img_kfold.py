import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
from datetime import datetime
import os
import pathlib
import sys

delfos_path = pathlib.Path(__name__).resolve().parent.parent
sys.path.append(str(delfos_path))
from utils import *
from data.transformations import transform_train, transform_test
from data.load_data import create_dataloaders
from img_script.train_img import train_one_epoch
from img_script.evaluate_img import evaluate
from img_script.get_img_model import ImageModel
from data.img_dataloader import DelfosDataset
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

# K-Fold Cross Validation
folds = args.folds

for fold in range(folds):
    set_seed(42)
    # Create Dataloaders
    if args.folds == 3:
        dataset_path = f"/home/dvegaa/DELFOS/delfos_final_dataset/delfos_images_kfold/fold_{fold+1}"
    elif args.folds == 4:
        dataset_path = f"/home/dvegaa/DELFOS/delfos_final_dataset/delfos_images_4kfold/fold_{fold+1}"
    
    train_loader, _, test_loader, num_negatives, num_positives = create_dataloaders(
        dataset_dir = dataset_path,
        json_root="",
        dataset_class=DelfosDataset,
        transform_train=transform_train,
        transform_test=transform_test,
        batch_size=args.batch_size,
        fold=fold,
        args=args
    )

    print("Dataloaders have been successfully created")

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImageModel(args)
    model = model.build_model()

    # Move the model to GPU (if available)
    model = model.to(device)

    # Define loss function
    loss_weights = torch.tensor([num_negatives / num_positives], dtype=torch.float32).to(device)
    if args.loss_factor == 0:
        loss_weights = None
    elif args.loss_factor >= 1:
        loss_weights = torch.tensor(args.loss_factor).to(device)
    else:
        loss_weights = loss_weights*args.loss_factor

    print(loss_weights)
    criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights)
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    # Save best model
    save_path = os.path.join("image_checkpoints", exp_name, f"fold{fold}_best_model.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Run training, validation and evaluation loop
    best_f1 = 0.0
    best_loss_count = 100.0
    patience = 6
    counter = 0

    for epoch in tqdm(range(args.num_epochs)):
        print(f"Epoch [{epoch+1}/{args.num_epochs}]")

        # Training phase
        train_one_epoch(model, train_loader, criterion, optimizer, epoch, device, fold, args)

        # Validation phase
        best_f1, current_f1, val_loss = evaluate(model, test_loader, criterion, device, fold, args, mode="val", save_path=save_path, best_f1=best_f1) 
        
        print(val_loss)
        if val_loss <= best_loss_count:
            best_loss_count = val_loss
            counter = 0
        else:
            print("counter + 1")
            counter += 1
            print(counter)
            print(best_loss_count)
        
        #if counter >= patience:
        #    print(f"Early stopping triggered after {epoch + 1} epochs.")
        #    break
        
        print(f"best f1-score: {best_f1}")
    #torch.save(model.state_dict(), save_path)
    # Test phase
    print("Loading the best model for test evaluation...")
    model.load_state_dict(torch.load(save_path))
    evaluate(model, test_loader, criterion, device, fold, args, mode="test")

# Finish wandb logging
wandb.finish()