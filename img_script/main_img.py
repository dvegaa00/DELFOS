import torch.nn as nn
import torch
import torch.optim as optim
from utils import *
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

# Parse the arguments
args = get_main_parser()

exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
wandb.init(project="MedVit", 
           entity="spared_v2", 
           name=exp_name )

config = wandb.config
wandb.log({"args": args,
           "model": args.model})

# Example usage
train_loader, val_loader, test_loader, num_negatives, num_positives = create_dataloaders(
    dataset_dir = "/home/dvegaa/DELFOS/delfos_final_dataset/delfos_images_split",
    dataset_class=DelfosDataset,
    transform_train=transform_train,
    transform_test=transform_test,
    batch_size=args.batch_size
)


# Create model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ImageModel(args)
model = model.build_model()
print(model)

# Move the model to GPU (if available)
model = model.to(device)

# Define loss function
loss_weights = torch.tensor([num_negatives / num_positives], dtype=torch.float32).to(device)
loss_weights = loss_weights*args.loss_factor
criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights)
print(loss_weights)

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

#Save best model
save_path = os.path.join("image_checkpoints", exp_name, "best_model.pth")
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Run training, validation and evaluation loop
best_f1 = 0.0

for epoch in tqdm(range(args.num_epochs)):
    print(f"Epoch [{epoch+1}/{args.num_epochs}]")

    # Training phase
    train_one_epoch(model, train_loader, criterion, optimizer, epoch, device, args)

    # Validation phase
    best_f1 = evaluate(model, val_loader, criterion, device, args, mode="Validation", save_path=save_path, best_f1=best_f1)

# Test phase
print("Loading the best model for test evaluation...")
model.load_state_dict(torch.load(save_path))
evaluate(model, test_loader, criterion, args, mode="Test")

# Finish wandb logging
wandb.finish()