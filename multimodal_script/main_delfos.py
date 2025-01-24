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
from data.multimodal_dataloader import DelfosDataset
from img_script.get_img_model import ImageModel
from tabular_script.get_tab_model import TabularModel
from multimodal_script.get_multimodal_model import MultimodalModel
from multimodal_script.train_multimodal import train_one_epoch
from multimodal_script.evaluate_multimodal import evaluate
from utils import *

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

# Load multimodal model
device = "cuda" if torch.cuda.is_available() else "cpu"

#Get image model
img_model = ImageModel(args)
img_model = img_model.build_model()
if args.img_checkpoint != None:
    checkpoint = torch.load(args.img_checkpoint, map_location = torch.device("cuda"), weights_only=True)
    img_model.load_state_dict(checkpoint, strict=False)

img_model.proj_head = torch.nn.Identity()
img_model = img_model.to(device)

#Get tabular model
tab_model = TabularModel(args)
tab_model = tab_model.build_model()
if args.tab_checkpoint != None:
    checkpoint = torch.load(args.tab_checkpoint, map_location = torch.device("cuda"), weights_only=True)
    tab_model.load_state_dict(checkpoint, strict=False)

tab_model.proj_head = torch.nn.Identity()
tab_model = tab_model.to(device)

#Get multimodal Model
multimodal_model = MultimodalModel(args, img_model=img_model, tab_model=tab_model)
multimodal_model = multimodal_model.build_model()

# Define loss function
loss_weights = torch.tensor([num_negatives / num_positives], dtype=torch.float32).to(device)
loss_weights = loss_weights*args.loss_factor
criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights)

# Define optimizer
optimizer = optim.AdamW(multimodal_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

#Save best model
save_path = os.path.join("multimodal_checkpoints", exp_name, "best_model.pth")
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Run training, validation and evaluation loop
best_f1 = 0.0

for epoch in range(args.num_epochs):
    print(f"Epoch [{epoch+1}/{args.num_epochs}]")

    # Training phase
    train_one_epoch(multimodal_model, train_loader, criterion, optimizer, epoch, device, args)

    # Validation phase
    best_f1 = evaluate(multimodal_model, val_loader, criterion, device, args, mode="val", save_path=save_path, best_f1=best_f1)

# Test phase
print("Loading the best model for test evaluation...")
multimodal_model.load_state_dict(torch.load(save_path))
evaluate(multimodal_model, test_loader, criterion, args, mode="test")

# Finish wandb logging
wandb.finish()