import torch
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


def extract_features(img_model, tab_model, loader, device, save_dir, test=False):
    """
    Extracts features from the model and stores them in lists.
    """
    img_features_list = []
    tab_features_list = []
    patient_list = []
    targets_list = []

    img_model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Extracting Image Features"):
            img_input, tab_input, targets = inputs[0].to(device), inputs[1].to(device), targets.to(device)

            img_features = img_model(img_input)
            img_features_list.append(img_features)
            
            if test:
                patient_list.append(inputs[2])


    tab_model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Extracting Tabular Features"):
            img_input, tab_input, targets = inputs[0].to(device), inputs[1].to(device), targets.to(device)

            tab_features = tab_model(tab_input)
            tab_features_list.append(tab_features)
            targets_list.append(targets)

    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
    
    # Stack all extracted features into single tensors
    img_features = torch.cat(img_features_list, dim=0)  # (num_samples, embed_dim)
    tab_features = torch.cat(tab_features_list, dim=0)  # (num_samples, embed_dim)
    targets = torch.cat(targets_list, dim=0)  # (num_samples,)
    if test:
        torch.save(patient_list, os.path.join(save_dir, "patients_ids.pt"))

    # Save the extracted features and targets
    torch.save(img_features, os.path.join(save_dir, "img_features.pt"))
    torch.save(tab_features, os.path.join(save_dir, "tab_features.pt"))
    torch.save(targets, os.path.join(save_dir, "targets.pt"))

    print(f"Saved extracted features for Fold {fold} in {save_dir}")

    return None


# K-Fold Cross Validation
folds = args.folds
best_f1_folds = []

for fold in range(folds):
    set_seed(42)
    
    if args.folds == 3:
        dataset_path = f"/home/dvegaa/DELFOS/delfos_final_dataset/delfos_images_kfold/fold_{fold+1}"
    elif args.folds == 4:
        dataset_path = f"/home/dvegaa/DELFOS/delfos_final_dataset/delfos_images_4kfold/fold_{fold+1}"
    
    train_loader, _, test_loader, num_negatives, num_positives = create_dataloaders(
        dataset_dir = dataset_path,
        json_root = "/home/dvegaa/DELFOS/delfos_final_dataset/delfos_clinical_data_wnm_standarized_woe.json",
        dataset_class=DelfosDataset,
        transform_train=transform_train,
        transform_test=transform_test,
        batch_size=args.batch_size,
        fold=fold,
        multimodal=True
    )
    #dataloader.append((train_loader, test_loader))
    print(num_negatives)
    print(num_positives)

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
    
    
    tab_model = TabularModel(args)
    tab_model = tab_model.build_model()
    if args.tab_checkpoint != None:
        print("tabular model pretrained weights loaded")
        tabular_checkpoint = "/home/dvegaa/DELFOS/DELFOS/tabular_script/tabular_checkpoints/model_2025-01-20_13-42-37.pth"
        checkpoint = torch.load(tabular_checkpoint, map_location = torch.device("cuda"), weights_only=True)
        tab_model.load_state_dict(checkpoint, strict=False)

    tab_model.mlp = torch.nn.Identity()
    tab_model = tab_model.to(device)

    save_dir_train = f"/home/dvegaa/DELFOS/delfos_final_dataset/multimodal_features/fold_{fold}/train"
    extract_features(img_model, tab_model, train_loader, device, save_dir_train)
    
    save_dir_test = f"/home/dvegaa/DELFOS/delfos_final_dataset/multimodal_features/fold_{fold}/test"
    extract_features(img_model, tab_model, test_loader, device, save_dir_test, test=True)