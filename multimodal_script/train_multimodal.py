import torch
from tqdm import tqdm
import wandb

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, device, args):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", unit="batch")

    for inputs, targets in progress_bar:
        img_data, tab_data, targets = inputs[0].to(device), inputs[1].to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(img_data, tab_data)
        outputs = outputs.squeeze(1)
        
        targets = targets.to(torch.float32)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{args.num_epoch}], Average Loss: {avg_loss:.4f}")
    wandb.log({"epoch": epoch+1, "loss": avg_loss})
    