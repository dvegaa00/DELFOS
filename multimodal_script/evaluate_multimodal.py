import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import wandb

def evaluate(model, loader, criterion, device, args, mode="val", save_path=None, best_f1=None):
    model.eval()
    total_loss = 0.0
    y_true = torch.tensor([], dtype=torch.int64)
    y_score = torch.tensor([])

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f"{mode}ing", unit="batch"):
            img_data, tab_data, targets = inputs[0].to(device), inputs[1].to(device), targets.to(device)

            # Forward pass
            outputs = model(img_data, tab_data).squeeze(1)
            
            # Pérdida
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            # Predicciones
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()

            # Acumular resultados
            y_score_val = torch.cat((y_score_val, probabilities), 0)
            y_true_val = torch.cat((y_true_val, targets), 0)


    avg_loss = total_loss / len(loader)
    print(f"{mode} Loss: {avg_loss:.4f}")
    wandb.log({f"{mode.lower()}_loss": avg_loss})

    # Compute metrics
    y_score = y_score.detach().numpy()
    y_true = y_true.detach().numpy()
    accuracy = accuracy_score(y_true, y_score)
    precision = precision_score(y_true, y_score)
    recall = recall_score(y_true, y_score)
    f1 = f1_score(y_true, y_score)

    print(f"{mode} Metrics: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    wandb.log({
        f"{mode.lower()}_accuracy": accuracy,
        f"{mode.lower()}_precision": precision,
        f"{mode.lower()}_recall": recall,
        f"{mode.lower()}_f1_score": f1,
    })

    if mode == "val" and best_f1 is not None and f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), save_path)
        print(f"New best model saved with F1 score: {best_f1:.4f}")

    if mode == "test":
        roc_auc = roc_auc_score(y_true, y_score)
        report = classification_report(y_true, y_score, target_names=["No Cardiopatia", "Cardiopatia"], output_dict=True)
        wandb.log({
            "roc_auc": roc_auc,
            "precision_No_Cardiopatia": report["No Cardiopatia"]["precision"],
            "recall_No_Cardiopatia": report["No Cardiopatia"]["recall"],
            "f1_No_Cardiopatia": report["No Cardiopatia"]["f1-score"],
            "precision_Cardiopatia": report["Cardiopatia"]["precision"],
            "recall_Cardiopatia": report["Cardiopatia"]["recall"],
            "f1_Cardiopatia": report["Cardiopatia"]["f1-score"]
        })

    return best_f1 if mode == "Validation" else None