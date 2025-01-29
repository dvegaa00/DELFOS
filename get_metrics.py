from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from collections import defaultdict
import wandb
import torch
import numpy as np

def compute_patient_metrics(y_score_patient, y_true_patient, mode="val", fold=None, threshold=0.5):
    # Average scores for each patient
    y_score_patient_avg = {pid: np.mean(scores) for pid, scores in y_score_patient.items()}
    y_true_patient = {pid: label for pid, label in y_true_patient.items()}

    # Convert to arrays for metric computation
    y_true = np.array(list(y_true_patient.values()))
    y_score = np.array(list(y_score_patient_avg.values()))
    y_pred = (y_score > threshold).astype(int)  # Apply threshold of 0.5

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"{mode} Metrics (Patient-level): Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    wandb.log({
        f"{mode.lower()}_accuracy_patient": accuracy,
        f"{mode.lower()}_precision_patient": precision,
        f"{mode.lower()}_recall_patient": recall,
        f"{mode.lower()}_f1_score_patient": f1,
    })

    if mode == "test":
        roc_auc = roc_auc_score(y_true, y_score)
        report = classification_report(y_true, y_pred, target_names=["No Cardiopatia", "Cardiopatia"], output_dict=True)
        print("Classification Report:\n", classification_report(y_true, y_pred, target_names=["No Cardiopatia", "Cardiopatia"]))
        wandb.log({
            f"{fold}_roc_auc_patient": roc_auc,
            f"{fold}_precision_No_Cardiopatia_patient": report["No Cardiopatia"]["precision"],
            f"{fold}_recall_No_Cardiopatia_patient": report["No Cardiopatia"]["recall"],
            f"{fold}_f1_No_Cardiopatia_patient": report["No Cardiopatia"]["f1-score"],
            f"{fold}_precision_Cardiopatia_patient": report["Cardiopatia"]["precision"],
            f"{fold}_recall_Cardiopatia_patient": report["Cardiopatia"]["recall"],
            f"{fold}_f1_Cardiopatia_patient": report["Cardiopatia"]["f1-score"]
        })

    return f1 
