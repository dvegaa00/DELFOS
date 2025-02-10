import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import os
import numpy as np
from tab_models.TabularModel import TabTransformer
from utils import *


def main(args):
    # Parámetros de configuración (deben coincidir con los usados en el entrenamiento)
    weights_dir = './tabular_script/tab_models'
    n_folds = 3
    embedding_size = 128
    num_heads = 8
    num_layers = 2
    batch_size = 64
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # Diccionario para guardar métricas de cada fold
    fold_metrics = {
        'auc': [], 'precision': [], 'recall': [], 'f1': [], 'threshold': []
    }
    all_val_probs = []
    all_val_true = []

    for fold_idx in range(n_folds):
        print(f"\n=== Evaluando Fold {fold_idx + 1} ===")
        # Cargar datos de validación
        _, _, X_val_fold, y_val_fold = load_fold(fold_idx, args.seed)
        val_dataset = TensorDataset(
            torch.tensor(X_val_fold, dtype=torch.float32),
            torch.tensor(y_val_fold, dtype=torch.float32)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=0, worker_init_fn=worker_init_fn)

        # Instanciar el modelo con la misma arquitectura
        model = TabTransformer(
            num_features=X_val_fold.shape[1],
            dim_embedding=embedding_size,
            num_heads=num_heads,
            num_layers=num_layers
        ).to(device)

        # Ruta del archivo de pesos guardados para este fold
        fold_weights_path = os.path.join(weights_dir, f"best_model_fold_{fold_idx + 1}.pth")
        if os.path.exists(fold_weights_path):
            model.load_state_dict(torch.load(fold_weights_path, map_location=device))
            print(f"Pesos cargados para el fold {fold_idx+1} desde: {fold_weights_path}")
        else:
            print(f"Archivo de pesos no encontrado para el fold {fold_idx+1}.")
            continue

        # Evaluación en el conjunto de validación
        model.eval()
        val_probs = []
        val_true = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).squeeze()
                # Convertir los logits a probabilidades
                outputs_prob = torch.sigmoid(outputs)
                val_probs.extend(outputs_prob.cpu().numpy())
                val_true.extend(targets.cpu().numpy())
        val_probs = np.array(val_probs)
        val_true = np.array(val_true)

        # Búsqueda del threshold óptimo según F1-score
        best_f1_score = 0
        best_threshold = 0.5
        for threshold in np.linspace(0.01, 0.99, 100):
            preds = (val_probs > threshold).astype(float)
            current_f1 = f1_score(val_true, preds, pos_label=1)
            if current_f1 > best_f1_score:
                best_f1_score = current_f1
                best_threshold = threshold

        val_preds = (val_probs > best_threshold).astype(float)

        # Calcular las métricas
        auc = roc_auc_score(val_true, val_probs)
        precision = precision_score(val_true, val_preds, pos_label=1)
        recall = recall_score(val_true, val_preds, pos_label=1)

        fold_metrics['auc'].append(auc)
        fold_metrics['precision'].append(precision)
        fold_metrics['recall'].append(recall)
        fold_metrics['f1'].append(best_f1_score)
        fold_metrics['threshold'].append(best_threshold)

        all_val_probs.extend(val_probs)
        all_val_true.extend(val_true)

        print(f"Fold {fold_idx+1}: AUC: {auc:.4f}, Precisión: {precision:.4f}, Recall: {recall:.4f}, F1: {best_f1_score:.4f}, Threshold: {best_threshold:.4f}")

    # Opcional: calcular un threshold global y métricas globales
    all_val_probs = np.array(all_val_probs)
    all_val_true = np.array(all_val_true)
    best_global_f1 = 0
    best_global_threshold = 0.5
    for threshold in np.linspace(0.01, 0.99, 100):
        preds = (all_val_probs > threshold).astype(float)
        current_f1 = f1_score(all_val_true, preds, pos_label=1)
        if current_f1 > best_global_f1:
            best_global_f1 = current_f1
            best_global_threshold = threshold

    print(f"\nThreshold global óptimo: {best_global_threshold:.4f}")
    print("\n=== Resultados Promedio de Cross-Validation ===")
    print(f"AUC: {np.mean(fold_metrics['auc']):.4f} ± {np.std(fold_metrics['auc']):.4f}")
    print(f"Precisión: {np.mean(fold_metrics['precision']):.4f} ± {np.std(fold_metrics['precision']):.4f}")
    print(f"Recall: {np.mean(fold_metrics['recall']):.4f} ± {np.std(fold_metrics['recall']):.4f}")
    print(f"F1-Score: {np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carga de pesos Crossvalidación")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    args = parser.parse_args()
    main(args)