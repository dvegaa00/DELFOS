import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from datetime import datetime
from utils import *

def main(args):
    # Configuración inicial
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Cargar folds y entrenar
    fold_metrics = {'auc': [], 'precision': [], 'recall': [], 'f1': [], 'threshold': []}
    all_val_probs, all_val_true = [], []
    
    for fold_idx in range(args.n_folds):
        print(f"\n=== Fold {fold_idx + 1} ===")
        X_train_fold, y_train_fold, X_val_fold, y_val_fold = load_fold(fold_idx, args.seed)
        num_negativos = (y_train_fold == 0).sum()
        num_positivos = (y_train_fold == 1).sum()
        
        train_dataset = TensorDataset(torch.tensor(X_train_fold, dtype=torch.float32),
                                      torch.tensor(y_train_fold, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val_fold, dtype=torch.float32),
                                    torch.tensor(y_val_fold, dtype=torch.float32))
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=0, worker_init_fn=lambda x: worker_init_fn(x, args.seed))
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, worker_init_fn=lambda x: worker_init_fn(x, args.seed))
        
        model = TabTransformer(num_features=X_train_fold.shape[1], dim_embedding=args.embedding_size,
                               num_heads=args.num_heads, num_layers=args.num_layers).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
        
        loss_weights = torch.tensor([num_negativos / num_positivos], dtype=torch.float32).to(device) * args.loss_weight_factor
        criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights)
        
        best_f1_val, best_threshold, fold_val_probs = train_and_validate(
            model, train_loader, val_loader, criterion, optimizer, scheduler, device, args.epochs, args.HN_epochs,
            fold_idx, args.seed, args.batch_size, train_dataset, num_positivos, args.weights_dir
        )
        
        # Guardar métricas del fold
        val_true = y_val_fold
        val_preds = (fold_val_probs > best_threshold).astype(float)
        fold_metrics['auc'].append(roc_auc_score(val_true, fold_val_probs))
        fold_metrics['precision'].append(precision_score(val_true, val_preds, pos_label=1))
        fold_metrics['recall'].append(recall_score(val_true, val_preds, pos_label=1))
        fold_metrics['f1'].append(best_f1_val)
        fold_metrics['threshold'].append(best_threshold)
        
        all_val_probs.extend(fold_val_probs)
        all_val_true.extend(val_true)
    
    # Calcular threshold global óptimo
    all_val_probs = np.array(all_val_probs)
    all_val_true = np.array(all_val_true)
    best_global_threshold = 0.5
    best_global_f1 = 0
    
    for threshold in np.linspace(0.01, 0.99, 100):
        preds = (all_val_probs > threshold).astype(float)
        current_f1 = f1_score(all_val_true, preds, pos_label=1)
        if current_f1 > best_global_f1:
            best_global_f1 = current_f1
            best_global_threshold = threshold
    
    print(f"Threshold usado: {best_global_threshold:.4f} (óptimo global)")
    print("\n=== Resultados de Cross-Validation ===")
    print(f"AUC: {np.mean(fold_metrics['auc']):.4f} ± {np.std(fold_metrics['auc']):.4f}")
    print(f"Precisión: {np.mean(fold_metrics['precision']):.4f} ± {np.std(fold_metrics['precision']):.4f}")
    print(f"Recall: {np.mean(fold_metrics['recall']):.4f} ± {np.std(fold_metrics['recall']):.4f}")
    print(f"F1-Score: {np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de TabTransformer con Cross-Validation")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    parser.add_argument("--device", type=str, default="cuda:1", help="Dispositivo de entrenamiento (cuda o cpu)")
    parser.add_argument("--n_folds", type=int, default=3, help="Número de folds para cross-validation")
    parser.add_argument("--lr", type=float, default=0.0000172679909611506, help="Tasa de aprendizaje")
    parser.add_argument("--epochs", type=int, default=80, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=64, help="Tamaño del lote")
    parser.add_argument("--num_heads", type=int, default=8, help="Número de cabezas de atención")
    parser.add_argument("--embedding_size", type=int, default=128, help="Tamaño de la dimensión de embedding")
    parser.add_argument("--num_layers", type=int, default=2, help="Número de capas del transformer")
    parser.add_argument("--HN_epochs", type=int, default=10, help="Frecuencia de Hard Negative Mining")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Decaimiento de peso para Adam")
    parser.add_argument("--scheduler_step", type=int, default=14, help="Step size para el scheduler")
    parser.add_argument("--scheduler_gamma", type=float, default=0.5, help="Gamma para el scheduler")
    parser.add_argument("--loss_weight_factor", type=float, default=0.6, help="Factor de peso para la pérdida")
    parser.add_argument("--weights_dir", type=str, default="./model_weights", help="Directorio para guardar los pesos de los modelos")
    args = parser.parse_args()
    main(args)