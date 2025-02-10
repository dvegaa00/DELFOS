import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import os
import copy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,  WeightedRandomSampler
from sklearn.metrics import f1_score
import random
from datetime import datetime
import torch.nn.functional as F

# PREPROCESSING FUNCTIONS #

def calculate_woe(data, feature, target):
    """Calcula el WoE para una característica categórica."""
    eps = 1e-7  # Evitar divisiones por cero
    grouped = data.groupby(feature)[target].agg(['count', 'sum'])
    grouped['non_event'] = grouped['count'] - grouped['sum']
    grouped['event_rate'] = (grouped['sum'] + eps) / grouped['sum'].sum()
    grouped['non_event_rate'] = (grouped['non_event'] + eps) / grouped['non_event'].sum()
    grouped['woe'] = np.log(grouped['event_rate'] / grouped['non_event_rate'])
    return grouped['woe'].to_dict()


def woe_encoder_list(data, feature, target, max_len, n_splits=5):
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    data_encoded = data.copy()
    data_encoded[f"{feature}_woe"] = [[] for _ in range(len(data))]

    for train_index, val_index in skf.split(data, data[target]):
        train_data = data.iloc[train_index]
        val_data = data.iloc[val_index]

        # Expandir las listas en filas separadas
        train_data_exploded = train_data.explode(feature)
        val_data_exploded = val_data.explode(feature)

        # Calcular el WoE usando los datos de entrenamiento
        woe_map = calculate_woe(train_data_exploded, feature, target)

        # Aplicar el WoE a los datos de validación
        val_data_exploded[f"{feature}_woe"] = val_data_exploded[feature].map(woe_map).fillna(0)

        # Reagrupar las filas para reconstruir las listas WoE
        val_data_woe = (
            val_data_exploded.groupby(val_data_exploded.index)[f"{feature}_woe"]
            .apply(list)
            .reindex(val_data.index)
        )

        # Rellenar las listas con ceros (padding)
        for idx, vector in zip(val_index, val_data_woe):
            padded_vector = vector + [0] * (max_len - len(vector))
            data_encoded.at[idx, f"{feature}_woe"] = padded_vector

    return data_encoded


def pad_categorical_lists(df, categorical_features, max_lengths):
    """Rellena las listas categóricas con valores predeterminados ('NA')."""
    df_padded = df.copy()
    for feature, max_len in zip(categorical_features, max_lengths):
        df_padded[feature] = df_padded[feature].apply(
            lambda x: x + ["NA"] * (max_len - len(x)) if isinstance(x, list) else ["NA"] * max_len
        )
    return df_padded


def pad_numeric_features(row, numeric_features, max_lengths):
    """Rellena las características numéricas con valores predeterminados (-1)."""
    for feature, max_len in zip(numeric_features, max_lengths):
        if feature in row:
            value = row[feature]
            if not isinstance(value, list):
                value = [value]
            padded_value = value + [-1] * (max_len - len(value))
            row[feature] = padded_value[:max_len]  # Recortar si excede max_len
    return row


def standardize_data(data, numeric_features, numerical_values):
    """Estandariza los datos ignorando los valores -1."""
    standardized_data = copy.deepcopy(data)

    # Estandarizar características numéricas
    for feature in numeric_features:
        values = []
        for item in data:
            if feature in item:
                if isinstance(item[feature], list):
                    values.extend([value for value in item[feature] if value != -1])
                elif isinstance(item[feature], (int, float)) and item[feature] != -1:
                    values.append(item[feature])
        if values:
            mean = np.mean(values)
            std = np.std(values)
            for item in standardized_data:
                if feature in item:
                    if isinstance(item[feature], list):
                        item[feature] = [
                            (value - mean) / std if value != -1 else -1
                            for value in item[feature]
                        ]
                    elif isinstance(item[feature], (int, float)) and item[feature] != -1:
                        item[feature] = (item[feature] - mean) / std

    # Estandarizar valores numéricos individuales
    for feature in numerical_values:
        values = [item[feature] for item in data if feature in item and item[feature] != -1]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            for item in standardized_data:
                if feature in item and item[feature] != -1:
                    item[feature] = (item[feature] - mean) / std

    return standardized_data


def create_folds(data, folder_path, file_names):
    """Crea los folds basados en los IDs de las imágenes."""
    folds = {}
    for i in file_names:
        files_path = os.path.join(folder_path, i)
        path_id_train = os.path.join(files_path, 'train')
        path_id_test = os.path.join(files_path, 'test')
        id_train = [f for f in os.listdir(path_id_train)]
        id_test = [f for f in os.listdir(path_id_test)]
        folds[f'{i}_train'] = id_train
        folds[f'{i}_test'] = id_test

    fold_1_train, fold_1_test = [], []
    fold_2_train, fold_2_test = [], []
    fold_3_train, fold_3_test = [], []

    for i in data:
        if i['id'] in folds['fold_1_train']:
            fold_1_train.append(i)
        if i['id'] in folds['fold_1_test']:
            fold_1_test.append(i)
        if i['id'] in folds['fold_2_train']:
            fold_2_train.append(i)
        if i['id'] in folds['fold_2_test']:
            fold_2_test.append(i)
        if i['id'] in folds['fold_3_train']:
            fold_3_train.append(i)
        if i['id'] in folds['fold_3_test']:
            fold_3_test.append(i)

    return fold_1_train, fold_1_test, fold_2_train, fold_2_test, fold_3_train, fold_3_test

# ENTRENAMIENTO #
class TabTransformer(nn.Module):
    def __init__(self, num_features, dim_embedding, num_heads, num_layers):
        super(TabTransformer, self).__init__()
        self.embedding = nn.Linear(1, dim_embedding)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_embedding, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp = MLP(input_size=num_features * dim_embedding)

    def forward(self, x):
        batch_size, num_features = x.shape
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.view(batch_size, -1)
        x = self.mlp(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.out = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.out(x)
        return x

# Configuración de semillas
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Función para inicializar los workers de DataLoader con la semilla fija
def worker_init_fn(worker_id, seed):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

# Cargar datos
def load_data(file, seed):
    with open(file, 'r') as file:
        data = json.load(file)
    X, y = [], []
    for i in data:
        embedding = []
        embedding.extend(i['a_patologicos_woe'])
        embedding.extend(i['a_hereditaria_woe'])
        embedding.extend(i['a_farmacologicos_woe'])
        for feature in [
            'plaquetas', 'hemoglobina', 'hematocrito', 'leucocitos', 'neutrofilos',
            'edad', 'semana_imagen', 'imc', 'semana_embarazo', 'anormalidad_cromosomica',
            'tamizaje', 'riesgo_tromboembolico',  
            'riesgo_psicosocial', 'tamizaje_depresion',
            'tabaco', 'deficiencias_nutricionales', 'alcohol', 'ejercicio', 'gestaciones',
            'partos', 'cesareas', 'abortos', 'ectopicos'
        ]:
            value = i.get(feature, -1)
            if isinstance(value, list):
                embedding.extend(value)
            elif isinstance(value, (int, float)):
                embedding.append(value)
        X.append(embedding)
        y.append(i['cardiopatia'])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# Cargar un fold específico
def load_fold(fold_number, seed):
    X_train, y_train = load_data(f'data/TabularData/output_folds/delfos_processed_dataset_fold_{fold_number + 1}_trainf.json', seed)
    X_val, y_val = load_data(f'data/TabularData/output_folds/delfos_processed_dataset_fold_{fold_number + 1}_testf.json', seed)
    return X_train, y_train, X_val, y_val

# Entrenamiento y validación con guardado de pesos
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, HN_epochs, fold_idx, seed, batch_size, train_dataset, num_positivos, weights_dir):
    best_f1_val = 0
    best_threshold = 0.5
    fold_val_probs = []
    
    # Crear el directorio para los pesos si no existe
    os.makedirs(weights_dir, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Entrenamiento
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss_train = total_loss / len(train_loader)
        
        # Validación
        model.eval()
        running_loss_val = 0.0
        val_probs, val_true = [], []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                running_loss_val += loss.item()
                outputs_prob = torch.sigmoid(outputs)
                val_probs.extend(outputs_prob.cpu().numpy())
                val_true.extend(targets.cpu().numpy())
        
        avg_loss_val = running_loss_val / len(val_loader)
        scheduler.step()
        
        # Calcular métricas
        val_probs = np.array(val_probs)
        val_true = np.array(val_true)
        best_f1_score = 0
        
        for threshold in np.linspace(0.01, 0.99, 100):
            preds = (val_probs > threshold).astype(float)
            current_f1 = f1_score(val_true, preds, pos_label=1)
            if current_f1 > best_f1_score:
                best_f1_score = current_f1
                best_threshold = threshold
        
        y_pred_bin_val = (val_probs > best_threshold).astype(float)
        f1_val = f1_score(val_true, y_pred_bin_val, pos_label=1)
        
        print(f'Epoch {epoch + 1}, Train Loss: {avg_loss_train:.4f}, Validation Loss: {avg_loss_val:.4f}, '
              f'Validation F1-Score: {f1_val:.4f}, Best Threshold: {best_threshold:.4f}')
        
        # Guardar el mejor modelo según F1 en validación
        if f1_val > best_f1_val:
            best_f1_val = f1_val
            fold_val_probs = val_probs.copy()
            
            # Guardar los pesos del mejor modelo
            model_path = os.path.join(weights_dir, f'best_model_fold_{fold_idx + 1}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Modelo guardado en {model_path} con F1-Score: {best_f1_val:.4f}")
        
        # Hard Negative Mining (HN)
        if (epoch + 1) % HN_epochs == 0 and epoch != 0:
            model.eval()
            all_preds, y_train_true = [], []
            temp_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=0, worker_init_fn=lambda x: worker_init_fn(x, seed))
            
            with torch.no_grad():
                for inputs, targets in temp_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs).squeeze()
                    outputs_prob = torch.sigmoid(outputs)
                    all_preds.append(outputs_prob.cpu().numpy())
                    y_train_true.append(targets.cpu().numpy())
            
            all_preds = np.concatenate(all_preds)
            y_train_true = np.concatenate(y_train_true)
            false_negatives_mask = (y_train_true == 1) & (all_preds <= best_threshold)
            fn_indices = np.where(false_negatives_mask)[0]
            
            new_sample_weights = torch.ones(len(train_dataset))
            fn_weight = 1 + (len(fn_indices) / num_positivos)
            new_sample_weights[fn_indices] = fn_weight
            
            sampler = WeightedRandomSampler(new_sample_weights, len(new_sample_weights), replacement=True, generator=torch.Generator().manual_seed(seed))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                      shuffle=False, num_workers=0, worker_init_fn=lambda x: worker_init_fn(x, seed))
            
            print(f"Epoch {epoch + 1}: Falsos Negativos: {len(fn_indices)} - Muestreo reforzado")
    
    return best_f1_val, best_threshold, fold_val_probs

