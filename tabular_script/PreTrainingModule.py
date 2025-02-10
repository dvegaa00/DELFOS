import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import os
import copy
from utils import *


def main(input_file, output_dir, image_folder_path):
    # Cargar los datos desde el archivo JSON
    with open(input_file, 'r') as file:
        data = json.load(file)

    df = pd.DataFrame(data)

    categorical_features = ["ocupacion", "a_patologicos", "a_hereditaria", "a_farmacologicos"]
    max_lengths_categorical = [2, 8, 8, 9]

    df_padded = pad_categorical_lists(df, categorical_features, max_lengths_categorical)

    # Aplicar WoE a las características categóricas
    for feature, max_len in zip(categorical_features, max_lengths_categorical):
        df_padded = woe_encoder_list(df_padded, feature, "cardiopatia", max_len)

    numeric_features = [
        "plaquetas", "hemoglobina", "hematocrito", "leucocitos",
        "neutrofilos", "edad", "semana_imagen", "imc", "semana_embarazo"
    ]
    max_lengths_numeric = [2, 2, 2, 2, 10, 10, 10, 10, 10]

    df_padded = df_padded.apply(lambda row: pad_numeric_features(row, numeric_features, max_lengths_numeric), axis=1)
    data_padded = df_padded.to_dict(orient="records")

    # Estandarizar los datos
    standardized_data = standardize_data(data_padded, numeric_features, [
        "ecografia_primer_trimestre", "anormalidad_cromosomica", "tamizaje", "riesgo_obstetrico",
        "riesgo_tromboembolico", "riesgo_psicosocial", "tamizaje_depresion", "transfusion", "tabaco",
        "deficiencias_nutricionales", "alcohol", "ejercicio", "gestaciones", "partos", "cesareas",
        "abortos", "ectopicos"
    ])

    # Crear los folds basados en los IDs de las imágenes
    file_names = ['fold_1', 'fold_2', 'fold_3']
    fold_1_train, fold_1_test, fold_2_train, fold_2_test, fold_3_train, fold_3_test = create_folds(
        standardized_data, image_folder_path, file_names
    )
    os.makedirs(output_dir, exist_ok=True)
    fold_files = {
        "fold_1_train": fold_1_train,
        "fold_1_test": fold_1_test,
        "fold_2_train": fold_2_train,
        "fold_2_test": fold_2_test,
        "fold_3_train": fold_3_train,
        "fold_3_test": fold_3_test
    }

    for fold_name, fold_data in fold_files.items():
        output_file = os.path.join(output_dir, f"delfos_processed_dataset_{fold_name}f.json")
        with open(output_file, 'w') as file:
            json.dump(fold_data, file, indent=4)


if __name__ == "__main__":
    input_file = "DELFOS/data/TabularData/delfos_clinical_data_wm.json"  
    output_dir = "DELFOS/data/TabularData/output_folds" 
    image_folder_path = "./delfos_images_kfold"  # Carpeta con los IDs de las imágenes

    main(input_file, output_dir, image_folder_path)