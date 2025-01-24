import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import os
import os.path
from PIL import Image
import random
import json

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using multi-GPU
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class DelfosDataset(Dataset):
    def __init__(self, root, json_root, transform, transform_doppler=None):
        self.classes = ["Cardiopatia", "No Cardiopatia"]
        self.root = root
        self.json_root = json_root
        self.transform = transform
        self.transform_doppler = transform_doppler

        self.images = []
        self.labels = []
        self.patient_info = {}
        self.class_to_idx = {'Cardiopatia': 1, 'No Cardiopatia': 0}

        # Load patient info from JSON
        self._load_patient_info()

        # Load images and labels
        self._load_data()

    def _load_patient_info(self):
        """Load patient information from JSON."""
        if not os.path.exists(self.json_root):
            raise FileNotFoundError(f"JSON file not found: {self.json_root}")

        with open(self.json_root, "r") as f:
            data = json.load(f)

        for patient in data:
            patient_id = patient['id']
            embedding = []
            # Extract patient-specific features
            embedding.extend(patient.get('ocupacion_woe', []))
            embedding.extend(patient.get('a_patologicos_woe', []))
            embedding.extend(patient.get('a_hereditaria_woe', []))
            embedding.extend(patient.get('a_farmacologicos_woe', []))
            for feature in [
                'plaquetas', 'hemoglobina', 'hematocrito', 'leucocitos', 'neutrofilos',
                'edad', 'semana_imagen', 'imc', 'semana_embarazo',
                'ecografia_primer_trimestre', 'anormalidad_cromosomica', 'tamizaje',
                'riesgo_obstetrico', 'riesgo_tromboembolico', 'riesgo_psicosocial',
                'tamizaje_depresion', 'transfusion', 'tabaco', 'deficiencias_nutricionales',
                'alcohol', 'ejercicio', 'gestaciones', 'partos', 'cesareas', 'abortos', 'ectopicos'
            ]:
                value = patient.get(feature, -1)  # Default to -1 if feature is missing
                if isinstance(value, list):
                    embedding.extend(value)
                elif isinstance(value, (int, float)):
                    embedding.append(value)
            self.patient_info[patient_id] = np.array(embedding, dtype=np.float32)

    def _load_data(self):
        """Load image paths and labels from class directories."""
        for class_name in self.classes:
            class_folder = os.path.join(self.root, class_name)

            if not os.path.exists(class_folder):
                print(f"Warning: Class folder not found: {class_folder}")
                continue

            for id_folder in os.listdir(class_folder):
                id_path = os.path.join(class_folder, id_folder)
                if not os.path.isdir(id_path):
                    continue

                # Check if patient info exists for this ID
                patient_info = self.patient_info.get(id_folder)
                if patient_info is None:
                    print(f"Warning: No patient info found for {id_folder}")
                    continue

                for image in os.listdir(id_path):
                    image_path = os.path.join(id_path, image)
                    if image.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.images.append((image_path, patient_info))
                        self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load and transform image
        image_path, patient_info = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(np.array(image))

        # Label and patient feature vector
        label = self.labels[idx]
        return (image, torch.tensor(patient_info, dtype=torch.float32)), label
