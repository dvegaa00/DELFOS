import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import os
import os.path
from PIL import Image
import random

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using multi-GPU
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class DelfosDataset(Dataset):
    def __init__(self, root, transform, transform_doppler=None):
        self.classes = ["Cardiopatia", "No_Cardiopatia"]
        self.root = root
        self.transform = transform
        self.transform_doppler = transform_doppler

        self.data = []
        self.views = []
        self.doppler_images = []
        self.class_to_idx = {'Cardiopatia': 1, 'No_Cardiopatia': 0}

        # Load images and labels
        self._load_data()

    def _load_data(self):
        for class_idx, class_name in enumerate(self.classes):
            class_folder = os.path.join(self.root, class_name)

            if not os.path.exists(class_folder):
                print(f"Warning: Class folder not found: {class_folder}")
                continue

            for id_folder in os.listdir(class_folder):
                id_path = os.path.join(class_folder, id_folder)
                if not os.path.isdir(id_path):
                    continue

                for image in os.listdir(id_path):
                    image_path = os.path.join(id_path, image)
                    if image.lower().endswith((".png", ".jpg", ".jpeg")):
                        #Preload images
                        image = Image.open(image_path).convert("RGB")
                        image = self.transform(np.array(image)) 
                        label = self.class_to_idx[class_name]
                        
                        self.data.append((image, id_folder, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, patient_id, label = self.data[idx]
        return (image, patient_id), label

