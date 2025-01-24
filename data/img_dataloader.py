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
        self.classes = ["Cardiopatia", "No Cardiopatia"]
        self.root = root
        self.transform = transform
        self.transform_doppler = transform_doppler

        self.images = []
        self.labels = []
        self.views = []
        self.doppler_images = []
        self.class_to_idx = {'Cardiopatia': 1, 'No Cardiopatia': 0}

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
                        self.images.append(image_path)
                        self.labels.append(self.class_to_idx[class_name])

    def _convert_png_to_jpeg(self, image_paths):
        for img_path in image_paths:
            if img_path.endswith('.png'):
                img = Image.open(img_path)
                img = img.convert('RGB')
                new_img_path = os.path.splitext(img_path)[0] + '.jpeg'
                # Save as JPEG
                # img.save(new_img_path)
                # Update the image path in the list
                image_paths[image_paths.index(img_path)] = new_img_path
        return image_paths

    def _concatenate_doppler(self, image_paths):
        doppler_image = [np.array(Image.open(img_path)) for img_path in image_paths if "Doppler" in img_path.split("/")[-1]]
        doppler_image = self.transform(doppler_image[0][:, :, :3])

        no_doppler_image = [np.array(Image.open(img_path)) for img_path in image_paths if "Doppler" not in img_path.split("/")[-1]]
        no_doppler_image = self.transform(no_doppler_image[0][:, :, :3])

        concatenated_image = torch.cat((no_doppler_image, doppler_image), 0)

        return concatenated_image

    def _create_patient_volume(self, image_paths):
        images = [np.array(Image.open(img_path)) for img_path in image_paths]
        images = [self.transform(image) for image in images]
        num_channels = 3
        max_images_per_patient = 70
        num_images = len(images)
        volume = np.zeros((num_channels * max_images_per_patient, 224, 224))
        index = 0
        for idx, image in enumerate(images):
            index = idx * 3
            volume[index: index + num_channels, :, :] = image
        return volume, num_images

    def _create_patient_volume_doppler(self, image_paths):
        images = [np.array(Image.open(img_path)) if img_path is not None else np.array(Image.new('RGB', (224, 224))) for img_path in image_paths]
        bool_list = [path is not None for path in image_paths]
        # Convert the list of boolean values into a PyTorch tensor
        has_doppler = torch.tensor(bool_list, dtype=torch.bool)
        images = [self.transform_doppler(image) for image in images]
        num_channels = 3
        max_images_per_patient = 70
        num_images = len(images)
        volume = np.zeros((num_channels * max_images_per_patient, 224, 224))
        volume_mask = torch.zeros(max_images_per_patient, dtype=torch.bool)
        volume_mask[:num_images] = has_doppler
        index = 0
        for idx, image in enumerate(images):
            index = idx * 3
            volume[index: index + num_channels, :, :] = image
        return volume, volume_mask

    def _create_patient_volume_views(self, views):
        max_images_per_patient = 70
        num_images = len(views)
        volume = np.zeros(max_images_per_patient)
        index = 0
        for idx, view in enumerate(views):
            volume[idx] = view
        return torch.Tensor(volume)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        image = np.array(image)
        image = self.transform(image)
        # images = self._create_patient_volume(self.images[idx])
        # doppler_images, doppler_mask = self._create_patient_volume_doppler(self.doppler_images[idx])
        # views = self._create_patient_volume_views(self.views[idx])
        # return images, self.labels[idx], doppler_images, doppler_mask, views
        return image, self.labels[idx]

