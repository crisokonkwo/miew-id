import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np



class MiewIdDataset(Dataset):
    def __init__(self, csv, images_dir, transforms=None, fliplr=False, fliplr_view=[], crop_bbox=False, use_full_image_path=False):

        self.csv = csv#.reset_index()
        self.augmentations = transforms
        self.images_dir = images_dir
        self.fliplr = fliplr
        self.fliplr_view = fliplr_view
        self.crop_bbox = crop_bbox
        self.use_full_image_path = use_full_image_path

    def __len__(self):
        return self.csv.shape[0]
    
    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


    def __getitem__(self, index):
        row = self.csv.iloc[index]

        if self.use_full_image_path:
            image_path = row['file_path']
        else:
            image_path = os.path.join(self.images_dir, row['file_name'])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox = row['bbox']

        if self.crop_bbox:
            x1, y1, w, h = [int(x) for x in bbox]
            image = image[y1 : y1 + h, x1 : x1 + w]
            if min(image.shape) < 1:
                # Use original image
                print('Using original image. Invalid bbox', bbox)
                print(image_path)
                image = self.load_image(image_path)
                bbox = [0, 0, image.shape[1], image.shape[0]]

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        if self.fliplr:
            if row['viewpoint'] in self.fliplr_view:
                image = torch.from_numpy(np.fliplr(image).copy())

        
        return {"image": image, "label":torch.tensor(row['name']), "image_idx": self.csv.index[index], "file_path": image_path, "bbox": torch.Tensor(bbox)}