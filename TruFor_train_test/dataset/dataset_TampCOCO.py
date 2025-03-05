"""
Created by Myung-Joon Kwon
mjkwon2021@gmail.com
27 Jan 2021

modified by Fabrizio Guillaro
fabrizio.guillaro@unina.it
September 2022
"""
from project_config import project_root, dataset_paths
from dataset.AbstractDataset import AbstractDataset

import os
import numpy as np
from PIL import Image


class tampCOCO(AbstractDataset):
    """
    directory structure
    tampCOCO
    ├── cm_images
    ├── cm_masks
    └── sp_images ...
    """
    def __init__(self, crop_size, grid_crop, img_list: str, max_dim=None, aug=None):
        super().__init__(crop_size, grid_crop, max_dim, aug=aug)
        self._root_path = dataset_paths['tampCOCO']
        with open(project_root / img_list, "r") as f:
            lines = f.readlines()
        self.img_list = [t.strip().split(',') for t in lines if os.path.getsize(os.path.join(self._root_path, t.strip().split(',')[0]))]



    def get_img(self, index):
        assert 0 <= index < len(self.img_list), f"Index {index} is not available!"
        rgb_path  = os.path.join(self._root_path, self.img_list[index][0])
        mask_path = os.path.join(self._root_path, self.img_list[index][1])
        mask = np.array(Image.open(mask_path).convert('L'))
        mask[mask > 1] = 1
        assert os.path.isfile(rgb_path)
        return self._create_tensor(mask=mask, rgb_path=rgb_path)
