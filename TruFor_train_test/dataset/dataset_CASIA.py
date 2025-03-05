"""
Created by Myung-Joon Kwon
mjkwon2021@gmail.com
July 14, 2020

modified by Fabrizio Guillaro
fabrizio.guillaro@unina.it
September 2022
"""

from project_config import project_root, dataset_paths
from dataset.AbstractDataset import AbstractDataset

import os
import numpy as np
from PIL import Image


class CASIA(AbstractDataset):
    """
    directory structure:
    CASIA2.0_revised
    ├── Au
    ├── Tp
    └── groundtruth
    """

    def __init__(self, crop_size, grid_crop, img_list: str, max_dim=None, aug=None):
        super().__init__(crop_size, grid_crop, max_dim, aug=aug)
        self._root_path    = dataset_paths['CA']
        
        with open(project_root / img_list, "r") as f:
            self.img_list = [t.strip().split(',') for t in f.readlines()]
        
        

    def get_img(self, index):
        assert 0 <= index < len(self.img_list), f"Index {index} is not available!"
        
        root = self._root_path

        mask_path = os.path.join(root, self.img_list[index][1])
        rgb_path  = os.path.join(root, self.img_list[index][0])
            
        if self.img_list[index][1] == 'None':
            mask = None
        else:
            mask = np.array(Image.open(mask_path).convert("L"))
            mask[mask > 0] = 1

        assert os.path.isfile(rgb_path)
        return self._create_tensor(mask=mask, rgb_path=rgb_path)