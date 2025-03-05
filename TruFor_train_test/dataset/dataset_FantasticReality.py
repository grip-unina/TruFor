"""
Created by Myung-Joon Kwon
mjkwon2021@gmail.com
July 7, 2020

modified by Fabrizio Guillaro
fabrizio.guillaro@unina.it
September 2022
"""

from project_config import project_root, dataset_paths
from dataset.AbstractDataset import AbstractDataset

import os
import numpy as np


class FantasticReality(AbstractDataset):
    """
    directory structure:
    FantasticReality
    ├── ColorFakeImages
    ├── ColorRealImages
    └── SegmentationFake
    """

    def __init__(self, crop_size, grid_crop, img_list: str=None, is_auth_list: bool=False, max_dim=None, aug=None):
        super().__init__(crop_size, grid_crop, max_dim, aug=aug)
        self._root_path    = dataset_paths['FR']
        with open(project_root / img_list, "r") as f:
            self.img_list = [t.strip() for t in f.readlines()]
        self._is_auth_list = is_auth_list
        

    def get_img(self, index):
        
        root = self._root_path

        if not self._is_auth_list:
            # tampered image
            assert 0 <= index < len(self.img_list), f"Index {index} is not available!"
            rgb_path = os.path.join(root, 'ColorFakeImages', self.img_list[index])
            mask_path = os.path.join(root, "SegmentationFake", self.img_list[index].replace('.jpg', '.npz'))
            matrix = np.load(mask_path)
            mask = matrix['arr_0'].squeeze()
            mask[mask > 0] = 1
            assert os.path.isfile(rgb_path)
            return self._create_tensor(mask=mask, rgb_path=rgb_path)
        else:
            # authentic image
            assert 0 <= index < len(self.img_list), f"Index {index} is not available!"
            rgb_path = os.path.join(root, 'ColorRealImages', self.img_list[index])
            assert os.path.isfile(rgb_path)
            return self._create_tensor(mask=None, rgb_path=rgb_path)