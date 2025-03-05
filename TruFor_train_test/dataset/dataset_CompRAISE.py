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

class compRAISE(AbstractDataset):
    """
    directory structure
    compRAISE
    ├── r000da54ft_Q67.jpg
    ├── r000da54ft_Q67_aligned_Q87.jpg
    └── r000da54ft_Q67_resize_1.15_Q90.jpg ...
    """
    def __init__(self, crop_size, grid_crop, img_list: str, max_dim=None, aug=None):
        super().__init__(crop_size, grid_crop, max_dim, aug=aug)
        self._root_path = dataset_paths['compRAISE']
        with open(project_root / img_list, "r") as f:
            lines = f.readlines()            
        self.img_list = [t.strip() for t in lines]
        

    def get_img(self, index):
        assert 0 <= index < len(self.img_list), f"Index {index} is not available!"
        rgb_path = os.path.join(self._root_path, self.img_list[index])
        assert os.path.isfile(rgb_path)
        return self._create_tensor(mask=None, rgb_path=rgb_path)
    
