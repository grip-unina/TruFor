# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt

"""
Created in September 2022
@author: fabrizio.guillaro
"""

from torch.utils.data import Dataset
import random
import numpy as np
import torch
from PIL import Image


class myDataset(Dataset):
    def __init__(self, list_img=None):
        self.tamp_list = list_img

    def shuffle(self):
        random.shuffle(self.tamp_list)

    def __len__(self):
        return len(self.tamp_list)

    def __getitem__(self, index):
        assert self.tamp_list
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        rgb_path = self.tamp_list[index]
        img_RGB = np.array(Image.open(rgb_path).convert("RGB"))
        return torch.tensor(img_RGB.transpose(2, 0, 1), dtype=torch.float) / 256.0, rgb_path

    def get_filename(self, index):
        item = self.tamp_list[index]
        if isinstance(item, list):
            return item[0]
        else:
            return item

