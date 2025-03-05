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

from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
import torch
import random
import cv2


class AbstractDataset(ABC):

    def __init__(self, crop_size, grid_crop: bool, max_dim=None, aug=None):
        """
        :param crop_size: (H, W) or None. H and W must be the multiple of 8 if grid_crop==True.
        :param grid_crop: T: crop within 8x8 grid. F: crop anywhere.
        :param max_dim: if image is bigger than this size, it is cropped
        :param aug: augmentation
        """
        self._crop_size = crop_size
        self._max_dim   = max_dim
        self._grid_crop = grid_crop

        if grid_crop and crop_size is not None:
            assert crop_size[0] % 8 == 0 and crop_size[1] % 8 == 0

        self.img_list = None
        self.aug = aug
        #if self.aug is not None:
        #    print('Augmentation:', self.aug)


    def _create_tensor(self, mask=None, rgb_path=None):
        ignore_index = -1

        try:
            img_RGB = np.array(Image.open(rgb_path).convert("RGB"))
        except:
            raise ValueError(f'error path: {rgb_path}')

        h, w = img_RGB.shape[0], img_RGB.shape[1]
        
        if mask is None:
            mask = np.zeros((h, w))
        elif mask.shape[0]!=h or mask.shape[1]!=w:
            # a small number of images have a mask that mismatches the size of the image
            print(f'MASK MISMATCH: {rgb_path} \n h:{h}, w:{w}, mask: {mask.shape}', flush=True)
            try:
                mask = np.ascontiguousarray(np.rot90(mask))
                assert mask.shape[0]==h and mask.shape[1]==w
            except:
                mask = cv2.resize(np.uint8(mask), (h, w), interpolation=cv2.INTER_NEAREST)>0

        # augmentation
        if self.aug is not None:
            mask = np.uint8(mask)
            dat = self.aug(image=img_RGB, mask=mask)
            assert dat['image'].dtype==img_RGB.dtype
            assert dat['mask'].dtype==mask.dtype
            img_RGB = dat['image']
            mask = dat['mask']>0
            h, w = img_RGB.shape[0], img_RGB.shape[1]
            del dat

        # cropping
        if self._crop_size is None and self._grid_crop:
            crop_size = (-(-h//8) * 8, -(-w//8) * 8)  # smallest 8x8 grid crop that contains image
        elif self._crop_size is None and not self._grid_crop:
            crop_size = None  # use entire image! no crop, no pad
        else:
            crop_size = self._crop_size

        if crop_size is not None:
            # Pad if crop_size is larger than image size
            if h < crop_size[0] or w < crop_size[1]:
                
                # pad RGB
                if img_RGB is not None:
                    temp = np.full((max(h, crop_size[0]), max(w, crop_size[1]), 3), 127.5)
                    temp[:img_RGB.shape[0], :img_RGB.shape[1], :] = img_RGB
                    img_RGB = temp

                # pad mask
                temp = np.full((max(h, crop_size[0]), max(w, crop_size[1])), ignore_index)  # pad with ignore_index(-1)
                try:
                    temp[:mask.shape[0], :mask.shape[1]] = mask
                    mask = temp
                except:
                    raise ValueError(f'{rgb_path}\nh:{h}, w:{w}, temp:{temp.shape}, mask: {mask.shape}')

            # Determine where to crop
            if self._grid_crop:
                s_r = (random.randint(0, max(h - crop_size[0], 0)) // 8) * 8
                s_c = (random.randint(0, max(w - crop_size[1], 0)) // 8) * 8
            else:
                s_r = random.randint(0, max(h - crop_size[0], 0))
                s_c = random.randint(0, max(w - crop_size[1], 0))

            # crop
            mask    = mask[s_r:s_r+crop_size[0], s_c:s_c+crop_size[1]]
            img_RGB = img_RGB[s_r:s_r+crop_size[0], s_c:s_c+crop_size[1], :]
                
        # cropping big images
        if self._max_dim is not None:
            max_dim = self._max_dim
            # Determine where to crop
            s_r = (max((h - max_dim)//2, 0) // 8) * 8
            s_c = (max((w - max_dim)//2, 0) // 8) * 8

            # crop
            mask    = mask[s_r:s_r+max_dim, s_c:s_c+max_dim]
            img_RGB = img_RGB[s_r:s_r+max_dim, s_c:s_c+max_dim, :]
        
        t_mask = torch.tensor(mask, dtype=torch.long)
        t_RGB  = torch.tensor(img_RGB.transpose(2,0,1), dtype=torch.float)/256.0
        return t_RGB, t_mask
    

    @abstractmethod
    def get_img(self, index):
        pass

    def get_img_name(self, index):
        item = self.img_list[index]
        if isinstance(item, list):
            return item[0]
        else:
            return item

    def __len__(self):
        return len(self.img_list)

