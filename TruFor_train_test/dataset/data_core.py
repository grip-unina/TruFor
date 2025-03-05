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

from dataset.dataset_FantasticReality import FantasticReality
from dataset.dataset_IMD2020 import IMD2020
from dataset.dataset_CASIA import CASIA
from dataset.dataset_TampCOCO import tampCOCO
from dataset.dataset_CompRAISE import compRAISE


class myDataset(Dataset):
    def __init__(self, config, crop_size, grid_crop, mode="train", max_dim=None, aug=None):
        self.dataset_list = []
        training_set = config.DATASET.TRAIN
        valid_set    = config.DATASET.VALID
        
        if mode == "train":
            if 'FR' in training_set:
                self.dataset_list.append(FantasticReality(crop_size, grid_crop, "dataset/data/FR_train_list.txt", aug=aug))
                self.dataset_list.append(FantasticReality(crop_size, grid_crop, "dataset/data/FR_auth_train_list.txt", is_auth_list=True, aug=aug))
                
            if 'IMD' in training_set:
                self.dataset_list.append(IMD2020(crop_size, grid_crop, "dataset/data/IMD_train_list.txt", aug=aug))
                
            if 'CA' in training_set:
                self.dataset_list.append(CASIA(crop_size, grid_crop, "dataset/data/CASIA_v2_train_list.txt", aug=aug))
                self.dataset_list.append(CASIA(crop_size, grid_crop, "dataset/data/CASIA_v2_auth_train_list.txt", aug=aug))

            if 'COCO' in training_set:
                self.dataset_list.append(tampCOCO(crop_size, grid_crop, "dataset/data/cm_COCO_train_list.txt",   aug=aug))
                self.dataset_list.append(tampCOCO(crop_size, grid_crop, "dataset/data/sp_COCO_train_list.txt",   aug=aug))
                self.dataset_list.append(tampCOCO(crop_size, grid_crop, "dataset/data/bcm_COCO_train_list.txt",  aug=aug))
                self.dataset_list.append(tampCOCO(crop_size, grid_crop, "dataset/data/bcmc_COCO_train_list.txt", aug=aug))
            
            if 'RAISE' in training_set:
                self.dataset_list.append(compRAISE(crop_size, grid_crop, "dataset/data/compRAISE_train.txt", aug=aug))


        elif mode == "valid":
            if 'FR' in valid_set:
                self.dataset_list.append(FantasticReality(crop_size, grid_crop, "dataset/data/FR_valid_list.txt", max_dim=max_dim, aug=aug))
                self.dataset_list.append(FantasticReality(crop_size, grid_crop, "dataset/data/FR_auth_valid_list.txt", is_auth_list=True, max_dim=max_dim, aug=aug))
                
            if 'IMD' in valid_set:
                self.dataset_list.append(IMD2020(crop_size, grid_crop, "dataset/data/IMD_valid_list.txt", max_dim=max_dim, aug=aug))
            
            if 'CA' in valid_set:
                self.dataset_list.append(CASIA(crop_size, grid_crop, "dataset/data/CASIA_v2_valid_list.txt", max_dim=max_dim, aug=aug))
                self.dataset_list.append(CASIA(crop_size, grid_crop, "dataset/data/CASIA_v2_auth_valid_list.txt", max_dim=max_dim, aug=aug))
            
            if 'COCO' in valid_set:
                self.dataset_list.append(tampCOCO(crop_size, grid_crop, "dataset/data/cm_COCO_valid_list.txt",   max_dim=max_dim, aug=aug))
                self.dataset_list.append(tampCOCO(crop_size, grid_crop, "dataset/data/sp_COCO_valid_list.txt",   max_dim=max_dim, aug=aug))
                self.dataset_list.append(tampCOCO(crop_size, grid_crop, "dataset/data/bcm_COCO_valid_list.txt",  max_dim=max_dim, aug=aug))
                self.dataset_list.append(tampCOCO(crop_size, grid_crop, "dataset/data/bcmc_COCO_valid_list.txt", max_dim=max_dim, aug=aug))
            
            if 'RAISE' in valid_set:
                self.dataset_list.append(compRAISE(crop_size, grid_crop, "dataset/data/compRAISE_valid.txt", max_dim=max_dim, aug=aug))

        else:
            raise KeyError("Invalid mode: " + mode)

        self.crop_size = crop_size
        self.grid_crop = grid_crop
        self.mode = mode
        lengths = [len(ds) for ds in self.dataset_list]
        self.smallest = min(lengths)
        if config.TRAIN.NUM_SAMPLES > 0 and config.TRAIN.NUM_SAMPLES < self.smallest:
            self.smallest = config.TRAIN.NUM_SAMPLES


    def shuffle(self):
        for dataset in self.dataset_list:
            random.shuffle(dataset.img_list)


    def get_filename(self, index):
        it = 0
        while True:
            if index >= len(self.dataset_list[it]):
                index -= len(self.dataset_list[it])
                it += 1
                continue
            return self.dataset_list[it].get_img_name(index)


    def __len__(self):
        if self.mode == 'train':
            # class-balanced sampling
            return self.smallest * len(self.dataset_list)
        else:
            return sum([len(lst) for lst in self.dataset_list])


    def __getitem__(self, index):
        if self.mode == 'train':
            # class-balanced sampling
            if index < self.smallest * len(self.dataset_list):
                return self.dataset_list[index//self.smallest].get_img(index % self.smallest)
            else:
                raise ValueError("Something wrong.")
        else:
            it = 0
            while True:
                if index >= len(self.dataset_list[it]):
                    index -= len(self.dataset_list[it])
                    it += 1
                    continue
                return self.dataset_list[it].get_img(index)


    def get_info(self):
        s = ''
        for ds in self.dataset_list:
            s += f'{ds.__class__.__name__}: \t{len(ds)} images \n'
        s += f'Smallest: {self.smallest}'
        return s




