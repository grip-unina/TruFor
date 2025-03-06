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

import os
from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = 'weights'
_C.LOG_DIR = 'log'
_C.GPUS = (0,)
_C.WORKERS = 4

# Cudnn parameters
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# Model parameters
_C.MODEL = CN()
_C.MODEL.NAME = 'detconfcmx'
_C.MODEL.PRETRAINED = 'pretrained_models/segformers/mit_b2.pth'
_C.MODEL.MODS = ('RGB','NP++')
_C.MODEL.EXTRA = CN(new_allowed=True)
_C.MODEL.EXTRA.BACKBONE = 'mit_b2'
_C.MODEL.EXTRA.DETECTION = None
_C.MODEL.EXTRA.MODULES = ['NP++','backbone','loc_head','conf_head','det_head']  # modules
# ['NP++',     -> Noiseprint++ extraction module
#  'backbone', -> encoder backbone
#  'loc_head', -> localization head
#  'conf_head',-> confidence head
#  'det_head'] -> detection head
_C.MODEL.EXTRA.FIX_MODULES = ['NP++']  # freezed modules

_C.LOSS = CN()
_C.LOSS.USE_OHEM = False
_C.LOSS.LOSSES = [['LOC', 1.0, 'cross_entropy']] # tuples (Loss, weight, criterion)
    # 'LOC' -> Localization Loss
    # 'CONF'-> Confidence Loss
    # 'DET' -> Detection Loss
# es:
    # -['LOC', 1.0, 'cross_entropy']
    # -['CONF', 1.0, 'mse']
    # -['DET',0.5,'cross_entropy']
_C.LOSS.SMOOTH = 0

# Dataset parameters
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.TRAIN = []
_C.DATASET.VALID = []
_C.DATASET.NUM_CLASSES = 2
_C.DATASET.CLASS_WEIGHTS = [0.5, 2.5]

# Training parameters
_C.TRAIN = CN()

_C.TRAIN.IMAGE_SIZE = [512, 512]  # width * height

_C.TRAIN.LR = 0.01
_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.IGNORE_LABEL = -1

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 100  # also used to compute LR adjustment!
_C.TRAIN.STOP_EPOCH = -1  # to stop before end_epoch
_C.TRAIN.EXTRA_EPOCH = 0

_C.TRAIN.RESUME = True
_C.TRAIN.PRETRAINING = ''  # to start from a fully pretrained network
_C.TRAIN.AUG = None
_C.TRAIN.BATCH_SIZE_PER_GPU = 18
_C.TRAIN.SHUFFLE = True
_C.TRAIN.NUM_SAMPLES = 0  # number of images for each dataset (upper limit is the size of the smaller dataset)

# Validation parameters
_C.VALID = CN()
_C.VALID.IMAGE_SIZE = None  # width * height
_C.VALID.AUG = None
_C.VALID.FIRST_VALID = True # To run a validation before training
_C.VALID.MAX_SIZE = None
_C.VALID.BEST_KEY = 'avg_mIoU'

# Testing parameters
_C.TEST = CN()
_C.TEST.MODEL_FILE = ''



def update_config(cfg, args):
    cfg.defrost()

    cfg.merge_from_file(f'lib/config/{args.experiment}.yaml')
    if cfg.TEST.MODEL_FILE == '':
        cfg.merge_from_list(['TEST.MODEL_FILE', f'weights/{args.experiment}/best.pth.tar'])
    try:
        cfg.merge_from_list(['GPUS', tuple(args.gpu)])
    except:
        pass

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()

