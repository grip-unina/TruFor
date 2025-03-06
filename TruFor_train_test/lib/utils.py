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

import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import itertools



def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, power=0.9):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr
    return lr


class FullModel(nn.Module):
    """
    Distribute the loss on multi-gpu to reduce the memory cost in the main gpu.
    """
    def __init__(self, model, config=None):
        super(FullModel, self).__init__()
        self.model = model
        self.model_name = config.MODEL.NAME
        self.cfg = config
        self.losses = config.LOSS.LOSSES
        self.loss_loc, self.loss_conf, self.loss_det = get_criterion(config)

    def forward(self, labels=None, rgbs=None):
        outputs, conf, det, npp = self.model(rgbs)
        final_loss = 0
        for (l,w,_) in self.losses:
            if l == 'LOC':
                loss = self.loss_loc(outputs, labels)     # localization loss
            elif l == 'CONF':
                loss = self.loss_conf(outputs, labels, conf)  # confidence loss
            elif l == 'DET':
                loss = self.loss_det(det, labels)             # detection loss

            loss = torch.unsqueeze(loss, 0)
            final_loss += w * loss

        return final_loss, outputs, conf, det





def get_model(config):
    if config.MODEL.NAME == 'detconfcmx':
        from lib.models.cmx.builder_np_conf import EncoderDecoder as detconfcmx
        return detconfcmx(cfg=config)
    else:
        raise NotImplementedError("Model not implemented")


def get_criterion(config):
    ignore_label = config.TRAIN.IGNORE_LABEL
    smooth       = config.LOSS.SMOOTH
    weight       = torch.FloatTensor(config.DATASET.CLASS_WEIGHTS)

    losses         = config.LOSS.LOSSES
    detection      = config.MODEL.EXTRA.DETECTION

    criterion_loc, criterion_conf, criterion_det = None, None, None

    for (l,_,criterion) in losses:
        assert l in ['LOC', 'CONF', 'DET']

        # Training the Localization Network
        if l == 'LOC':
            if criterion == 'dice':
                from lib.core.criterion import DiceLoss
                criterion_loc = DiceLoss(ignore_label=ignore_label, smooth=smooth).cuda()
            elif criterion == 'binary_dice':
                from lib.core.criterion import BinaryDiceLoss
                criterion_loc = BinaryDiceLoss(ignore_label=ignore_label, smooth=smooth).cuda()
            elif criterion == 'cross_entropy':
                from lib.core.criterion import CrossEntropy
                criterion_loc = CrossEntropy(ignore_label=ignore_label, weight=weight).cuda()
            elif criterion == 'dice_entropy':
                from lib.core.criterion import DiceEntropyLoss
                criterion_loc = DiceEntropyLoss(ignore_label=ignore_label, weight=weight, smooth=smooth).cuda()
            else:
                raise ValueError('Localization loss not implemented')

        # Training the Confidence
        elif l == 'CONF':
            if criterion == 'mse':
                from lib.core.criterion_conf import MSE
                criterion_conf = MSE().cuda()
            else:
                raise ValueError('Confidence loss not implemented')

        # Training the Detector
        elif l == 'DET':
            if detection is not None and not detection == 'none':
                if criterion == 'cross_entropy':
                    from lib.core.criterion_det import CrossEntropy
                    criterion_det = CrossEntropy().cuda()
                else:
                    raise ValueError('Detection loss not implemented')

    return criterion_loc, criterion_conf, criterion_det



def get_optimizer(model, config):
    if 'cmx' in config.MODEL.NAME:
        from lib.models.cmx.init_func import group_weight
        params_list = []
        params_list = group_weight(params_list, model, nn.BatchNorm2d, config.TRAIN.LR)
    else:
        params_list = [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': config.TRAIN.LR}]

    if config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(params_list,
                                    lr = config.TRAIN.LR,
                                    momentum = config.TRAIN.MOMENTUM,
                                    weight_decay = config.TRAIN.WD,
                                    nesterov = config.TRAIN.NESTEROV)
    elif config.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(params_list,
                                     lr = config.TRAIN.LR,
                                     betas = (0.9, 0.999),
                                     weight_decay = config.TRAIN.WD)
    else:
        raise ValueError('Optimizer not implemented')

    return optimizer




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg



def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    model = cfg.MODEL.NAME
    final_output_dir = root_output_dir / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name.replace('/','_'), time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / model / (cfg_name + '_' + time_str)
    return logger, str(final_output_dir), str(tensorboard_log_dir)



def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix



#### se ho un canale
def get_confusion_matrix_1ch(label, confid, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    #label: tcp_binary 

    output = confid.squeeze(dim=1).cpu().numpy()
    
    # confid is without the sigmoid, so have to do >0
    seg_pred = np.asarray(output>0, dtype=np.uint8)
    seg_gt = np.asarray(
        label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix):    
    
    fig = plt.figure(figsize=(3, 3), dpi=200, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(confusion_matrix, cmap='bwr')
    
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_xticks([0,1])
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=10)
    ax.set_yticks([0,1])
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()
    
    for i, j in itertools.product(range(2), range(2)):
        ax.text(j, i, format(confusion_matrix[i, j], '.3e') if confusion_matrix[i,j]!=0 else '.', horizontalalignment="center", fontsize=10, verticalalignment='center', color= "black")
    
    fig.set_tight_layout(True)
    fig.colorbar(im,fraction=0.046, pad=0.04)
    
    fig.canvas.draw()
    canvas = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    cm = np.frombuffer(canvas, dtype=np.uint8).reshape(nrows, ncols, 3).transpose(2, 0, 1)
    plt.close(fig)
    return cm
