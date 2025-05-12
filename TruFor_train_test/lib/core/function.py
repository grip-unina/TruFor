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
import os
import time

import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F

from lib.utils import AverageMeter
from lib.utils import get_confusion_matrix, get_confusion_matrix_1ch
from lib.utils import adjust_learning_rate as default_adjust_learning_rate

def train(epoch, num_epoch, epoch_iters, base_lr, num_iters,
          trainloader, optimizer, model, writer_dict,
          adjust_learning_rate=default_adjust_learning_rate):
    
    # Training
    model.train()
    batch_time = AverageMeter()
    avg_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    
    for i_iter, (rgbs, labels) in enumerate(tqdm(trainloader)):
        rgbs = rgbs.cuda()
        labels = labels.long().cuda()

        losses, *_ = model(labels=labels, rgbs=rgbs)
        loss = losses.mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        avg_loss.update(loss.item())

        lr = adjust_learning_rate(optimizer, base_lr, num_iters, i_iter+cur_iters)
      
    print_loss = avg_loss.average()
    msg = 'Epoch: [{}/{}], Time: {:.2f}, ' \
          'lr: {:.6f}, Loss: {:.6f}' .format(
              epoch, num_epoch, batch_time.average(), lr, print_loss)
    logging.info(msg)

    writer.add_scalar('train_loss', print_loss, global_steps)
    writer.add_scalar('learning_rate', lr, global_steps)
    global_steps += 1
    writer_dict['train_global_steps'] = global_steps



def validate(config, testloader, model, writer_dict, valid_set="valid"):

    model.eval()
    avg_loss = AverageMeter()
    confusion_matrix      = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    confusion_matrix_CONF = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    

    # PRED metrics
    avg_mse      = AverageMeter()

    avg_mIoU     = AverageMeter()
    avg_p_mIoU   = AverageMeter()
    avg_mIoU_s   = AverageMeter() # smoothed
    avg_p_mIoU_s = AverageMeter() # smoothed
    avg_IoU_1_s  = AverageMeter() # smoothed

    avg_p_F1     = AverageMeter()
    avg_p_F1_s   = AverageMeter()

    # CONF metrics
    c_avg_mse    = AverageMeter()
    c_avg_mIoU   = AverageMeter()
    c_avg_mIoU_s = AverageMeter() # smoothed

    # DET metrics
    avg_det_tpr = AverageMeter()
    avg_det_tnr = AverageMeter()

    
    with torch.no_grad():
        for it, (rgb, label) in enumerate(tqdm(testloader)):
            size = label.size()

            rgb = rgb.cuda()
            label = label.long().cuda()

            losses, pred, conf, det = model(labels=label, rgbs=rgb)
            
            if pred is not None:
                pred = F.upsample(input=pred, size=(size[-2], size[-1]), mode='bilinear')
                pred_prob = F.softmax(pred, dim=1)
                
            if conf is not None:
                conf = F.upsample(input=conf, size=(size[-2], size[-1]), mode='bilinear')
                tcp  = pred_prob[:,1]*(label==1) + pred_prob[:,0]*(label==0)
            
            loss = losses.mean()
            avg_loss.update(loss.item())

            smooth = 1.

            # PRED METRICS
            current_confusion_matrix = get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            val_mse = torch.mean((pred_prob[:,1]-label)**2).item()
            avg_mse.update(val_mse)
            confusion_matrix += current_confusion_matrix

            TN = current_confusion_matrix[0, 0]
            FN = current_confusion_matrix[1, 0]
            FP = current_confusion_matrix[0, 1]
            TP = current_confusion_matrix[1, 1]
            pos = current_confusion_matrix.sum(1)    # ground truth label count
            res = current_confusion_matrix.sum(0)    # prediction count
            tp  = np.diag(current_confusion_matrix)  # Intersection part
            
            # mIoU
            IoU_array = (tp / np.maximum(1.0, pos + res - tp))  # Union part
            mean_IoU = IoU_array.mean()                         # mean of the classes IoU            
            avg_mIoU.update(mean_IoU)
            
            # mIoU_s
            IoU_array_smooth = ((tp + smooth)/ (pos + res - tp + smooth))  # Union part
            mean_IoU_smooth = IoU_array_smooth.mean()                      # mean of the classes IoU      
            avg_mIoU_s.update(mean_IoU_smooth)
            avg_IoU_1_s.update(IoU_array_smooth[1])                        # IoU of class 1
            
            # p_mIoU
            p_mIoU = 0.5 * (FN / np.maximum(1.0, FN + TP + TN)) + 0.5 * (FP / np.maximum(1.0, FP + TP + TN))
            avg_p_mIoU.update(np.maximum(mean_IoU, p_mIoU))
            
            # p_mIoU_smooth
            p_mIoU_smooth = 0.5 * ((FN + smooth)/ (FN + TP + TN + smooth)) + 0.5 * ((FP + smooth)/ (FP + TP + TN + smooth))
            avg_p_mIoU_s.update(np.maximum(mean_IoU_smooth, p_mIoU_smooth))

            # p_F1
            F1   = 2 * TP / np.maximum(2 * TP + FN + FP, 1.0)
            p_F1 = 2 * FN / np.maximum(2 * TP + FN + TN, 1.0)
            avg_p_F1.update(np.maximum(F1, p_F1))

            # p_F1_smooth
            F1_s   = (2 * TP + smooth) / (2 * TP + FN + FP + smooth)
            p_F1_s = (2 * FN + smooth) / (2 * TP + FN + TN + smooth)
            avg_p_F1_s.update(np.maximum(F1_s, p_F1_s))


            # CONF metrics
            check_conf = 'conf_head' in config.MODEL.EXTRA.MODULES
            if check_conf:
                current_confusion_matrix_CONF = get_confusion_matrix_1ch(
                    tcp > 0.5,
                    conf,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL)
                val_mse = torch.mean((torch.sigmoid(conf[:, 0]) - tcp) ** 2).item()
                c_avg_mse.update(val_mse)
                confusion_matrix_CONF += current_confusion_matrix_CONF

                c_pos = current_confusion_matrix_CONF.sum(1)  # ground truth label count
                c_res = current_confusion_matrix_CONF.sum(0)  # prediction count
                c_tp = np.diag(current_confusion_matrix_CONF)  # Intersection part

                # mIoU (conf)
                c_IoU_array = (c_tp / np.maximum(1.0, c_pos + c_res - c_tp))  # Union part
                c_mean_IoU = c_IoU_array.mean()  # mean of the classes IoU
                c_avg_mIoU.update(c_mean_IoU)

                # mIoU_s (conf)
                c_IoU_array_smooth = ((c_tp + smooth) / (c_pos + c_res - c_tp + smooth))  # Union part
                c_mean_IoU_smooth = c_IoU_array_smooth.mean()  # mean of the classes IoU
                c_avg_mIoU_s.update(c_mean_IoU_smooth)


            # DET metrics
            if det is not None:
                det = det[:,0].cpu().numpy()
            else:
                det = np.max(pred[:,1].cpu().numpy(), axis=(1,2))
            target_det = torch.count_nonzero(label * (label >= 0), (-1, -2)) > 3
            target_det = target_det.cpu().numpy()
            if np.any(target_det>0.5):
                avg_det_tpr.update(np.sum(det[target_det>0.5]>0), np.sum(target_det>0.5))
            if np.any(target_det<0.5):
                avg_det_tnr.update(np.sum(det[target_det<0.5]<0), np.sum(target_det<0.5))


    confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
    confusion_matrix = confusion_matrix.cpu().numpy()
    
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum() / pos.sum()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    IoU_array_smooth = ((tp + smooth)/ (pos + res - tp + smooth))  # Union part
    mean_IoU_smooth = IoU_array_smooth.mean()                      # mean of the classes IoU      
    
    print_loss = avg_loss.average()
    try:
        bacc = (avg_det_tpr.average()+avg_det_tnr.average())/2
    except:
        bacc = 0
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']

    metric_dict = {
        'loss'              : print_loss,
        'mIoU'              : mean_IoU,
        'mIoU_smooth'       : mean_IoU_smooth,
        'avg_mIoU'          : avg_mIoU.average(),
        'avg_mIoU_smooth'   : avg_mIoU_s.average(),
        'avg_det_tpr'       : avg_det_tpr.average(),
        'avg_det_tnr'       : avg_det_tnr.average(),
        'avg_det_bacc'      : bacc,
        'avg_mse'           : avg_mse.average(),
        'avg_IoU_1_smooth'  : avg_IoU_1_s.average(),
        'avg_p-mIoU'        : avg_p_mIoU.average(),
        'avg_p-mIoU_smooth' : avg_p_mIoU_s.average(),
        'avg_p-F1'          : avg_p_F1.average(),
        'avg_p-F1_smooth'   : avg_p_F1_s.average(),
        'pixel_acc'         : pixel_acc,
    }

    if check_conf:
        metric_dict['avg_mse_CONF']         = c_avg_mse.average()
        metric_dict['avg_mIoU_CONF']        = c_avg_mIoU.average()
        metric_dict['avg_mIoU_smooth_CONF'] = c_avg_mIoU_s.average()

    for metric in metric_dict:
        writer.add_scalar(valid_set + '_' + metric, metric_dict[metric], global_steps)

    writer_dict['valid_global_steps'] = global_steps + 1

    return metric_dict, IoU_array, confusion_matrix




