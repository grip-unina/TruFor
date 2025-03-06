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

import torch
import torch.nn as nn
from torch.nn import functional as F



class MSE(nn.Module):
    def __init__(self, ignore_label=-1, criterion='mse'):
        super(MSE, self).__init__()
        self.ignore_label = ignore_label
        if criterion=='mse':
            self.criterion = nn.MSELoss()
        else:
            assert False
    
    def calcolaGTs(self, gt, erodeKernSize=15, dilateKernSize=11):
        from torch.nn.functional import max_pool2d
        gt1 = 1 - max_pool2d(1-gt[:,None,:,:], erodeKernSize, stride=1, padding=(erodeKernSize-1)//2)[:,0]
        gt0 = 1 - max_pool2d(gt[:,None,:,:], dilateKernSize, stride=1, padding=(dilateKernSize-1)//2)[:,0]
        return gt0, gt1


    def forward(self, pred, target, conf):
        # conf: confidence prediction (1 channel)
        # pred: 2 channels cmx prediction
        ch, cw = conf.size(2), conf.size(3)
        ph, pw = pred.size(2), pred.size(3)
        h, w = target.size(1), target.size(2)
        
        if ph != h or pw != w:
            pred = F.upsample(input=pred, size=(h, w), mode='bilinear')
        if ch != h or cw != w:
            conf = F.upsample(input=conf, size=(h, w), mode='bilinear')
        
        conf = torch.sigmoid(conf)
        pred = F.softmax(pred, dim=1)
        
        target0, target1 = self.calcolaGTs((target==1).float())
        conf = conf.squeeze(dim=1)
        tcp  = pred[:,1]*target1 + pred[:,0]*target0
        
        assert conf.shape == tcp.shape
        
        valid = torch.logical_and(target!=self.ignore_label, torch.logical_or(target1>0, target0>0))
                
        conf = conf[valid]
        tcp  = tcp[valid]
        loss = self.criterion(conf, tcp)
        return loss
    




