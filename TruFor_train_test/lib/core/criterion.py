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



class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)

    def forward(self, score, target):        
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)
        return loss

    
    
class DiceLoss(nn.Module):
    def __init__(self, ignore_label=-1, smooth=1, exponent=2): #because padding adds -1 to the targets
        super(DiceLoss, self).__init__()  
        self.ignore_index = ignore_label
        self.smooth = smooth
        self.exponent = exponent
        
    def dice_loss(self, pred, target, valid_mask):
        assert pred.shape[0] == target.shape[0]
        total_loss = 0
        num_classes = pred.shape[1]
        for i in range(num_classes):
            if i != self.ignore_index:
                dice_loss = self.binary_dice_loss(
                    pred[:, i],
                    target[..., i],
                    valid_mask=valid_mask,)
                total_loss += dice_loss
        return total_loss / num_classes

    def binary_dice_loss(self, pred, target, valid_mask):
        assert pred.shape[0] == target.shape[0]
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

        num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + self.smooth
        den = torch.sum(pred.pow(self.exponent)*valid_mask + target.pow(self.exponent)*valid_mask, dim=1) + max(self.smooth, 1e-5)
        
        dice = num / den
        dice = torch.mean(dice)
        return 1 - dice
        
    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')
        
        score = F.softmax(score,dim=1)
        num_classes = score.shape[1]
        
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()
        
        loss = self.dice_loss(score, one_hot_target, valid_mask)
        return loss
    
    

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, ignore_label=-1): #because padding adds -1 to the targets
        super(BinaryDiceLoss, self).__init__()  
        self.ignore_index = ignore_label
        self.smooth = smooth
        self.exponent = exponent

    def binary_dice_loss(self, pred, target, valid_mask):
        assert pred.shape[0] == target.shape[0]
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

        num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + self.smooth
        den = torch.sum(pred.pow(self.exponent)*valid_mask + target.pow(self.exponent)*valid_mask, dim=1) + max(self.smooth, 1e-5)
        
        dice = num / den
        dice = torch.mean(dice)
        return 1 - dice
        
    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')
        
        score = F.softmax(score,dim=1)
        num_classes = score.shape[1]
        
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()
        
        loss = self.binary_dice_loss(
                    score[:, 1],
                    one_hot_target[..., 1],
                    valid_mask)
        return loss
    

class DiceEntropyLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, ignore_label=-1, weight=None): #because padding adds -1 to the targets
        super(DiceEntropyLoss, self).__init__()  
        self.ignore_label = ignore_label
        self.smooth = smooth
        self.exponent = exponent
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)
    
    def binary_dice_loss(self, pred, target, valid_mask):
        assert pred.shape[0] == target.shape[0]
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

        num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + self.smooth
        den = torch.sum(pred.pow(self.exponent)*valid_mask + target.pow(self.exponent)*valid_mask, dim=1) + max(self.smooth, 1e-5)
        
        dice = num / den
        dice = torch.mean(dice)
        return 1 - dice
        
    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')
        
        CE_loss   = self.cross_entropy(score, target)
        
        
        score = F.softmax(score,dim=1)
        num_classes = score.shape[1]
        
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_label).long()
        
        dice_loss = self.binary_dice_loss(
                    score[:, 1],
                    one_hot_target[..., 1],
                    valid_mask)

        return 0.3*CE_loss + 0.7*dice_loss



    
class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2., ignore_label=-1):  #alpha 0.25, gamma=2.
        super(FocalLoss, self).__init__()
        self.alpha=alpha
        self.gamma= gamma
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction="none")
       
    def forward(self, score, target):  
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')
            
        ce_loss = self.criterion(score, target)
        pt = torch.exp(-ce_loss)
        f_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return f_loss.mean()
        
        
