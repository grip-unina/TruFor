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
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, score, target):
        target_det = (torch.count_nonzero(target * (target >= 0), (-1, -2)) > 3).float().clamp(0, 1)
        weights_det = target_det * 0.5 / 0.7 + (1 - target_det) * 0.5 / 0.3
        loss_det = F.binary_cross_entropy_with_logits(score[:, 0], target_det, reduction='mean', weight=weights_det)
        return loss_det
