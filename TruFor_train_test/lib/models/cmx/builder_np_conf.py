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
Edited in September 2022
@author: fabrizio.guillaro, davide.cozzolino
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from lib.models.cmx.init_func import init_weight

import logging


def preprc_imagenet_torch(x):
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(x.device)
    std  = torch.Tensor([0.229, 0.224, 0.225]).to(x.device)
    x = (x-mean[None, :, None, None]) / std[None, :, None, None]
    return x

def preprc_xception_torch(x):
    return 2.0*x-1.0


def create_backbone(typ, norm_layer):
    channels = [64, 128, 320, 512]
    if typ == 'mit_b5':
        logging.info('Using backbone: Segformer-B5')
        from .encoders.dual_segformer import mit_b5 as backbone_
        backbone = backbone_(norm_fuse=norm_layer)
    elif typ == 'mit_b4':
        logging.info('Using backbone: Segformer-B4')
        from .encoders.dual_segformer import mit_b4 as backbone_
        backbone = backbone_(norm_fuse=norm_layer)
    elif typ == 'mit_b2':
        logging.info('Using backbone: Segformer-B2')
        from .encoders.dual_segformer import mit_b2 as backbone_
        backbone = backbone_(norm_fuse=norm_layer)
    elif typ == 'mit_b1':
        logging.info('Using backbone: Segformer-B1')
        from .encoders.dual_segformer import mit_b1 as backbone_
        backbone = backbone_(norm_fuse=norm_layer)
    elif typ == 'mit_b0':
        logging.info('Using backbone: Segformer-B0')
        channels = [32, 64, 160, 256]
        from .encoders.dual_segformer import mit_b0 as backbone_
        backbone = backbone_(norm_fuse=norm_layer)
    else:
        raise NotImplementedError('Backbone not implemented')
    return backbone, channels




class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        
        self.norm_layer = norm_layer
        self.cfg  = cfg.MODEL.EXTRA
        self.mods = cfg.MODEL.MODS    # input modalities

        # setting number of Noiseprint++ output channels
        if 'NP_OUT_CHANNELS' in self.cfg:
            self.np_out_ch = self.cfg.NP_OUT_CHANNELS
        else:
            self.np_out_ch = 1

        modules_list = ['NP++','backbone','loc_head','conf_head','det_head']
        for module in self.cfg.MODULES:
            assert module in modules_list
        assert 'backbone' in self.cfg.MODULES

        for module in self.cfg.FIX_MODULES:
            assert module in modules_list

        # importing backbone
        self.backbone, self.channels = create_backbone(self.cfg.BACKBONE, norm_layer)

        # defining heads
        self.decode_head      = None   # localization head
        self.decode_head_conf = None   # confidence head
        self.detection        = None   # detection head


        if self.cfg.DECODER == 'MLPDecoder':
            logging.info('Using MLP Decoder')
            from .decoders.MLPDecoder import DecoderHead

            # localization head
            if 'loc_head' in self.cfg.MODULES:
                self.decode_head = DecoderHead(
                    in_channels=self.channels,
                    num_classes=cfg.DATASET.NUM_CLASSES,
                    norm_layer=norm_layer,
                    embed_dim=self.cfg.DECODER_EMBED_DIM)

            # confidence head
            if 'conf_head' in self.cfg.MODULES:
                self.decode_head_conf = DecoderHead(
                    in_channels=self.channels,
                    num_classes=1,
                    norm_layer=norm_layer,
                    embed_dim=self.cfg.DECODER_EMBED_DIM)

            # detection head
            self.conf_detection = self.cfg.DETECTION
            if 'det_head' in self.cfg.MODULES:
                if self.conf_detection == 'confpool':
                    assert 'conf_head' in self.cfg.MODULES
                    self.detection  = nn.Sequential(
                            nn.Linear(in_features=8, out_features=128),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(in_features=128, out_features=1),
                            )
                else:
                    raise NotImplementedError('Detection mechanism not implemented')

        else:
            raise NotImplementedError('Decoder not implemented')

        # Noiseprint++ extractor
        from lib.models.DnCNN import make_net
        num_levels = 17
        out_channel = self.np_out_ch
        self.dncnn = make_net(3, kernels=[3, ] * num_levels,
                       features=[64, ] * (num_levels - 1) + [out_channel],
                       bns=[False, ] + [True, ] * (num_levels - 2) + [False, ],
                       acts=['relu', ] * (num_levels - 1) + ['linear', ],
                       dilats=[1, ] * num_levels,
                       bn_momentum=0.1, padding=1)
        
        if self.cfg.PREPRC is None or self.cfg.PREPRC == 'none': #RGB01 (0,1)
            self.prepro = None
        elif self.cfg.PREPRC == 'imagenet': #RGB (mean and variance)
            self.prepro = preprc_imagenet_torch
        elif self.cfg.PREPRC == 'xception': #RGB0 (-1,1)
            self.prepro = preprc_xception_torch
        else:
            assert False

        # pretraining
        self.init_weights(pretrained=cfg.MODEL.PRETRAINED)

        
    
    def init_weights(self, pretrained=None):

        # loading Noiseprint++ weights
        if 'NP_WEIGHTS' in self.cfg and not self.cfg.NP_WEIGHTS == '' and self.cfg.NP_WEIGHTS is not None:
            np_weights = self.cfg.NP_WEIGHTS
            assert os.path.isfile(np_weights)
            dat = torch.load(np_weights, map_location=torch.device('cpu'))['network']
            logging.info(f'Noiseprint++ weights: {np_weights}')
            self.dncnn.load_state_dict(dat)

        # backbone pretraining
        if pretrained:
            logging.info('Loading backbone model: {}'.format(pretrained))
            assert os.path.isfile(pretrained)
            self.backbone.init_weights(pretrained=pretrained)

        # initing heads weights
        logging.info('Initing heads weights ...')
        if self.decode_head:
            init_weight(self.decode_head, nn.init.kaiming_normal_,
                        self.norm_layer, self.cfg.BN_EPS, self.cfg.BN_MOMENTUM,
                        mode='fan_in', nonlinearity='relu')

        if self.decode_head_conf:
            init_weight(self.decode_head_conf, nn.init.kaiming_normal_,
                        self.norm_layer, self.cfg.BN_EPS, self.cfg.BN_MOMENTUM,
                        mode='fan_in', nonlinearity='relu')
            

        # freezing modules
        if 'NP++' in self.cfg.FIX_MODULES:
            for param in self.dncnn.parameters():
                param.requires_grad = False

        if 'backbone' in self.cfg.FIX_MODULES:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if 'loc_head' in self.cfg.FIX_MODULES:
            for param in self.decode_head.parameters():
                param.requires_grad = False

        if 'conf_head' in self.cfg.FIX_MODULES:
            for param in self.decode_head_conf.parameters():
                param.requires_grad = False



    def encode_decode(self, rgb, modal_x):

        if rgb is not None:
            orisize = rgb.shape
        else:
            orisize = modal_x.shape
        
        # CMX encoder
        if 'backbone' in self.cfg.FIX_MODULES:
            with torch.no_grad():
                self.backbone.eval()
                x = self.backbone(rgb, modal_x)
        else:            
            x = self.backbone(rgb, modal_x)


        # anomaly localization
        if 'loc_head' in self.cfg.FIX_MODULES:
            with torch.no_grad():
                self.decode_head.eval()
                out = self.decode_head(x)
        else:
            out = self.decode_head(x)

        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)


        # confidence estimation
        if self.decode_head_conf:
            if 'conf_head' in self.cfg.FIX_MODULES:
                with torch.no_grad():
                    self.decode_head_conf.eval()
                    conf = self.decode_head_conf(x)
            else:
                conf = self.decode_head_conf(x)
            conf = F.interpolate(conf, size=orisize[2:], mode='bilinear', align_corners=False)
        else: 
            conf = None

        
        # detection
        if self.detection:
            if self.conf_detection == 'confpool':
                from .layer_utils import weighted_statistics_pooling
                f1 = weighted_statistics_pooling(conf).view(out.shape[0],-1)
                f2 = weighted_statistics_pooling(out[:,1:2,:,:]-out[:,0:1,:,:], F.logsigmoid(conf)).view(out.shape[0],-1)
                det = self.detection(torch.cat((f1,f2),-1))
            else:
                assert False
        else:
            det = None
        
        return out, conf, det



    def forward(self, rgb, save_np=False):
        # rgb should be a float tensor in the range [0,1], since Noiseprint++ has been trained with this input

        # Noiseprint++ extraction
        if 'NP++' in self.mods:
            if 'NP++' in self.cfg.FIX_MODULES:
                with torch.no_grad():
                    self.dncnn.eval()
                    modal_x = self.dncnn(rgb)
            else:
                modal_x = self.dncnn(rgb)

            if self.np_out_ch == 1:
                modal_x = torch.tile(modal_x, (3, 1, 1))
            else:
                assert self.np_out_ch == 3
        else:
            modal_x = None


        if 'RGB' not in self.mods:
            rgb = None
        # from [0,1] to other normalization, before going in the CMX network
        elif self.prepro is not None:
            rgb = self.prepro(rgb)

        # Localization and Detection
        out, conf, det = self.encode_decode(rgb, modal_x)

        if save_np:
            return out, conf, det, modal_x
        else:
            return out, conf, det, None
            