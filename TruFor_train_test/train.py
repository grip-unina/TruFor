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

import sys, os
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

import argparse

import logging
import time
import timeit

import gc
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim
torch.autograd.set_detect_anomaly(True)
from tensorboardX import SummaryWriter

from lib.config import config, update_config
from lib.core.function import train, validate
from lib.utils import get_model, get_optimizer
from lib.utils import create_logger, FullModel, adjust_learning_rate

from dataset.data_core import myDataset
import albumentations


def main():
    parser = argparse.ArgumentParser(description='Train TruFor')
    parser.add_argument('-exp', '--experiment', type=str)
    parser.add_argument('-g',   '--gpu', type=int, default=[0], nargs="+", help='device(s)')
    parser.add_argument('opts', help='other options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)
    args.gpu = range(len(args.gpu))

    update_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(config, f'{args.experiment}', 'train')
    logger.info(config)
    logger.info('\n')

    # cudnn setting
    cudnn.benchmark     = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled       = config.CUDNN.ENABLED

    gpus = list(config.GPUS)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    if config.TRAIN.AUG is not None:
        aug_train = albumentations.load(config.TRAIN.AUG, data_format='yaml')
    else:
        aug_train = None

    if config.VALID.AUG is not None:
        aug_valid = albumentations.load(config.VALID.AUG, data_format='yaml')
    else:
        aug_valid = None

    logger.info(f'Train augmentation: {config.TRAIN.AUG} {aug_train}')
    logger.info(f'Validation augmentation: {config.VALID.AUG} {aug_valid}')

    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = myDataset(config, crop_size=crop_size, grid_crop=False, mode='train', aug=aug_train)
    valid_dataset = myDataset(config, crop_size=None, grid_crop=False, mode="valid", aug=aug_valid,
                              max_dim=config.VALID.MAX_SIZE)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size  = config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle     = config.TRAIN.SHUFFLE,
        num_workers = config.WORKERS)

    validloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size  = 1,      # 1 to allow arbitrary input sizes
        shuffle     = False,  # must be False to get accurate filename
        num_workers = config.WORKERS)

    # model
    model = get_model(config)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    model = FullModel(model, config)

    # optimizer
    optimizer = get_optimizer(model, config)

    epoch_iters = np.int32(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))

    best_key = config.VALID.BEST_KEY
    if 'loss' in best_key:
        best_value = np.inf
    else:
        best_value = 0
    logger.info(f'best valid key: {best_key}')


    last_epoch = 0
    if not config.TRAIN.PRETRAINING == '' and not config.TRAIN.PRETRAINING == None:
        model_state_file = config.TRAIN.PRETRAINING
        assert os.path.isfile(model_state_file)
        checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
        state_dict = checkpoint['state_dict']
        try:
            model.model.module.load_state_dict(state_dict, strict=False)
        except:
            state_dict = {k: state_dict[k] for k in state_dict if not k.startswith('detection')}
            model.model.module.load_state_dict(state_dict, strict=False)
        del checkpoint
        del state_dict
        logger.info("=> loaded pretraining ({})".format(model_state_file))

        
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
            best_value = checkpoint['best_value']
            assert checkpoint['best_key']==best_key
            last_epoch = checkpoint['epoch']
            model.model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
            writer_dict['train_global_steps'] = last_epoch
        else:
            logger.info("No previous checkpoint.")


    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    start_epoch = last_epoch
    if config.VALID.FIRST_VALID:
        start_epoch = start_epoch -1

    for epoch in range(start_epoch, end_epoch):
        # train
        if epoch>=last_epoch:
            train_dataset.shuffle()  # for class-balanced sampling

            print(f'TRAINING epoch {epoch}:')
            train(epoch, config.TRAIN.END_EPOCH,
                  epoch_iters, config.TRAIN.LR, num_iters,
                  trainloader, optimizer, model, writer_dict,
                  adjust_learning_rate=adjust_learning_rate)

            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(1.0)
            
            logger.info('=> saving checkpoint to {}'.format(
                os.path.join(final_output_dir, 'checkpoint.pth.tar')))
            torch.save({
                'epoch': epoch + 1,
                'best_value': best_value,
                'best_key': best_key,
                'state_dict': model.model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))


        # valid
        print(f'VALIDATION epoch {epoch}:')
        writer_dict['valid_global_steps'] = epoch

        value_valid, IoU_array, confusion_matrix = \
            validate(config, validloader, model, writer_dict, "valid")

        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(3.0)

        if 'loss' in best_key:
            if value_valid[best_key] < best_value:  # smallest loss
                best_value = value_valid[best_key]
                torch.save({
                    'epoch': epoch + 1,
                    'best_value': best_value,
                    'best_key': best_key,
                    'state_dict': model.model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(final_output_dir, 'best.pth.tar'))
                logger.info("best.pth.tar updated.")

        elif value_valid[best_key] > best_value:  # highest metric
            best_value = value_valid[best_key]
            torch.save({
                'epoch': epoch + 1,
                'best_value': best_value,
                'best_key': best_key,
                'state_dict': model.model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir, 'best.pth.tar'))
            logger.info("best.pth.tar updated.")

        msg = '(Valid) Loss: {:.3f}, Best_{:s}: {: 4.4f}'.format(
            value_valid['loss'], best_key, best_value)
        logging.info(msg)
        logging.info(IoU_array)
        logging.info("confusion_matrix:")
        logging.info(confusion_matrix)




if __name__ == '__main__':
    main()
