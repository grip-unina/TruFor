import os
import numpy as np



def extractGTs(gt, erodeKernSize=15, dilateKernSize=11):
    from scipy.ndimage.filters import minimum_filter, maximum_filter
    gt1 = minimum_filter(gt, erodeKernSize)
    gt0 = np.logical_not(maximum_filter(gt, dilateKernSize))
    return gt0, gt1


def computeMetricsContinue(values, gt0, gt1):
    values = values.flatten().astype(np.float32)
    gt0 = gt0.flatten().astype(np.float32)
    gt1 = gt1.flatten().astype(np.float32)
    
    inds = np.argsort(values) 
    inds = inds[(gt0[inds]+gt1[inds])>0]
    vet_th = values[inds]
    gt0 = gt0[inds]
    gt1 = gt1[inds]
        
    TN = np.cumsum(gt0)
    FN = np.cumsum(gt1)
    FP = np.sum(gt0) - TN
    TP = np.sum(gt1) - FN
    
    msk = np.pad(vet_th[1:]>vet_th[:-1], (0,1), mode='constant', constant_values=True)
    FP = FP[msk]
    TP = TP[msk]
    FN = FN[msk]
    TN = TN[msk]
    vet_th = vet_th[msk]
    
    return FP, TP, FN, TN, vet_th


def computeMetrics_th(values, gt, gt0, gt1, th):    
    values = values>th
    values = values.flatten().astype(np.uint8)
    gt  = gt.flatten().astype(np.uint8)
    gt0 = gt0.flatten().astype(np.uint8)
    gt1 = gt1.flatten().astype(np.uint8)
    
    gt     = gt[(gt0+gt1)>0]
    values = values[(gt0+gt1)>0]

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(gt, values)
    
    TN = cm[0, 0]
    FN = cm[1, 0]
    FP = cm[0, 1]
    TP = cm[1, 1]
    
    return FP, TP, FN, TN


def computeMCC(FP, TP, FN, TN):
    FP = np.float64(FP)
    TP = np.float64(TP)
    FN = np.float64(FN)
    TN = np.float64(TN)
    return np.abs(TP*TN - FP*FN) / np.maximum(np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN) ), 1e-32)


def computeF1(FP, TP, FN, TN):
    return 2*TP / np.maximum((2*TP + FN + FP), 1e-32)



def computeLocalizationMetrics(map, gt):
    gt0, gt1 = extractGTs(gt)
    
    # best threshold
    try:
        FP, TP, FN, TN, _  = computeMetricsContinue(map, gt0, gt1)
        f1  = computeF1(FP, TP, FN, TN)
        f1i = computeF1(TN, FN, TP, FP)
        F1_best = max(np.max(f1), np.max(f1i))
    except:
        import traceback
        traceback.print_exc()
        F1_best = np.nan
    
    # fixed threshold
    try:
        FP, TP, FN, TN  = computeMetrics_th(map, gt, gt0, gt1, 0.5)
        f1  = computeF1(FP, TP, FN, TN)
        f1i = computeF1(TN, FN, TP, FP)
        F1_th = max(f1, f1i)
    except:
        import traceback
        traceback.print_exc()
        F1_th = np.nan
        
    return F1_best, F1_th
    
    
def computeDetectionMetrics(scores, labels):
    lbl = np.array(labels)
    lbl = lbl[np.isfinite(scores)]
    
    scores = np.array(scores, dtype='float32')
    scores[scores==np.PINF]  = np.nanmax(scores[scores<np.PINF])
    scores = scores[np.isfinite(scores)]
    assert lbl.shape == scores.shape

    # AUC
    from sklearn.metrics import roc_auc_score
    AUC = roc_auc_score(lbl, scores)

    # Balanced Accuracy
    from sklearn.metrics import balanced_accuracy_score
    bACC = balanced_accuracy_score(lbl, scores>0.5)
    
    return AUC, bACC
    



# ---------------------------------------------------------------------------- #
# DETECTION METRICS
# ---------------------------------------------------------------------------- #
'''
path = '/path-to-DSO1-dataset/'

from glob import glob
scores = []
labels = []

for map_path in glob(path + 'normal*'):
    scores.append(np.load(map_path)['score'])
    labels.append(0)
for map_path in glob(path + 'splicing*'):
    scores.append(np.load(map_path)['score'])
    labels.append(1)
    
print(computeDetectionMetrics(scores, labels))
'''


# ---------------------------------------------------------------------------- #
# LOCALIZATION METRICS
# ---------------------------------------------------------------------------- #
'''
path    = '/path-to-DSO1-dataset/'
gt_path = '/path-to-DSO1-masks/'

from glob import glob
from tqdm import tqdm
from PIL import Image
import os

F1_best_list = []
F1_th_list   = []
for map_path in tqdm(glob(path + 'splicing*')):
    map = np.load(map_path)['map']
    
    # gt can be <0.1 or >0.1 depending on the dataset. DSO-1 has inverted masks, so <
    gt  = np.array(Image.open(gt_path + os.path.basename(map_path[:-4])).convert('L')) < 0.1
    assert gt.shape == map.shape
    F1_best, F1_th = computeLocalizationMetrics(map, gt)
    F1_best_list.append(F1_best)
    F1_th_list.append(F1_th)
    
print('skipped:', np.count_nonzero(np.isnan(F1_best_list)))
print(np.nanmean(F1_best_list), np.nanmean(F1_th_list))
'''
