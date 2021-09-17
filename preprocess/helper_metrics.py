#
#
#      0===============================0
#      |      PLY evaluate metrics     |
#      0===============================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      function to evaluate .ply segmentation results
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Zhenhong Zou - 08/30/2021
#
# ----------------------------------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#

from typing import Iterator, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


def mIoU(pred, gt, class_num=13) -> Tuple[float, list, list]:
    """calc the mean IoU of pred based on gt
    
    Args::
        pred: shape of (num,) or (num, 3), int
        gt: same shape of pred, int
        num: num of pts, int
    Returns: mIoU score, iou list for each class
    """

    gt_classes = [0 for _ in range(class_num)]
    positive_classes = [0 for _ in range(class_num)]
    true_positive_classes = [0 for _ in range(class_num)]

    conf_matrix = confusion_matrix(gt, pred, labels=np.arange(0, class_num, 1))
    gt_classes += np.sum(conf_matrix, axis=1)
    positive_classes += np.sum(conf_matrix, axis=0)
    true_positive_classes += np.diagonal(conf_matrix)

    iou_list = []
    gt_list = []
    for n in range(0, class_num, 1):
        iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n] + 0.1)
        iou_list.append(iou)
        gt_list.append(gt_classes[n])
    mean_iou = sum(iou_list) / float(class_num)

    return mean_iou, iou_list, gt_list


def batch_mIoU(iter_data: Iterator[Tuple[np.array, np.array]]):
    """calc the mean IoU of pred based on gt
    
    Args::
        pred: shape of (num,) or (num, 3), int
        gt: same shape of pred, int
        num: num of pts, int
    Returns: mIoU score, iou list for each class
    """

    gt_classes = [0 for _ in range(13)]
    positive_classes = [0 for _ in range(13)]
    true_positive_classes = [0 for _ in range(13)]
    iou_list = []
    gt_list = []
    macc = []
    tp = 0
    all = 0

    for preds, gt in tqdm(iter_data):
        pred, OAcc, pts_num = preds
        tp += OAcc
        all += pts_num
        conf_matrix = confusion_matrix(gt, pred, labels=np.arange(0, 13, 1))
        gt_classes += np.sum(conf_matrix, axis=1)
        positive_classes += np.sum(conf_matrix, axis=0)
        true_positive_classes += np.diagonal(conf_matrix)

    for n in range(0, 13, 1):
        iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n] + 0.1)
        iou_list.append(iou)
        gt_list.append(gt_classes[n])
        macc.append(true_positive_classes[n]/gt_classes[n])
    mean_iou = sum(iou_list) / float(13)

    return mean_iou, iou_list, gt_list, tp/all, macc