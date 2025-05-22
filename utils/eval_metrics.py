from torch.nn.functional import one_hot
from numbers import Number
import numpy as np
import torch


def calculate_confusion_matrix(pred, target):
    """
    计算混淆矩阵

    Parameters:
        pred (torch.Tensor | np.array):   模型预测结果, 大小为 (N, C)
        target (torch.Tensor | np.array): 每个预测结果的目标标签, 大小为 (N, 1) or (N,)

    Returns:
        torch.Tensor: 混淆矩阵, 大小为 (C, C), C为类别数目
    """

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    assert (
        isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor)), \
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    # Modified from PyTorch-Ignite
    num_classes = pred.size(1)
    pred_label = torch.argmax(pred, dim=1).flatten()
    target_label = target.flatten()
    assert len(pred_label) == len(target_label)

    with torch.no_grad():
        indices = num_classes * target_label + pred_label
        matrix = torch.bincount(indices, minlength=num_classes**2)
        matrix = matrix.reshape(num_classes, num_classes)
    return matrix.detach().cpu().numpy()


def calculate_precision_recall_f1(pred, target, average_mode='macro', thrs=0.):
    """
    计算precision, recall和f1_score

    Parameters:
        pred (torch.Tensor | np.array):   模型预测结果, 大小为 (N, C).
        target (torch.Tensor | np.array): 每个预测结果的目标标签, 大小为 (N, 1) or (N,).        
        average_mode (str): 结果如何在多类之间进行平均
            'none':  返回每个类别的指标
            'macro': 计算每个类别的指标, 然后计算不加权平均作为最后结果
            默认: 'macro'
        thrs (Number | tuple[Number], optional): 阈值, 预测结果在阈值之下的认为负
            默认: 0.

    Returns:
        tuple: 元组, 包含precision, recall和f1 score

        precision, recall和f1 score可能的组织形式如下:
        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """

    allowed_average_mode = ['macro', 'none']
    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    assert isinstance(pred, torch.Tensor), \
        (f'pred should be torch.Tensor or np.ndarray, but got {type(pred)}.')
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).long()
    assert isinstance(target, torch.Tensor), \
        f'target should be torch.Tensor or np.ndarray, ' \
        f'but got {type(target)}.'

    if isinstance(thrs, Number):
        thrs = (thrs, )
        return_single = True
    elif isinstance(thrs, tuple):
        return_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    num_classes = pred.size(1) # size 0为图片数量，size 1 为类别数量
    pred_score, pred_label = torch.topk(pred, k=1)
    pred_score = pred_score.flatten()
    pred_label = pred_label.flatten()

    gt_positive = one_hot(target.flatten(), num_classes)

    precisions = []
    recalls = []
    f1_scores = []
    for thr in thrs:
        # Only prediction values larger than thr are counted as positive
        pred_positive = one_hot(pred_label, num_classes)
        if thr is not None:
            pred_positive[pred_score <= thr] = 0
        class_correct = (pred_positive & gt_positive).sum(0).detach().cpu().numpy()
        precision = class_correct / np.maximum(pred_positive.sum(0).detach().cpu().numpy(), 1.) * 100
        recall = class_correct / np.maximum(gt_positive.sum(0).detach().cpu().numpy(), 1.) * 100
        f1_score = 2 * precision * recall / np.maximum(
            precision + recall,
            torch.finfo(torch.float32).eps)
        if average_mode == 'macro':
            precision = float(precision.mean())
            recall = float(recall.mean())
            f1_score = float(f1_score.mean())
        elif average_mode == 'none':
            precision = precision
            recall = recall
            f1_score = f1_score
        else:
            raise ValueError(f'Unsupport type of averaging {average_mode}.')
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = np.array(f1_scores)

    if return_single:
        return precisions[0].tolist(), recalls[0].tolist(), f1_scores[0].tolist()
    else:
        return precisions, recalls, f1_scores
