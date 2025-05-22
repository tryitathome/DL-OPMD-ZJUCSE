import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np
import os
import time
import datetime
import pandas as pd
from tqdm import tqdm
from terminaltables import AsciiTable

from eval_metrics import calculate_confusion_matrix, calculate_precision_recall_f1
from train_vis import print_confusion_matrix

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def test(test_process_name, test_loader, model, checkpoints_path, device, average_mode, thrs, description):

    print(f'Start Test...')
    weights = torch.load(checkpoints_path, map_location='cpu')
    # new_state_dict = {k: v for k, v in pretrained_weights.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
    # model.load_state_dict(pretrained_weights, strict=True)
    # msg = model.load_state_dict(pretrained_weights['model'], strict=False)
    model.load_state_dict(weights['model_state_dict'], strict=True)
    print(f'Best model weights loaded from {checkpoints_path}')

    model.to(device)

    # 计算参数量和FLOPS
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_parameters / 1e7}M.')
    '''
    flops = model.flops()
    print(f'Number of FLOPS:      {flops / 1e9}G.')
    '''

    # 创建保存训练结果的目录
    os.makedirs(f'test_results/{test_process_name}', exist_ok=True)

    # 开始测试
    print(f'Using {device}')
    print()
    # mixup_function = None
    test_start_time = time.time()
 
    model.eval()

    correct = 0
    total = 0
    pred_list, target_list = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=description):
            images = images.to(device) 
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            pred_list.append(outputs.data)
            target_list.append(labels)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total

    # 计算其他指标
    precision, recall, f1_score = calculate_precision_recall_f1(torch.cat(pred_list), 
                                                                torch.cat(target_list), 
                                                                average_mode=average_mode,
                                                                thrs=thrs)

    TITLE = 'Validation Results'
    TABLE_DATA = (
        ('Top-1 Acc', 'Mean Precision', 'Mean Recall', 'Mean F1 Score'),
        ('{:.2f}'.format(accuracy), 
                         '{:.2f}'.format(precision), 
                         '{:.2f}'.format(recall),
                         '{:.2f}'.format(f1_score)
        )
    )
    table_instance = AsciiTable(TABLE_DATA, TITLE)
    print(table_instance.table)

    # 计算混淆矩阵
    confusion_matrix_data = calculate_confusion_matrix(torch.cat(pred_list), torch.cat(target_list))
    print_confusion_matrix(confusion_matrix_data)

    # 同步
    torch.cuda.synchronize()

    results = {
        "accuracy":  [accuracy],
        "precision": [precision],
        "recall":    [recall],
        "f1_score":  [f1_score]
    }

    df = pd.DataFrame(results)

    # 将测试结果写入Excel文件
    file_path = f'test_results/{test_process_name}/test_results.xlsx'
    df.to_excel(file_path, index=False)
    print(f"Test Results saved to {file_path}")
    
    test_total_time = time.time() - test_start_time
    total_time_str = str(datetime.timedelta(seconds=int(test_total_time)))
    print()
    print(f'Whole Test Time: {total_time_str}')