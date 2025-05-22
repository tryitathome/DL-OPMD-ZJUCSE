import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import os
import time
import datetime
import pandas as pd
from tqdm import tqdm
from PIL import Image
from terminaltables import AsciiTable

import sys
sys.path.insert(0, os.getcwd())
from utils.train_process import calculate_precision_recall_f1, calculate_confusion_matrix, print_confusion_matrix

# 导入自定义模型
from models.simple_cnn import SimpleCNN
from models.resnet18 import ResNet18
from models.resnet50 import ResNet50
from models.vit import ViT
from models.densenet import DenseNet121

from build_model.build_swin import build_swin_transformer, build_swin_transformer_v2
from build_model.build_convnext import build_convnext
from build_model.build_convnextv2 import build_convnextv2
from build_model.build_replknet import build_replknet
from build_model.build_metaformer import caformer_b36_in21k, convformer_b36_in21ft1k, identityformer_m48, randformer_m48, poolformerv2_m48
from build_model.build_metaformer import caformer_b36_384_in21ft1k

from utils.load_data import build_loader_memmap, build_loader

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

def test(model, loader, device, criterion, epoch, history, average_mode, thrs, description, image_paths):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0
    correct = 0
    total = 0
    pred_list, target_list = [] , []
    all_preds = []
    all_labels = []
    all_image_paths = image_paths
    start_time = time.time()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=description):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            pred_list.append(outputs.data)
            target_list.append(labels)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    average_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    # 计算其他指标
    precision, recall, f1_score = calculate_precision_recall_f1(torch.cat(pred_list),
                                                                torch.cat(target_list),
                                                                average_mode=average_mode,
                                                                thrs=thrs)

    # 保存历史数据
    history['val_loss'].append(average_loss)
    history['val_acc'].append(accuracy)
    history['val_precision'].append(precision)
    history['val_recall'].append(recall)
    history['val_f1_score'].append(f1_score)

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
    print(f'Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.2f}%')

    # 同步
    torch.cuda.synchronize()

    # 打印验证时间
    end_time = time.time()
    val_time = end_time - start_time
    if epoch >= 0:
        print(f'Epoch {epoch+1} Validation Time: {val_time:.2f}')
    else:
        pass

    return accuracy, precision, recall, f1_score, all_preds, all_labels, all_image_paths

def get_image_paths(dataset_dir):
    image_paths = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.JPG') or file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.JPEG') or file.endswith('.png') or file.endswith('.PNG'):
                image_paths.append(os.path.join(root, file))
    return image_paths

def main():
    '''数据参数'''
    # 测试过程名字
    test_process_name = 'JDR_FinalRunGen1onJIUYUAN_20250420#2'

    # 数据集路径
    dataset_dir = r'Jiuyuan_test_80MARK'

    # 是否使用内存映射加速数据读取
    use_memmap  = False
    memmap_dir  = 'memmap_file'

    # 是否将数据集先存入内存, 严重过拟合
    load_into_memory = False

    # 模型是否为transformer类模型
    is_transformer = False

    # 是否加载预训练权重
    pretrained      = True
    pretrained_path = r"output\JDR_FinalRunGen1_20250407\checkpoints\best_model.pth"

    image_size  = 224
    batch_size  = 4
    num_workers = 4
    persistent_workers = True
    pin_memory         = True
    interpolation      = 'bicubic'    # 'random' or 'bilinear'
    # 使用memmap将num_workers设为0更快
    if use_memmap:
        num_workers        = 0
        persistent_workers = False
        pin_memory         = True

    '''模型参数'''
    num_classes     = 2

    '''增强参数'''
    color_jitter = 0.4
    auto_augment = 'rand-m9-mstd0.5-inc1'    # 'v0' or 'original'
    re_prob      = 0.25
    re_mode      = 'pixel'
    re_count     = 1

    '''测试参数'''
    # 先将图片进行resize放大, 再中心裁剪
    test_crop = False

    # 设置指标参数
    # average_mode = 'none'     # 输出为每个类的指标
    average_mode = 'macro'      # 输出为每个类的平均
    thrs = 0.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''创建模型'''
    # swin_name = 'swin_base_224'
    # model = build_swin_transformer(swin_name)

    # swin_v2_name = 'swin_v2_small_w8'
    # model = build_swin_transformer_v2(swin_v2_name)

    # convnext_name = 'convnext_s'
    # model = build_convnext(convnext_name)

    # replknet_name = 'replknet_xl'
    # model = build_replknet(replknet_name)

    # model = caformer_b36_384_in21ft1k(pretrained=False, num_classes=num_classes)
    
    from build_model.build_metaformer import caformer_s18
    from build_model.build_my_metaformer import caformer_s18_msf, caformer_s18_eva02, fulla_former_s18_eva02, caformer_b36_eva02
    # model = caformer_b36_384_in21ft1k(pretrained=False, num_classes=num_classes)
    model = build_convnextv2('convnextv2_tiny', num_classes)

    # model = metaformer.__dict__[args.model]()

    head_modify = False
    sys.path.insert(0, os.getcwd())
    from build_model.build_metaformer import caformer_b36_384_in21ft1k, poolformerv2_m48
    from build_model.build_my_metaformer import caformer_b36_384_in21ft1k_msf
    # model = caformer_b36_384_in21ft1k(pretrained=False, num_classes=num_classes)
    from build_model.build_metaformer import caformer_s18
    # model = caformer_s18(num_classes=num_classes)
    
    after_head = False

    # 修改分类头
    if after_head:
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, num_classes)

    if pretrained:
        pretrained_weights = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(pretrained_weights['model_state_dict'], strict=True)
        print('Pretrained weights loaded.')

    model.to(device)

    # 计算参数量和FLOPS
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_parameters / 1e7}M.')

    # 建立DataLoader
    if use_memmap:
        train_loader, val_loader, test_loader = build_loader_memmap(memmap_dir, dataset_dir, 
                                                                    image_size, batch_size,
                                                                    num_workers, pin_memory, persistent_workers, 
                                                                    color_jitter, auto_augment, re_prob, re_mode, re_count, interpolation, test_crop)
    else:
        train_loader, val_loader, test_loader = build_loader(dataset_dir, image_size, batch_size, 
                                                             load_into_memory, num_workers, pin_memory, persistent_workers, 
                                                             color_jitter, auto_augment, re_prob, re_mode, re_count, interpolation, test_crop)
    
    criterion = None

    # 准备收集训练历史数据
    history = {'train_loss': [], 'val_loss': [], 
               'train_acc':  [], 'val_acc': [], 
               'train_precision': [], 'val_precision': [],
               'train_recall':    [], 'val_recall':    [],
               'train_f1_score':  [], 'val_f1_score':  [], 
               'lr': []
    }

    # 创建保存训练结果的目录
    os.makedirs(f'test_results/{test_process_name}', exist_ok=True)

    # 获取所有测试图片的路径
    image_paths = get_image_paths(os.path.join(dataset_dir, 'test'))

    # 训练和验证模型
    print(f'Using {device}')
    print(f'Start Test...')
    print()
    full_train_start_time = time.time()
 
    # 在所有训练结束后，评估模型在测试集上的表现
    epoch = -1
    accuracy, precision, recall, f1_score, all_preds, all_labels, all_image_paths = test(model, test_loader, 
                                                                                         device, criterion, 
                                                                                         epoch, history, 
                                                                                         average_mode=average_mode, thrs=thrs,
                                                                                         description='Final Test Evaluation',
                                                                                         image_paths=image_paths)
    
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

    # 创建保存预测正确和预测错误图片的文件夹
    correct_dir = f'test_results/{test_process_name}/correct'
    incorrect_dir = f'test_results/{test_process_name}/incorrect'
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(incorrect_dir, exist_ok=True)

    # 保存图片
    for pred, label, img_path in zip(all_preds, all_labels, all_image_paths):
        img = Image.open(img_path)
        category = os.path.basename(os.path.dirname(img_path))
        if pred == label:
            save_dir = os.path.join(correct_dir, category)
        else:
            save_dir = os.path.join(incorrect_dir, category)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(img_path))
        img.save(save_path)

    full_train_total_time = time.time() - full_train_start_time
    total_time_str = str(datetime.timedelta(seconds=int(full_train_total_time)))
    print()
    print(f'Whole Test Time: {total_time_str}')

if __name__ == '__main__':
    main()