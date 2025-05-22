import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np
import os
import time
import datetime

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data import Mixup

from utils.train_process import train_epoch, train_epoch_mixup
from utils.train_process import evaluate
from utils.train_vis import plot_history, plot_lr
from utils.resume import save_checkpoint, load_checkpoint
from utils.final_test import test

# 导入自定义模型
from models.simple_cnn import SimpleCNN
from models.resnet18 import ResNet18
from models.resnet50 import ResNet50
from models.vit import ViT
from models.densenet import DenseNet121

from build_model.build_swin import build_swin_transformer, build_swin_transformer_v2
from build_model.build_convnext import build_convnext
from build_model.build_replknet import build_replknet
from build_model.build_metaformer import caformer_b36_in21k, convformer_b36_in21ft1k, identityformer_m48, randformer_m48, poolformerv2_m48
from build_model.build_metaformer import caformer_b36_384_in21ft1k

from utils.learning_rate import build_lr_schedular
from utils.build_loss import BinaryFocalLoss, FocalLoss
from utils.build_optimizer import build_optimizer, build_optimizer_transformer
from utils.load_data import build_loader_memmap, build_loader_imagenet1k


try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def main():

    '''数据参数'''
    # 训练过程名字
    train_process_name = 'caformer_s18_eva02_imagenet1k'
    # train_process_name = 'eva02_l_test'

    # 数据集路径和输出路径
    dataset_dir = 'ImageNet1k'
    output_dir  = 'pretrain_imagenet1k'

    # 是否使用内存映射加速数据读取
    use_memmap  = False
    memmap_dir  = 'memmap_file'

    # 是否将数据集先存入内存, 严重过拟合
    load_into_memory = False

    # 是否断点重训
    resume      = False
    resume_path = 'output/caformer_b36_eva02_nopre/checkpoints/model_epoch_120.pth'

    # 是否加载预训练权重
    pretrained      = False
    pretrained_path = 'pretrained/metaformer/caformer_s18.pth'

    image_size  = 384
    batch_size  = 32
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
    num_classes     = 1000
    dropout_rate    = 0.5
    droppath_rate   = 0.1
    label_smoothing = True
    label_smoothing_ratio = 0.1

    # 模型是否为transformer类模型
    is_transformer = False
    
    '''训练参数'''
    num_epochs    = 300
    save_epochs   = 2
    warmup_epochs = 5
    start_epoch   = 0

    init_lr       = 5e-4
    # init_lr       = 1e-3
    warm_init_lr  = 5e-7
    min_lr        = 5e-6

    weight_decay  = 0.05

    clip_grad = 5
    accumulation_steps = 0

    # 调度器
    schedular_name = 'cosine'    # 'linear' 'step'
    decay_epochs   = 30
    decay_rate     = 0.1

    # 优化器
    optimizer_name  = 'adamw'    # 'sgd'
    optimizer_eps   = 1e-8
    optimizer_betas = (0.9, 0.999)
    momentum        = 0.9

    '''增强参数'''
    color_jitter = 0.4
    auto_augment = 'rand-m9-mstd0.5-inc1'    # 'v0' or 'original'
    re_prob      = 0.25
    re_mode      = 'pixel'
    re_count     = 1

    mixup        = True
    mixup_ratio  = 0.8

    cutmix       = True
    cutmix_ratio = 1.0

    cutmix_minmax     = None
    mixup_prob        = 1.0
    mixup_switch_prob = 0.5
    mixup_mode        = 'batch'    # 'pair' or 'elem

    '''测试参数'''
    best_accuracy = 0.
    # 先将图片进行resize放大, 再中心裁剪
    test_crop = False

    '''其他参数'''
    # loss有关
    focal_loss = False
    class_loss_weights = np.ones([num_classes], np.float32)
    # 设置指标参数
    # average_mode = 'none'     # 输出为每个类的指标
    average_mode = 'macro'      # 输出为每个类的平均
    thrs = 0.

    '''线性调整学习率'''
    linear_scaled_lr        = init_lr * batch_size / 512.0
    linear_scaled_warmup_lr = warm_init_lr * batch_size / 512.0
    linear_scaled_min_lr    = min_lr * batch_size / 512.0

    if accumulation_steps > 1:
        linear_scaled_lr        = linear_scaled_lr * accumulation_steps
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * accumulation_steps
        linear_scaled_min_lr    = linear_scaled_min_lr * accumulation_steps

    init_lr      = linear_scaled_lr
    warm_init_lr = linear_scaled_warmup_lr
    min_lr       = linear_scaled_min_lr

    # 建立DataLoader
    if use_memmap:
        train_loader, val_loader, test_loader = build_loader_memmap(memmap_dir, dataset_dir, 
                                                                    image_size, batch_size,
                                                                    num_workers, pin_memory, persistent_workers, 
                                                                    color_jitter, auto_augment, re_prob, re_mode, re_count, interpolation, test_crop)
    else:
        train_loader, val_loader, test_loader = build_loader_imagenet1k(dataset_dir, image_size, batch_size, 
                                                             load_into_memory, num_workers, pin_memory, persistent_workers, 
                                                             color_jitter, auto_augment, re_prob, re_mode, re_count, interpolation, test_crop)

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

    # from build_model.build_metaformer import caformer_s18
    # model = caformer_s18(pretrained=False, num_classes=num_classes)

    from build_model.build_metaformer import caformer_s18
    from build_model.build_my_metaformer import caformer_s18_msf, caformer_s18_eva02, fulla_former_s18_eva02, caformer_b36_eva02
    model = caformer_s18_eva02(pretrained=False, num_classes=num_classes)
    is_metaformer = True


    # from models.eva02 import eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE
    # model = eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE(num_classes=1000, img_size=image_size)
    # is_metaformer = False

    # 计算参数量和FLOPS
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_parameters / 1e7}M.')
    '''
    flops = model.flops()
    print(f'Number of FLOPS:      {flops / 1e9}G.')
    '''

    # 定义优化器 
    if is_transformer:
        optimizer = build_optimizer_transformer(optimizer_name, model, momentum, init_lr, weight_decay, optimizer_eps, optimizer_betas)
    else:
        optimizer = build_optimizer(optimizer_name, model, momentum, init_lr, weight_decay, optimizer_eps, optimizer_betas)

    # 定义学习率调度器
    num_iters_per_epoch = len(train_loader)
    schedular = build_lr_schedular(schedular_name, num_epochs, warmup_epochs, decay_epochs, 
                                   min_lr, warm_init_lr, decay_rate, 
                                   optimizer, num_iters_per_epoch)
    
    # 断点续训
    if resume:
        try:
            model, optimizer, start_epoch = load_checkpoint(resume_path, model, optimizer, device)
            print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        except FileNotFoundError:
            print("No checkpoint found. Starting from scratch.")
    else:
        # 载入预训练参数
        if pretrained:
            pretrained_weights = torch.load(pretrained_path, map_location='cpu')
            # 为不完全匹配的预训练权重设计
            new_state_dict = {k: v for k, v in pretrained_weights.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
            model.load_state_dict(new_state_dict, strict=False)
            # 为 Swin Transformer 模型设计
            # msg = model.load_state_dict(pretrained_weights['model'], strict=False)
            # 一般权重全匹配设计
            # model.load_state_dict(pretrained_weights, strict=False)
            print('Pretrained weights loaded.')
        else:
            pass

    # 修改分类头
    if is_metaformer:
        model.to(device)
    else:
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, num_classes)
        model.to(device)

    # print(model)

    # 混合精度
    if amp_level != 'O0':
        model, optimizer = amp.initialize(model, optimizer, opt_level=amp_level)
    
    # 定义loss
    if mixup and mixup_ratio > 0:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif label_smoothing and label_smoothing_ratio > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing_ratio)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    '''
    # 定义损失函数和优化器
    if focal_loss and num_classes==2:
        criterion = BinaryFocalLoss(gamma=2.0, alpha=0.25)
    elif focal_loss and num_classes!=2:
        criterion = FocalLoss(gamma=2.0, alpha=1, epsilon=1.e-9, device=device)
    else:
        criterion = nn.CrossEntropyLoss()
    '''

    # 定义 mixup 函数
    mixup_function = None
    is_mixup = mixup or cutmix or cutmix_minmax is not None
    if is_mixup:
        mixup_function = Mixup(
            mixup_alpha=mixup_ratio, cutmix_alpha=cutmix_ratio, cutmix_minmax=cutmix_minmax, 
            prob=mixup_prob, switch_prob=mixup_switch_prob, mode=mixup_mode,
            label_smoothing=label_smoothing_ratio, num_classes=num_classes
        )

    # 准备收集训练历史数据
    history = {'train_loss': [], 'val_loss': [], 
               'train_acc':  [], 'val_acc': [], 
               'train_precision': [], 'val_precision': [],
               'train_recall':    [], 'val_recall':    [],
               'train_f1_score':  [], 'val_f1_score':  [], 
               'lr': []
    }

    # 创建保存训练结果的目录
    os.makedirs(f'{output_dir}/{train_process_name}/checkpoints', exist_ok=True)
    os.makedirs(f'{output_dir}/{train_process_name}/results',     exist_ok=True)

    # 训练和验证模型
    print(f'Using {device}')
    print('Start Training...')
    print()
    # mixup_function = None
    full_train_start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        '''
        train_epoch(model, train_loader,
                    device, schedular, criterion, optimizer,
                    epoch, num_epochs, mixup_function,
                    history, average_mode, thrs)
        '''
        train_epoch_mixup(model, train_loader,
                    device, schedular, criterion, optimizer,
                    epoch, num_epochs, mixup_function, accumulation_steps, clip_grad, amp_level,
                    history)
        val_accuracy = evaluate(model, val_loader, 
                                device, criterion, 
                                epoch,
                                history, average_mode, thrs,
                                description=f'Epoch {epoch+1}/{num_epochs} - Validation')
        
        # 输出学习率
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        print(f'Epoch {epoch+1} completed. Current learning rate: {current_lr:.6f}\n')

        # 绘制并保存图表
        plot_history(history, epoch, f'./{output_dir}/{train_process_name}/results', train_process_name)
        plot_lr(history, epoch, f'./{output_dir}/{train_process_name}/results', train_process_name)

        # 最优保存
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint_path = f'./{output_dir}/{train_process_name}/checkpoints/best_model.pth'
            save_checkpoint(epoch+1, model, optimizer, save_checkpoint_path)
            print(f'New best model saved with accuracy: {val_accuracy:.2f}%\n')

        # 定期保存模型
        if (epoch + 1) % save_epochs == 0:
            save_checkpoint_path = f'./{output_dir}/{train_process_name}/checkpoints/model_epoch_{epoch+1}.pth'
            save_checkpoint(epoch+1, model, optimizer, save_checkpoint_path)
            print(f'Model saved at epoch {epoch+1}\n')

    # 在所有训练结束后，评估模型在测试集上的表现
    best_checkpoints_path = f'./{output_dir}/{train_process_name}/checkpoints/best_model.pth'
    test(train_process_name, test_loader, model, best_checkpoints_path,
         device, average_mode, thrs, description='Final Test Evaluation')
    
    full_train_total_time = time.time() - full_train_start_time
    total_time_str = str(datetime.timedelta(seconds=int(full_train_total_time)))
    print()
    print(f'Whole Training Time: {total_time_str}')


if __name__ == '__main__':

    seed = -1
    if seed >= 0 and seed <= 2**32 - 1:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    cudnn.benchmark = True

    '''是否使用混合精度训练'''
    amp_level = 'O1'    # 'O0' = None, 'O1', 'O2'
    main()
