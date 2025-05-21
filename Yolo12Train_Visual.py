import os
import yaml
from ultralytics import YOLO
import torch
import shutil
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import json
import logging
import csv

def setup_logger(log_file):
    """设置日志记录器"""
    logger = logging.getLogger('yolo_training')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def parse_training_results(csv_file):
    """解析训练日志CSV文件"""
    if not os.path.exists(csv_file):
        return None
    
    metrics_data = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'precision': [], 'recall': [], 
        'mAP50': [], 'mAP50-95': [], 'learning_rate': []
    }
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # 读取表头
            
            # 找到各个指标的列索引
            epoch_idx = header.index('epoch') if 'epoch' in header else None
            box_loss_idx = header.index('train/box_loss') if 'train/box_loss' in header else None
            cls_loss_idx = header.index('train/cls_loss') if 'train/cls_loss' in header else None
            dfl_loss_idx = header.index('train/dfl_loss') if 'train/dfl_loss' in header else None
            val_box_loss_idx = header.index('val/box_loss') if 'val/box_loss' in header else None
            val_cls_loss_idx = header.index('val/cls_loss') if 'val/cls_loss' in header else None
            val_dfl_loss_idx = header.index('val/dfl_loss') if 'val/dfl_loss' in header else None
            precision_idx = header.index('metrics/precision(B)') if 'metrics/precision(B)' in header else None
            recall_idx = header.index('metrics/recall(B)') if 'metrics/recall(B)' in header else None
            map50_idx = header.index('metrics/mAP50(B)') if 'metrics/mAP50(B)' in header else None
            map_idx = header.index('metrics/mAP50-95(B)') if 'metrics/mAP50-95(B)' in header else None
            lr_idx = header.index('lr/pg0') if 'lr/pg0' in header else None
            
            for row in reader:
                if epoch_idx is not None:
                    metrics_data['epoch'].append(int(float(row[epoch_idx])))
                
                # 计算训练损失总和
                train_loss = 0
                if box_loss_idx is not None:
                    train_loss += float(row[box_loss_idx])
                if cls_loss_idx is not None:
                    train_loss += float(row[cls_loss_idx])
                if dfl_loss_idx is not None:
                    train_loss += float(row[dfl_loss_idx])
                metrics_data['train_loss'].append(train_loss)
                
                # 计算验证损失总和
                val_loss = 0
                if val_box_loss_idx is not None:
                    val_loss += float(row[val_box_loss_idx])
                if val_cls_loss_idx is not None:
                    val_loss += float(row[val_cls_loss_idx])
                if val_dfl_loss_idx is not None:
                    val_loss += float(row[val_dfl_loss_idx])
                metrics_data['val_loss'].append(val_loss)
                
                if precision_idx is not None:
                    metrics_data['precision'].append(float(row[precision_idx]))
                if recall_idx is not None:
                    metrics_data['recall'].append(float(row[recall_idx]))
                if map50_idx is not None:
                    metrics_data['mAP50'].append(float(row[map50_idx]))
                if map_idx is not None:
                    metrics_data['mAP50-95'].append(float(row[map_idx]))
                if lr_idx is not None:
                    metrics_data['learning_rate'].append(float(row[lr_idx]))
        
        return metrics_data
    except Exception as e:
        print(f"解析训练日志时出错: {e}")
        return None

def create_training_plots(metrics, save_dir):
    """创建训练过程的可视化图表"""
    if not metrics or len(metrics['epoch']) == 0:
        print("没有足够的数据创建图表")
        return
    
    last_epoch = metrics['epoch'][-1]
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    axs[0, 0].plot(metrics['epoch'], metrics['train_loss'], label='训练损失')
    axs[0, 0].plot(metrics['epoch'], metrics['val_loss'], label='验证损失')
    axs[0, 0].set_title('损失曲线')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # 精确率和召回率
    axs[0, 1].plot(metrics['epoch'], metrics['precision'], label='精确率')
    axs[0, 1].plot(metrics['epoch'], metrics['recall'], label='召回率')
    axs[0, 1].set_title('精确率和召回率')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Value')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # mAP曲线
    axs[1, 0].plot(metrics['epoch'], metrics['mAP50'], label='mAP@0.5')
    axs[1, 0].plot(metrics['epoch'], metrics['mAP50-95'], label='mAP@0.5:0.95')
    axs[1, 0].set_title('mAP曲线')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('mAP')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # 学习率曲线
    axs[1, 1].plot(metrics['epoch'], metrics['learning_rate'])
    axs[1, 1].set_title('学习率曲线')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Learning Rate')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_progress_epoch_{last_epoch}.png'))
    plt.close()

def create_class_performance_plot(metrics, class_names, save_dir):
    """创建类别性能对比图"""
    plt.figure(figsize=(12, 8))
    
    classes = list(class_names.values())
    x = np.arange(len(classes))
    width = 0.15
    
    plt.bar(x - width*2, metrics.box.p, width, label='精确率')
    plt.bar(x - width, metrics.box.r, width, label='召回率')
    plt.bar(x, metrics.box.ap50, width, label='mAP@0.5')
    plt.bar(x + width, metrics.box.ap, width, label='mAP@0.5:0.95')
    plt.bar(x + width*2, metrics.box.f1, width, label='F1分数')
    
    plt.xlabel('类别')
    plt.ylabel('分数')
    plt.title('各类别性能对比')
    plt.xticks(x, classes, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y')
    plt.savefig(os.path.join(save_dir, 'class_performance.png'))
    plt.close()

def train_yolov11():
    # 配置参数
    data_yaml_path = r"D:\MyWorkSpace\YOLO12\yolo_dataset_shengkouGen2\data.yaml"
    model_size = "s"  # 可选: n(nano), s(small), m(medium), l(large), x(xlarge)
    batch_size = 32   # 根据你的GPU内存调整
    epochs = 200      # 训练轮数
    img_size = 640    # 输入图像大小
    patience = 20     # 早停patience
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"runs/train_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 设置日志记录
    log_file = os.path.join(results_dir, "training.log")
    logger = setup_logger(log_file)
    
    logger.info("开始训练YOLOv12检测模型")
    logger.info(f"配置参数: model_size={model_size}, batch_size={batch_size}, epochs={epochs}, img_size={img_size}")
    
    # 备份data.yaml到结果目录
    shutil.copy(data_yaml_path, os.path.join(results_dir, "data.yaml"))
    
    # 检查CUDA是否可用
    device = 0 if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {'CUDA' if device == 0 else 'CPU'}")
    
    # 加载模型
    model = YOLO(f"yolo12{model_size}.pt") #(r"D:\MyWorkSpace\YOLO12\runs\train_20250226_1413412\weights\last.pt")
    
    # 打印数据集信息
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    logger.info(f"训练集路径: {data_config['train']}")
    logger.info(f"验证集路径: {data_config['val']}")
    logger.info(f"类别数量: {data_config['nc']}")
    logger.info(f"类别名称: {data_config['names']}")
    
    # 训练模型
    logger.info(f"开始训练YOLOv12-{model_size}模型...")
    results = model.train(
        resume=False,  # 重新开始中断的训练
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        patience=patience,
        project=results_dir.split('/')[0],
        name=results_dir.split('/')[1],
        pretrained=True,
        optimizer="AdamW",  # 使用AdamW优化器
        lr0=0.001,          # 初始学习率
        lrf=0.005,           # 最终学习率(相对于初始学习率的比例)
        weight_decay=0.0005,# 权重衰减
        warmup_epochs=3,    # 预热轮数
        cos_lr=True,        # 使用余弦学习率调度
        close_mosaic=10,    # 最后10个epoch关闭mosaic增强
        augment=True,       # 使用数据增强
        save=True,          # 保存模型
        save_period=10,     # 每10个epoch保存一次
        verbose=True        # 详细输出
    )
    
    # 训练完成后，获取训练日志文件
    logger.info("训练完成，正在处理训练日志...")
    csv_path = os.path.join(results_dir, 'results.csv')  # YOLO自动生成的日志文件
    
    # 如果没有找到自动生成的CSV，尝试在其他可能的位置查找
    if not os.path.exists(csv_path):
        possible_paths = glob(os.path.join(results_dir, '**', 'results.csv'), recursive=True)
        if possible_paths:
            csv_path = possible_paths[0]
    
    # 解析训练日志并生成Excel文件
    if os.path.exists(csv_path):
        logger.info(f"找到训练日志CSV文件: {csv_path}")
        
        # 将CSV直接复制到结果目录的根目录
        shutil.copy(csv_path, os.path.join(results_dir, 'training_log.csv'))
        
        # 解析日志
        training_metrics = parse_training_results(csv_path)
        if training_metrics:
            # 保存为Excel
            df = pd.DataFrame(training_metrics)
            excel_path = os.path.join(results_dir, 'training_metrics.xlsx')
            df.to_excel(excel_path, index=False)
            logger.info(f"训练指标已保存到Excel: {excel_path}")
            
            # 创建训练进度图
            create_training_plots(training_metrics, results_dir)
            logger.info(f"训练进度图已生成")
    else:
        logger.warning("未找到训练日志CSV文件")
    
    # 评估模型
    logger.info("开始在验证集上评估模型...")
    metrics = model.val()
    
    # 打印评估结果
    logger.info("\n评估结果:")
    logger.info(f"mAP@0.5: {metrics.box.map50:.4f}")
    logger.info(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    
    # 创建类别性能对比图
    create_class_performance_plot(metrics, data_config['names'], results_dir)
    logger.info("类别性能对比图已生成")
    
    # 收集类别指标并保存到Excel
    class_metrics = {
        'class_name': [],
        'precision': [],
        'recall': [],
        'mAP50': [],
        'mAP50-95': [],
        'f1_score': []
    }
    
    for i, name in enumerate(data_config['names'].values()):
        class_metrics['class_name'].append(name)
        class_metrics['precision'].append(float(metrics.box.p[i]))
        class_metrics['recall'].append(float(metrics.box.r[i]))
        class_metrics['mAP50'].append(float(metrics.box.ap50[i]))
        class_metrics['mAP50-95'].append(float(metrics.box.ap[i]))
        class_metrics['f1_score'].append(float(metrics.box.f1[i]))
    
    class_df = pd.DataFrame(class_metrics)
    class_excel_path = os.path.join(results_dir, 'class_metrics.xlsx')
    class_df.to_excel(class_excel_path, index=False)
    logger.info(f"类别指标已保存到Excel: {class_excel_path}")
    
    # 保存训练结果摘要
    with open(os.path.join(results_dir, "training_summary.txt"), "w") as f:
        f.write(f"训练时间: {timestamp}\n")
        f.write(f"模型: YOLOv12-{model_size}\n")
        f.write(f"训练轮数: {epochs}\n")
        f.write(f"批次大小: {batch_size}\n")
        f.write(f"图像大小: {img_size}\n")
        f.write(f"mAP@0.5: {metrics.box.map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {metrics.box.map:.4f}\n")
        f.write(f"平均精确率: {metrics.box.mp:.4f}\n")
        f.write(f"平均召回率: {metrics.box.mr:.4f}\n")
        f.write(f"平均F1分数: {np.mean(metrics.box.f1):.4f}\n\n")
        
        f.write("各类别详细指标：\n")
        for i, name in enumerate(data_config['names'].values()):
            f.write(f"{name}:\n")
            f.write(f"  精确率: {metrics.box.p[i]:.4f}\n")
            f.write(f"  召回率: {metrics.box.r[i]:.4f}\n")
            f.write(f"  mAP@0.5: {metrics.box.ap50[i]:.4f}\n")
            f.write(f"  mAP@0.5:0.95: {metrics.box.ap[i]:.4f}\n")
            f.write(f"  F1分数: {metrics.box.f1[i]:.4f}\n\n")
    
    logger.info(f"训练完成! 结果保存在: {results_dir}")
    return model, results_dir

if __name__ == "__main__":
    # 检查环境
    print("Python版本:", os.sys.version)
    print("PyTorch版本:", torch.__version__)
    print("CUDA是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA设备:", torch.cuda.get_device_name(0))
        print("可用GPU数量:", torch.cuda.device_count())
    
    # 开始训练
    model, results_dir = train_yolov11()