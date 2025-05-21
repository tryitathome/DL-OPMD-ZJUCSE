import os
import yaml
from ultralytics import YOLO
import torch
import shutil
from datetime import datetime

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
    
    # 备份data.yaml到结果目录
    shutil.copy(data_yaml_path, os.path.join(results_dir, "data.yaml"))
    
    # 检查CUDA是否可用
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {'CUDA' if device == 0 else 'CPU'}")
    
    # 加载模型
    model = YOLO(f"yolo12{model_size}.pt") #(r"D:\MyWorkSpace\YOLO12\runs\train_20250226_1413412\weights\last.pt")
    
    # 打印数据集信息
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    print(f"训练集路径: {data_config['train']}")
    print(f"验证集路径: {data_config['val']}")
    print(f"类别数量: {data_config['nc']}")
    print(f"类别名称: {data_config['names']}")
    
    # 训练模型
    print(f"开始训练YOLOv12-{model_size}模型...")
    results = model.train(
        resume=False,#重新开始中断的训练
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
    
    # 评估模型
    print("开始在验证集上评估模型...")
    metrics = model.val()
    
    # 打印评估结果
    print("\n评估结果:")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    
    # 保存训练结果摘要
    with open(os.path.join(results_dir, "training_summaryV2.txt"), "w") as f:
        f.write(f"训练时间: {timestamp}\n")
        f.write(f"模型: YOLOv12-{model_size}\n")
        f.write(f"训练轮数: {epochs}\n")
        f.write(f"批次大小: {batch_size}\n")
        f.write(f"图像大小: {img_size}\n")
        f.write(f"mAP@0.5: {metrics.box.map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {metrics.box.map:.4f}\n")
    
    print(f"\n训练完成! 结果保存在: {results_dir}")
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
    
    # # 可选：在训练后进行推理测试
    # test_img_path = r"D:\MyWorkSpace\LM数据集\yolo_dataset\val\images\DSC_0001.jpg"  # 替换为你的测试图像
    # if os.path.exists(test_img_path):
    #     print(f"\n在测试图像上进行推理: {test_img_path}")
    #     results = model.predict(test_img_path, save=True, project=results_dir, name="test_inference")
    #     print(f"推理结果保存在: {results_dir}/test_inference")
