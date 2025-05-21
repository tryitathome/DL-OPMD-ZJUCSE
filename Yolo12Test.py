import os
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix

# 配置参数
TEST_IMAGES_DIR = "./yolo_dataset_shengkouGen2/images/test"
TEST_LABELS_DIR = "./yolo_dataset_shengkouGen2/labels/test"
MODEL_PATH = "./best_155epoch_shengkouV2.pt"
DATA_YAML = "./yolo_dataset_shengkouGen2/data.yaml"
RESULTS_DIR = "./evaluation_results"
CONF_THRES = 0.25
IOU_THRES = 0.6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 加载模型
    model = YOLO(MODEL_PATH).to(DEVICE)
    
    # 执行验证
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        batch=16,
        imgsz=640,
        conf=CONF_THRES,
        iou=IOU_THRES,
        save_json=True,
        plots=True,
        project=RESULTS_DIR
    )
    
    # 打印关键指标（修正部分）
    print("\n测试集评估结果：")
    print(f"mAP@0.5:       {metrics.box.map50:.4f}")
    print(f"mAP@0.5-0.95: {metrics.box.map:.4f}")
    print(f"平均精确率:    {metrics.box.mp:.4f}")  # 使用mp属性
    print(f"平均召回率:     {metrics.box.mr:.4f}")  # 使用mr属性
    print(f"F1分数均值:    {np.mean(metrics.box.f1):.4f}")  # 计算f1均值
    
    # 生成混淆矩阵
    confusion_matrix = ConfusionMatrix(nc=model.model.names)
    confusion_matrix.process_batch(
        detections=metrics.pred,
        gt_bboxes=metrics.labels[:, 1:],
        gt_cls=metrics.labels[:, 0]
    )
    confusion_matrix.plot(save_dir=RESULTS_DIR, names=list(model.model.names.values()))
    
    # 保存详细结果
    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
        f.write(f"模型路径: {MODEL_PATH}\n")
        f.write(f"测试集样本数: {metrics.nt_per_class.sum()}\n")
        f.write(f"mAP@0.5: {metrics.box.map50:.4f}\n")
        f.write(f"mAP@0.5-0.95: {metrics.box.map:.4f}\n")
        f.write(f"平均精确率: {metrics.box.mp:.4f}\n")
        f.write(f"平均召回率: {metrics.box.mr:.4f}\n")
        f.write(f"F1分数均值: {np.mean(metrics.box.f1):.4f}\n\n")
        
        f.write("各类别详细指标：\n")
        for i, name in model.model.names.items():
            f.write(f"{name}:\n")
            f.write(f"  精确率: {metrics.box.p[i]:.4f}\n")  # 使用p属性
            f.write(f"  召回率: {metrics.box.r[i]:.4f}\n")  # 使用r属性
            f.write(f"  AP@0.5: {metrics.box.ap50[i]:.4f}\n")
            f.write(f"  AP@0.5-0.95: {metrics.box.ap[i]:.4f}\n")

if __name__ == "__main__":
    main()
    print(f"\n评估结果已保存至：{RESULTS_DIR}")
