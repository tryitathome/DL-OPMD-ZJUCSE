import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from glob import glob
from tqdm import tqdm
import argparse
import shutil
import pandas as pd

#使用例子：python Yolo12Inference.py --model best_155epoch_shengkouV2.pt --source MiniTestData --output MiniInferenceResults --conf 0.5 --line-thickness 10
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv12 推理脚本')
    parser.add_argument('--model', type=str, default='./best_155epoch_shengkouV2.pt', help='模型路径')
    parser.add_argument('--source', type=str, default='./inference_images', help='输入图像文件夹路径')
    parser.add_argument('--output', type=str, default='./inference_results', help='输出文件夹路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU阈值')
    parser.add_argument('--device', type=str, default='', help='设备 (可选: cuda, cpu)')
    parser.add_argument('--save-txt', action='store_true', help='保存标签文件')
    parser.add_argument('--save-conf', action='store_true', help='在标签文件中保存置信度')
    parser.add_argument('--classes', nargs='+', type=int, help='仅检测指定类别')
    parser.add_argument('--max-det', type=int, default=300, help='每张图像的最大检测数')
    parser.add_argument('--line-thickness', type=int, default=2, help='边界框线条粗细')
    parser.add_argument('--as-classifier', action='store_true', help='将检测器作为分类器使用，保存按类别分类的图片')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 确保输入路径存在
    if not os.path.exists(args.source):
        print(f"错误: 输入路径 '{args.source}' 不存在")
        return
    
    # 创建输出文件夹
    os.makedirs(args.output, exist_ok=True)
    
    # 创建检测结果可视化文件夹
    detection_vis_dir = os.path.join(args.output, "detection_visualizations")
    os.makedirs(detection_vis_dir, exist_ok=True)
    
    # 创建标签输出文件夹（如果需要）
    if args.save_txt:
        labels_dir = os.path.join(args.output, "labels")
        os.makedirs(labels_dir, exist_ok=True)
    
    # 如果使用检测器作为分类器，创建分类结果文件夹
    classify_dir = None
    if args.as_classifier:
        classify_dir = os.path.join(args.output, "classification_results")
        os.makedirs(classify_dir, exist_ok=True)
        # 创建Excel记录数据列表
        classification_records = []
    
    # 设置设备
    if args.device:
        device = args.device
    else:
        device = 0 if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {device}")
    
    # 加载模型
    try:
        model = YOLO(args.model).to(device)
        print(f"模型已加载: {args.model}")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return
    
    # 打印类别信息
    print(f"类别: {model.model.names}")
    
    # 为分类器模式创建每个类别的文件夹
    if args.as_classifier:
        for cls_id, cls_name in model.model.names.items():
            class_dir = os.path.join(classify_dir, f"{cls_id}_{cls_name}")
            os.makedirs(class_dir, exist_ok=True)
    
    # 定义不同类别的颜色
    colors = {
        0: (0, 255, 0),    # 绿色
        1: (255, 255, 255), # 白色
        2: (0, 0, 255),    # 蓝色
        3: (128, 0, 0),    # 深红色
        4: (255, 255, 0),  # 黄色
        5: (255, 0, 255),  # 紫色
        6: (0, 255, 255),  # 青色
        7: (255, 0, 0),    # 红色
        8: (0, 128, 0),    # 深绿色
        9: (0, 0, 128)     # 深蓝色
    }
    
    # 获取所有图像文件
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(args.source, f"*.{ext}")))
        image_files.extend(glob(os.path.join(args.source, f"*.{ext.upper()}")))
    
    if not image_files:
        print(f"警告: 在 '{args.source}' 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 创建结果摘要文件
    summary_file = os.path.join(args.output, "inference_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"模型: {args.model}\n")
        f.write(f"图像数量: {len(image_files)}\n")
        f.write(f"置信度阈值: {args.conf}\n")
        f.write(f"IoU阈值: {args.iou}\n")
        f.write(f"设备: {device}\n\n")
        f.write("类别索引对应关系:\n")
        for idx, name in model.model.names.items():
            f.write(f"{idx}: {name}\n")
        f.write("\n检测结果摘要:\n")
    
    # 统计每个类别的检测数量
    class_counts = {i: 0 for i in model.model.names.keys()}
    total_detections = 0
    detection_confidence = {i: [] for i in model.model.names.keys()}
    
    # 在所有图像上进行推理
    print("开始推理...")
    for idx, img_path in enumerate(tqdm(image_files)):
        # 获取图像文件名
        img_basename = os.path.splitext(os.path.basename(img_path))[0]
        img_extension = os.path.splitext(os.path.basename(img_path))[1]
        
        # 使用模型推理
        results = model(img_path, conf=args.conf, iou=args.iou, classes=args.classes, max_det=args.max_det)
        
        # 如果有检测结果
        if len(results[0].boxes) > 0:
            # 获取所有检测框的置信度和类别
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # 统计各类别检测结果
            for cls, conf in zip(classes, confidences):
                class_counts[cls] += 1
                detection_confidence[cls].append(conf)
                total_detections += 1
            
            # 保存标签文件（如果需要）
            if args.save_txt:
                # 获取图像尺寸
                img = cv2.imread(img_path)
                height, width = img.shape[:2]
                
                # 创建标签文件
                label_path = os.path.join(labels_dir, f"{img_basename}.txt")
                with open(label_path, 'w') as f:
                    for box, cls, conf in zip(boxes, classes, confidences):
                        # 转换为YOLO格式 (class x_center y_center width height)
                        x_center = (box[0] + box[2]) / (2 * width)
                        y_center = (box[1] + box[3]) / (2 * height)
                        box_width = (box[2] - box[0]) / width
                        box_height = (box[3] - box[1]) / height
                        
                        if args.save_conf:
                            f.write(f"{cls} {x_center} {y_center} {box_width} {box_height} {conf}\n")
                        else:
                            f.write(f"{cls} {x_center} {y_center} {box_width} {box_height}\n")
            
            # 将检测器作为分类器使用（如果需要）
            if args.as_classifier:
                # 找到置信度最高的检测框作为分类结果
                max_conf_idx = np.argmax(confidences)
                pred_class = classes[max_conf_idx]
                max_conf = confidences[max_conf_idx]
                
                # 保存图片到对应类别的文件夹
                class_dir = os.path.join(classify_dir, f"{pred_class}_{model.model.names[pred_class]}")
                dest_path = os.path.join(class_dir, f"{img_basename}_conf{max_conf:.2f}{img_extension}")
                shutil.copy(img_path, dest_path)
                
                # 添加记录到分类结果列表
                classification_records.append({
                    '序号': idx + 1,
                    '文件名': f"{img_basename}{img_extension}",
                    '检测类别': model.model.names[pred_class],
                    '类别ID': pred_class,
                    '置信度': max_conf
                })
            
            # 保存检测框可视化结果
            # 获取原始图像
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
            
            # 绘制所有检测框
            for i in range(len(results[0].boxes)):
                box = results[0].boxes[i].xyxy.cpu().numpy()[0].astype(int)
                cls = int(results[0].boxes[i].cls.cpu().numpy()[0])
                conf = float(results[0].boxes[i].conf.cpu().numpy()[0])
                
                # 获取当前类别的颜色，如果类别超出预定义范围，则使用最后一个颜色
                color = colors.get(cls, colors.get(9))
                
                # 绘制边界框
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, args.line_thickness)#彩色版本
                # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), args.line_thickness)#单色版本
                # 添加类别名称和置信度，字体大小扩大4倍
                #label = f"{model.model.names[cls]}"#不保存置信度
                label = f"{model.model.names[cls]}: {conf:.2f}" #保存置信度
                #cv2.putText(img, label, (box[0], box[1] + 50), 
                            #cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 10)#单色版本中字
                cv2.putText(img, label, (box[0], box[1] + 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 10) #彩色版本小字
            # 转回BGR格式用于保存
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # 保存带有检测框的图像
            vis_img_path = os.path.join(detection_vis_dir, f"{img_basename}_detection{img_extension}")
            cv2.imwrite(vis_img_path, img)
        else:
            # 如果没有检测到任何目标，保存原始图像到可视化文件夹
            shutil.copy(img_path, os.path.join(detection_vis_dir, f"{img_basename}_nodetection{img_extension}"))
            
            # 如果使用分类器模式，将未检测到的图片记录为"未知"
            if args.as_classifier:
                classification_records.append({
                    '序号': idx + 1,
                    '文件名': f"{img_basename}{img_extension}",
                    '检测类别': "未知",
                    '类别ID': -1,
                    '置信度': 0.0
                })
    
    # 更新结果摘要
    with open(summary_file, 'a') as f:
        f.write(f"总检测数: {total_detections}\n\n")
        f.write("各类别检测统计:\n")
        for cls_id, count in class_counts.items():
            if count > 0:
                avg_conf = np.mean(detection_confidence[cls_id])
                f.write(f"{model.model.names[cls_id]}: {count} 个检测, 平均置信度: {avg_conf:.4f}\n")
    
    # 如果使用分类器模式，生成分类结果Excel文件
    if args.as_classifier:
        classification_df = pd.DataFrame(classification_records)
        excel_path = os.path.join(args.output, "classification_results.xlsx")
        classification_df.to_excel(excel_path, index=False)
        print(f"分类结果已保存到Excel: {excel_path}")
        
        # 生成分类结果统计信息
        class_stats = classification_df['检测类别'].value_counts().reset_index()
        class_stats.columns = ['类别', '数量']
        class_stats_path = os.path.join(args.output, "classification_stats.xlsx")
        class_stats.to_excel(class_stats_path, index=False)
        print(f"分类统计结果已保存到Excel: {class_stats_path}")
        
        # 生成分类结果可视化
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            class_stats.plot(kind='bar', x='类别', y='数量', legend=False, figsize=(12, 6))
            plt.title('分类结果统计')
            plt.xlabel('类别')
            plt.ylabel('图片数量')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output, 'classification_stats.png'))
            plt.close()
        except Exception as e:
            print(f"创建分类统计图表时出错: {e}")
    
    # 创建类别检测统计图
    try:
        import matplotlib.pyplot as plt
        
        # 各类别检测数量柱状图
        plt.figure(figsize=(12, 6))
        class_names = [model.model.names[i] for i in class_counts.keys()]
        counts = list(class_counts.values())
        
        plt.bar(class_names, counts)
        plt.xlabel('类别')
        plt.ylabel('检测数量')
        plt.title('各类别检测数量')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, 'class_detection_counts.png'))
        plt.close()
        
        # 各类别平均置信度柱状图
        plt.figure(figsize=(12, 6))
        avg_confidences = []
        for cls_id in class_counts.keys():
            if class_counts[cls_id] > 0:
                avg_confidences.append(np.mean(detection_confidence[cls_id]))
            else:
                avg_confidences.append(0)
        
        plt.bar(class_names, avg_confidences)
        plt.xlabel('类别')
        plt.ylabel('平均置信度')
        plt.title('各类别平均检测置信度')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, 'class_average_confidence.png'))
        plt.close()
    except Exception as e:
        print(f"创建统计图表时出错: {e}")
    
    print(f"\n推理完成! 结果保存在: {args.output}")
    print(f"检测可视化: {detection_vis_dir}")
    if args.save_txt:
        print(f"标签文件: {labels_dir}")
    if args.as_classifier:
        print(f"分类结果: {classify_dir}")
    print(f"摘要文件: {summary_file}")

if __name__ == "__main__":
    main()