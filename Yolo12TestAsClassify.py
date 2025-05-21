import os
import torch
import numpy as np
from ultralytics import YOLO
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import shutil
import yaml
from difflib import SequenceMatcher

# 配置参数
TEST_IMAGES_DIR = "./yolo_dataset_jiuyuan/images/train"
TEST_LABELS_DIR = "./yolo_dataset_jiuyuan/labels/train"
MODEL_PATH = "./best_155epoch_shengkouV2.pt"
DATA_YAML = "./yolo_dataset_jiuyuan/data.yaml"
RESULTS_DIR = "./evaluation_results"
CONF_THRES = 0.1
IOU_THRES = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_ground_truth_class(label_file):
    """从标签文件中读取第一个目标的类别"""
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            line = f.readline().strip()
            if line:
                class_id = int(float(line.split()[0]))
                return class_id
    return -1  # 如果没有找到标签

def calculate_string_similarity(str1, str2):
    """计算两个字符串的相似度"""
    return SequenceMatcher(None, str1, str2).ratio()

def create_class_mapping(model_names, dataset_names_dict, similarity_threshold=0.8):
    """
    创建类别名称之间的映射关系
    返回:
      - dataset_to_model: 数据集类别ID到模型类别ID的映射
      - model_to_dataset: 模型类别ID到数据集类别ID的映射
      - similarity_matrix: 类别名称相似度矩阵
    """
    dataset_to_model = {}  # 数据集类别ID -> 模型类别ID
    model_to_dataset = {}  # 模型类别ID -> 数据集类别ID
    
    # 创建类别名称相似度矩阵
    dataset_cls_count = len(dataset_names_dict)
    model_cls_count = len(model_names)
    similarity_matrix = np.zeros((dataset_cls_count, model_cls_count))
    
    print("\n构建类别名称映射关系...")
    print("使用相似度阈值:", similarity_threshold)
    
    # 计算所有类别名称之间的相似度
    for d_id, d_name in dataset_names_dict.items():
        for m_id, m_name in model_names.items():
            # 优先检查完全匹配
            if d_name.upper() == m_name.upper():
                similarity = 1.0
            else:
                # 计算字符串相似度
                similarity = calculate_string_similarity(d_name.upper(), m_name.upper())
                
                # 检查一个名称是否包含另一个(如果相似度不够高)
                if similarity < similarity_threshold:
                    if d_name.upper() in m_name.upper() or m_name.upper() in d_name.upper():
                        similarity = max(similarity, similarity_threshold)  # 至少达到阈值
            
            similarity_matrix[d_id, m_id] = similarity
    
    # 为每个数据集类别找到最匹配的模型类别
    matched_model_ids = set()  # 已匹配的模型类别ID
    for d_id in range(dataset_cls_count):
        max_similarity = 0
        best_m_id = None
        
        for m_id in range(model_cls_count):
            if similarity_matrix[d_id, m_id] > max_similarity:
                max_similarity = similarity_matrix[d_id, m_id]
                best_m_id = m_id
        
        # 如果相似度超过阈值，建立映射关系
        if max_similarity >= similarity_threshold and best_m_id is not None:
            dataset_to_model[d_id] = best_m_id
            # 如果已经存在对该模型类别的映射，保留相似度更高的
            if best_m_id in model_to_dataset:
                prev_d_id = model_to_dataset[best_m_id]
                if similarity_matrix[d_id, best_m_id] > similarity_matrix[prev_d_id, best_m_id]:
                    model_to_dataset[best_m_id] = d_id
            else:
                model_to_dataset[best_m_id] = d_id
            matched_model_ids.add(best_m_id)
    
    # 打印映射关系
    print("\n类别映射关系:")
    print("数据集类别ID -> 模型类别ID:")
    for d_id, m_id in dataset_to_model.items():
        print(f"  {d_id} ({dataset_names_dict[d_id]}) -> {m_id} ({model_names[m_id]}) [相似度: {similarity_matrix[d_id, m_id]:.2f}]")
    
    print("\n模型类别ID -> 数据集类别ID:")
    for m_id, d_id in model_to_dataset.items():
        print(f"  {m_id} ({model_names[m_id]}) -> {d_id} ({dataset_names_dict[d_id]}) [相似度: {similarity_matrix[d_id, m_id]:.2f}]")
    
    # 列出未匹配的类别
    unmatched_dataset = [d_id for d_id in dataset_names_dict.keys() if d_id not in dataset_to_model.keys()]
    unmatched_model = [m_id for m_id in model_names.keys() if m_id not in model_to_dataset.keys()]
    
    if unmatched_dataset:
        print("\n未找到匹配的数据集类别:")
        for d_id in unmatched_dataset:
            print(f"  {d_id}: {dataset_names_dict[d_id]}")
    
    if unmatched_model:
        print("\n未找到匹配的模型类别:")
        for m_id in unmatched_model:
            print(f"  {m_id}: {model_names[m_id]}")
    
    return dataset_to_model, model_to_dataset, similarity_matrix

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 创建正确和错误分类的文件夹
    correct_dir = os.path.join(RESULTS_DIR, "correct")
    incorrect_dir = os.path.join(RESULTS_DIR, "incorrect")
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(incorrect_dir, exist_ok=True)
    
    # 创建检测结果可视化文件夹
    detection_vis_dir = os.path.join(RESULTS_DIR, "detection_vis")
    os.makedirs(detection_vis_dir, exist_ok=True)
    
    # 加载模型
    model = YOLO(MODEL_PATH).to(DEVICE)
    
    # 加载数据集配置
    with open(DATA_YAML, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # 处理数据集类别名称可能是列表或字典的情况
    model_names = model.model.names  # 模型类别名称（字典）
    
    # 打印类别映射信息
    print("\n模型类别:")
    for cls_id, cls_name in model_names.items():
        print(f"  {cls_id}: {cls_name}")
    
    print("\n数据集类别:")
    dataset_names = data_config['names']
    # 检查dataset_names是列表还是字典
    if isinstance(dataset_names, list):
        # 如果是列表，转换为字典
        dataset_names_dict = {i: name for i, name in enumerate(dataset_names)}
        for cls_id, cls_name in dataset_names_dict.items():
            print(f"  {cls_id}: {cls_name}")
    else:
        # 如果是字典，直接使用
        dataset_names_dict = dataset_names
        for cls_id, cls_name in dataset_names_dict.items():
            print(f"  {cls_id}: {cls_name}")
    
    # 创建类别映射关系
    dataset_to_model, model_to_dataset, similarity_matrix = create_class_mapping(
        model_names, dataset_names_dict, similarity_threshold=0.7
    )
    
    # 手动进行分类评估
    # 获取所有测试图片
    test_images = glob(os.path.join(TEST_IMAGES_DIR, "*.jpg")) + \
                  glob(os.path.join(TEST_IMAGES_DIR, "*.jpeg")) + \
                  glob(os.path.join(TEST_IMAGES_DIR, "*.png"))
    
    # 初始化结果存储
    model_nc = len(model_names)  # 模型的类别数量
    dataset_nc = len(dataset_names_dict)  # 数据集的类别数量
    confusion_matrix = np.zeros((dataset_nc, model_nc), dtype=int)  # [真实类别, 预测类别]
    
    # 预测并评估每张图像
    predictions = []  # 预测类别ID
    ground_truths = []  # 真实类别ID
    name_matched = []  # 类别名称是否匹配
    
    # 为每个数据集类别创建文件夹
    for class_id, class_name in dataset_names_dict.items():
        os.makedirs(os.path.join(correct_dir, f"{class_id}_{class_name}"), exist_ok=True)
        os.makedirs(os.path.join(incorrect_dir, f"{class_id}_{class_name}"), exist_ok=True)
    
    print("\n开始分类评估...")
    print(f"找到 {len(test_images)} 个测试图片")
    
    for img_path in tqdm(test_images):
        # 获取对应标签文件路径
        img_basename = os.path.splitext(os.path.basename(img_path))[0]
        img_extension = os.path.splitext(os.path.basename(img_path))[1]
        label_file = os.path.join(TEST_LABELS_DIR, f"{img_basename}.txt")
        
        # 获取真实标签
        gt_class = get_ground_truth_class(label_file)
        if gt_class == -1 or gt_class >= dataset_nc:
            continue  # 跳过没有标签的图像或标签超出范围的图像
            
        # 使用模型预测
        results = model(img_path, conf=CONF_THRES)
        
        # 如果有检测结果，获取置信度最高的类别作为图片分类结果
        if len(results[0].boxes) > 0:
            # 获取所有检测框的置信度和类别
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            # 找到置信度最高的检测框
            max_conf_idx = np.argmax(confidences)
            pred_class = classes[max_conf_idx]
            max_confidence = confidences[max_conf_idx]
            
            # 检查预测类别是否在模型支持的范围内
            if pred_class >= model_nc:
                continue  # 跳过预测类别超出模型范围的情况
        else:
            # 如果没有检测到任何目标，分类结果为-1（或者可以设为其他合适的值）
            pred_class = -1
            max_confidence = 0.0
        
        # 更新混淆矩阵
        if gt_class != -1 and pred_class != -1:
            confusion_matrix[gt_class, pred_class] += 1
        
        # 保存预测结果和真实标签用于后续计算
        if pred_class != -1:
            predictions.append(pred_class)
            ground_truths.append(gt_class)
            
            # 检查类别名称是否匹配（通过映射）
            is_name_match = False
            
            # 检查数据集类别是否有对应的模型类别映射
            if gt_class in dataset_to_model:
                mapped_model_class = dataset_to_model[gt_class]
                is_name_match = (mapped_model_class == pred_class)
            
            name_matched.append(is_name_match)
            
            # 获取真实类别和预测类别的名称
            gt_class_name = dataset_names_dict[gt_class]
            pred_class_name = model_names[pred_class]
            
            # 根据名称匹配结果决定保存到哪个文件夹
            if is_name_match:  # 名称匹配正确
                dest_dir = os.path.join(correct_dir, f"{gt_class}_{gt_class_name}")
                dest_path = os.path.join(dest_dir, f"{img_basename}_conf{max_confidence:.2f}{img_extension}")
                shutil.copy2(img_path, dest_path)
            else:  # 名称匹配错误
                dest_dir = os.path.join(incorrect_dir, f"{gt_class}_{gt_class_name}")
                dest_path = os.path.join(dest_dir, f"{img_basename}_pred{pred_class}_{pred_class_name}_conf{max_confidence:.2f}{img_extension}")
                shutil.copy2(img_path, dest_path)
            
            # 保存检测框可视化结果
            # 获取原始图像
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
            
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
            
            # 绘制所有检测框
            for i in range(len(results[0].boxes)):
                box = results[0].boxes[i].xyxy.cpu().numpy()[0].astype(int)
                cls = int(results[0].boxes[i].cls.cpu().numpy()[0])
                conf = float(results[0].boxes[i].conf.cpu().numpy()[0])
                
                # 获取当前类别的颜色，如果类别超出预定义范围，则使用最后一个颜色
                color = colors.get(cls, colors.get(9))
                
                # 绘制边界框
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                
                # 添加类别名称和置信度，字体大小扩大4倍
                if 0 <= cls < model_nc:
                    # 如果有映射关系，同时显示数据集中的类别名称
                    if cls in model_to_dataset:
                        mapped_dataset_class = model_to_dataset[cls]
                        mapped_name = dataset_names_dict[mapped_dataset_class]
                        label = f"{model_names[cls]}({mapped_name}): {conf:.2f}"
                    else:
                        label = f"{model_names[cls]}: {conf:.2f}"
                else:
                    label = f"未知类别: {conf:.2f}"
                cv2.putText(img, label, (box[0], box[1] + 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 10)
            
            # 转回BGR格式用于保存
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # 保存带有检测框的图像
            vis_img_path = os.path.join(detection_vis_dir, f"{img_basename}_detection{img_extension}")
            cv2.imwrite(vis_img_path, img)
    
    # 计算分类准确率
    if len(predictions) > 0:
        # 严格匹配的准确率（类别ID必须完全一致）
        strict_correct = sum(p == gt for p, gt in zip(predictions, ground_truths))
        strict_accuracy = strict_correct / len(predictions)
        
        # 名称匹配的准确率（通过映射关系判断类别名称是否匹配）
        name_correct = sum(name_matched)
        name_accuracy = name_correct / len(predictions)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(12, 10))
        plt.imshow(confusion_matrix, cmap='Blues')
        plt.colorbar()
        plt.xlabel('预测类别 (模型)')
        plt.ylabel('真实类别 (数据集)')
        plt.title(f'分类混淆矩阵 (严格准确率: {strict_accuracy:.4f}, 名称准确率: {name_accuracy:.4f})')
        
        # 添加类别名称
        model_class_names = [f"{i}_{name}" for i, name in model_names.items()]
        dataset_class_names = [f"{i}_{name}" for i, name in dataset_names_dict.items()]
        plt.xticks(np.arange(model_nc), model_class_names, rotation=45, ha='right')
        plt.yticks(np.arange(dataset_nc), dataset_class_names)
        
        # 在每个单元格中添加数值
        for i in range(dataset_nc):
            for j in range(model_nc):
                # 如果在映射关系中，单元格标红
                if i in dataset_to_model and dataset_to_model[i] == j:
                    plt.text(j, i, str(confusion_matrix[i, j]),
                            ha="center", va="center", color="red", fontweight='bold')
                else:
                    plt.text(j, i, str(confusion_matrix[i, j]),
                            ha="center", va="center", color="black")
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "classification_confusion_matrix.png"))
        plt.close()
        
        print(f"\n严格匹配准确率: {strict_accuracy:.4f}")
        print(f"名称匹配准确率: {name_accuracy:.4f}")
        print(f"正确分类的图片已保存至: {correct_dir}")
        print(f"错误分类的图片已保存至: {incorrect_dir}")
        print(f"检测框可视化结果已保存至: {detection_vis_dir}")
    else:
        strict_accuracy = 0.0
        name_accuracy = 0.0
        print("\n没有有效的预测结果，无法计算准确率")
    
    # 保存详细结果
    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
        f.write(f"模型路径: {MODEL_PATH}\n")
        f.write(f"测试图像数: {len(test_images)}\n")
        f.write(f"有效预测数: {len(predictions)}\n\n")
        
        f.write("模型类别映射:\n")
        for i, name in model_names.items():
            f.write(f"  {i}: {name}\n")
        
        f.write("\n数据集类别映射:\n")
        for i, name in dataset_names_dict.items():
            f.write(f"  {i}: {name}\n")
        
        f.write("\n自动匹配的类别对应关系:\n")
        for d_id, m_id in dataset_to_model.items():
            d_name = dataset_names_dict[d_id]
            m_name = model_names[m_id]
            similarity = similarity_matrix[d_id, m_id]
            f.write(f"  数据集 {d_id} ({d_name}) -> 模型 {m_id} ({m_name}) [相似度: {similarity:.2f}]\n")
            
        f.write(f"\n严格匹配准确率: {strict_accuracy:.4f}\n")
        f.write(f"名称匹配准确率: {name_accuracy:.4f}\n")
        f.write("\n类别混淆矩阵 (行: 真实类别, 列: 预测类别):\n")
        for i in range(dataset_nc):
            for j in range(model_nc):
                f.write(f"{confusion_matrix[i, j]}\t")
            f.write("\n")

if __name__ == "__main__":
    main()
    print(f"\n评估结果已保存至：{RESULTS_DIR}")