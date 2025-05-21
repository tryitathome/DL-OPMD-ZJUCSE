import os
import random
import shutil
import yaml
from pathlib import Path

# 固定随机种子以确保可重现性
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# 数据集划分比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def create_yolo_dataset(source_images_dir, source_labels_dir, output_dir, class_names):
    """
    创建YOLO格式数据集并按比例划分
    
    参数:
        source_images_dir: 源图像目录
        source_labels_dir: 源标签目录
        output_dir: 输出目录
        class_names: 类别名称列表
    """
    # 创建目录结构
    dataset_dir = Path(output_dir)
    
    # 创建images和labels目录及其子目录
    for data_type in ['images', 'labels']:
        for split in ['train', 'val', 'test', 'all']:
            os.makedirs(dataset_dir / data_type / split, exist_ok=True)
    
    # 写入classes.txt
    with open(dataset_dir / 'classes.txt', 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(source_images_dir).glob(f"*{ext}")))
    
    # 随机打乱文件列表
    random.shuffle(image_files)
    
    # 计算每个集合的大小
    total_files = len(image_files)
    train_size = int(total_files * TRAIN_RATIO)
    val_size = int(total_files * VAL_RATIO)
    
    # 划分数据集
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size+val_size]
    test_files = image_files[train_size+val_size:]
    
    # 复制文件到相应目录
    for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        print(f"正在处理{split_name}集，共{len(files)}个文件")
        
        for img_path in files:
            # 图像文件名和扩展名
            img_filename = img_path.name
            img_stem = img_path.stem
            
            # 对应的标签文件
            label_path = Path(source_labels_dir) / f"{img_stem}.txt"
            
            if not label_path.exists():
                print(f"警告: 找不到图像 {img_filename} 对应的标签文件")
                continue
            
            # 复制到all目录
            shutil.copy(img_path, dataset_dir / 'images' / 'all' / img_filename)
            shutil.copy(label_path, dataset_dir / 'labels' / 'all' / label_path.name)
            
            # 复制到对应的分割目录
            shutil.copy(img_path, dataset_dir / 'images' / split_name / img_filename)
            shutil.copy(label_path, dataset_dir / 'labels' / split_name / label_path.name)
    
    # 创建data.yaml文件
    yaml_content = {
        'path': str(dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(dataset_dir / 'data.yaml', 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    print(f"数据集创建完成！")
    print(f"训练集: {len(train_files)}个文件")
    print(f"验证集: {len(val_files)}个文件")
    print(f"测试集: {len(test_files)}个文件")

if __name__ == "__main__":
    # 用户输入
    source_images_dir = r'D:\MyWorkSpace\LM数据集\yolo_dataset\train'#input("请输入源图像目录路径: ")
    source_labels_dir = r'D:\MyWorkSpace\LM数据集\yolo_dataset\labels'#input("请输入源标签目录路径: ")
    output_dir = r'D:\MyWorkSpace\YOLO12\yolo_dataset'#input("请输入输出目录路径 (默认为 'yolo_dataset'): ") or "yolo_dataset"
    
    # 获取类别名称
    classes_input = "OLK,OLP,OSF"#input("请输入类别名称，用逗号分隔 (如 person,car,dog): ")
    class_names = [c.strip() for c in classes_input.split(',')]
    
    create_yolo_dataset(source_images_dir, source_labels_dir, output_dir, class_names)
