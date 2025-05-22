import os
import shutil
import random

def split_data(source_dir, target_dir, train_size=0.7, val_size=0.15, test_size=0.15, seed=42):
    # 设置随机种子以确保结果可复现
    random.seed(seed)

    classes = [d.name for d in os.scandir(source_dir) if d.is_dir()]
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    test_dir = os.path.join(target_dir, 'test')

    # 创建训练集、验证集和测试集目录
    for dir in [train_dir, val_dir, test_dir]:
        for cls in classes:
            os.makedirs(os.path.join(dir, cls), exist_ok=True)

    # 划分数据集
    for cls in classes:
        class_dir = os.path.join(source_dir, cls)
        images = [img for img in os.listdir(class_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
        random.shuffle(images)  # 打乱数据

        total     = len(images)
        train_end = int(total * train_size)
        val_end   = train_end + int(total * val_size)

        train_data = images[:train_end]
        val_data   = images[train_end:val_end]
        test_data  = images[val_end:]

        # 将图片复制到相应的目录
        for data, folder in zip([train_data, val_data, test_data], [train_dir, val_dir, test_dir]):
            for img in data:
                src_path = os.path.join(source_dir, cls, img)
                dst_path = os.path.join(folder, cls, img)
                shutil.copy(src_path, dst_path)

source_dir  = 'data_third_balanced/'
dataset_dir = 'dataset_third_2class_balance/'
train_ratio = 0.6
val_ratio   = 0.2
test_ratio  = 0.2
random_seed = 0 
print('Start dataset split...')
split_data(source_dir, dataset_dir, train_ratio, val_ratio, test_ratio, random_seed)
print('Dataset split finshed.')
