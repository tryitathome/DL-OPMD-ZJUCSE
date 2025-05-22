from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import numpy as np

# 假设我们有一个训练集的 DataLoader
dataset = datasets.ImageFolder(root='path_to_your_dataset', transform=transforms.ToTensor())
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# 用于累积通道和
mean = 0.
std = 0.
nb_samples = 0.

for data in loader:
    images, _ = data
    batch_samples = images.size(0)  # 计算批量的图片数量
    images = images.view(batch_samples, images.size(1), -1)  # 将图片展平
    mean += images.mean(2).sum(0)  # 累加通道的均值
    std += images.std(2).sum(0)    # 累加通道的标准差
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(f'Mean: {mean}')
print(f'Std: {std}')
