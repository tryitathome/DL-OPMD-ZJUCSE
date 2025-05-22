import torch
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets


from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import json

from timm.data import create_transform
from timm.data.transforms import str_to_pil_interp
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .memmap import create_memmap_file


'''用于memmap的ImageFolder'''
class CustomImageFolder(Dataset):
    def __init__(self, memmap_file):
        self.data = np.memmap(memmap_file, dtype=np.dtype([('image', np.uint8, (3, 224, 224)), ('label', np.int64)]), mode='r')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        item = self.data[idx]
        image = torch.tensor(item['image'], dtype=torch.float32).div(255)  
        label = item['label']
        return image, label
        '''
        try:
            item = self.data[idx]
            image = Image.fromarray(item['image'].transpose(1, 2, 0))
            label = item['label']
            return image, label
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return None


'''用于memmap的DataLoader自定义Collate_fn'''
class CustomCollateFunction:
    def __init__(self, img_size, is_train, 
                    color_jitter, auto_augment, re_prob, re_mode, re_count, 
                    interpolation, test_crop):
        self.img_size  = img_size
        self.is_train  = is_train
        self.transform = build_transform(is_train, img_size, 
                                         color_jitter, auto_augment, re_prob, re_mode, re_count, 
                                         interpolation, test_crop)
        # print(self.transform)
        '''
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(90),
                transforms.Resize((img_size, img_size)),
                # transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                # transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        '''

    def __call__(self, batch):
        images, labels = zip(*batch)
        transformed_images = [self.transform(img) for img in images]
        images_tensor = torch.stack(transformed_images)
        labels_tensor = torch.tensor(labels)
        return images_tensor, labels_tensor
    

def create_tensor_dataset(dataset):
    data, labels = [], []

    for img, label in tqdm(dataset):
        data.append(img)
        labels.append(label)

    # 将数据和标签转换为张量
    data   = torch.stack(data)
    labels = torch.tensor(labels)

    # 创建 TensorDataset
    tensor_dataset = TensorDataset(data, labels)
    return tensor_dataset


def load_dataset_to_memory(data_loader, load_process_name):
    # 将所有数据和标签存储在列表中
    data, labels = [], []
    for images, targets in tqdm(data_loader, desc=load_process_name):
        data.append(images)
        labels.append(targets)
    
    # 将列表转换为tensor，这样可以直接在内存中操作
    data = torch.cat(data, dim=0)  # 沿着第一个维度（batch维度）拼接
    labels = torch.cat(labels, dim=0)
    return data, labels


def build_transform(is_train, image_size, 
                    color_jitter, auto_augment, re_prob, re_mode, re_count, interpolation,
                    test_crop):
    # 是否需要对图像进行resize
    resize = image_size > 32
    # 训练所需的变换
    if is_train:
        transform = create_transform(
            input_size=image_size,
            is_training=True,
            color_jitter=color_jitter if color_jitter > 0 else None,
            auto_augment=auto_augment if auto_augment != 'none' else None,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            interpolation=interpolation,
        )
        if not resize:
            transform.transforms[0] = transforms.RandomCrop(image_size, padding=4)
            transform.transforms.insert(1, transforms.CenterCrop(size=(224, 224)))
            transform.transforms.insert(2, transforms.RandomGrayscale(p=0.5))
        return transform

    else:
        transforms_list = []
        if resize:
            if test_crop:
                size = int((256 / 224) * image_size)
                transforms_list.append(
                    transforms.Resize(size, interpolation=str_to_pil_interp(interpolation)),
                    # to maintain same ratio w.r.t. 224 images
                )
                transforms_list.append(transforms.CenterCrop(image_size))
            else:
                transforms_list.append(
                    transforms.Resize((image_size, image_size),
                                    interpolation=str_to_pil_interp(interpolation))
                )
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize(0.7717, 0.4983, 0.4611))
    return transforms.Compose(transforms_list)


'''使用memmap建立dataloader'''
def build_loader_memmap(memmap_dir, dataset_dir, image_size, batch_size, num_workers, pin_memory, persistent_workers, color_jitter, auto_augment, re_prob, re_mode, re_count, interpolation, test_crop):
    os.makedirs(memmap_dir, exist_ok=True)
    print('Creating memmap...')

    create_memmap_file(f'./{dataset_dir}/train', f'./{memmap_dir}/{dataset_dir}_train.dat', (3, 224, 224), description='train')
    train_dataset = CustomImageFolder(f'./{memmap_dir}/{dataset_dir}_train.dat')
    print('Training dataset created!')

    create_memmap_file(f'./{dataset_dir}/val', f'./{memmap_dir}/{dataset_dir}_val.dat', (3, 224, 224), description='val')
    val_dataset = CustomImageFolder(f'./{memmap_dir}/{dataset_dir}_val.dat')
    print('Validation dataset created!')

    create_memmap_file(f'./{dataset_dir}/test', f'./{memmap_dir}/{dataset_dir}_test.dat', (3, 224, 224), description='test')
    test_dataset = CustomImageFolder(f'./{memmap_dir}/{dataset_dir}_test.dat')
    print('Test dataset created!')
    print('Finished memmap...')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  
                              collate_fn=CustomCollateFunction(image_size, True, color_jitter, auto_augment, 
                                                               re_prob, re_mode, re_count, interpolation, test_crop),
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True, persistent_workers=persistent_workers)
    
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                              collate_fn=CustomCollateFunction(image_size, False, color_jitter, auto_augment, 
                                                               re_prob, re_mode, re_count, interpolation, test_crop),
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=False, persistent_workers=persistent_workers)
    
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                              collate_fn=CustomCollateFunction(image_size, False, color_jitter, auto_augment, 
                                                               re_prob, re_mode, re_count, interpolation, test_crop),
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=False, persistent_workers=persistent_workers)
    
    print(f'Training   Dataset Size: {len(train_dataset)}')
    print(f'Validation Dataset Size: {len(val_dataset)}')
    print(f'Testing    Dataset Size: {len(test_dataset)}')
    
    return train_loader, val_loader, test_loader


'''常规建立dataloader'''
def build_loader(dataset_dir, image_size, batch_size, load_into_memory, num_workers, pin_memory, persistent_workers, color_jitter, auto_augment, re_prob, re_mode, re_count, interpolation, test_crop):
    is_train = True
    transform_train = build_transform(is_train, image_size, 
                                      color_jitter, auto_augment, re_prob, re_mode, re_count, 
                                      interpolation, test_crop)
    is_train = False
    transform_val   = build_transform(is_train, image_size, 
                                      color_jitter, auto_augment, re_prob, re_mode, re_count, 
                                      interpolation, test_crop)
    # 加载数据集
    train_dataset = datasets.ImageFolder(root=f'./{dataset_dir}/train', transform=transform_train)
    val_dataset   = datasets.ImageFolder(root=f'./{dataset_dir}/val',   transform=transform_val)
    test_dataset  = datasets.ImageFolder(root=f'./{dataset_dir}/test',  transform=transform_val)

    if load_into_memory:
        # 加载整个数据集到内存
        print('Put all data into memory...')
        train_dataset_tensor = create_tensor_dataset(train_dataset)
        val_dataset_tensor   = create_tensor_dataset(val_dataset)
        test_dataset_tensor  = create_tensor_dataset(test_dataset)

        train_loader = DataLoader(train_dataset_tensor, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory, drop_last=True,  persistent_workers=persistent_workers)
        val_loader   = DataLoader(val_dataset_tensor,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, persistent_workers=persistent_workers)
        test_loader  = DataLoader(test_dataset_tensor,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, persistent_workers=persistent_workers)
        print('All data has been put into memory')
        
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory, drop_last=True,  persistent_workers=persistent_workers)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, persistent_workers=persistent_workers)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, persistent_workers=persistent_workers)

    print(f'Training   Dataset Size: {len(train_dataset)}')
    print(f'Validation Dataset Size: {len(val_dataset)}')
    print(f'Testing    Dataset Size: {len(test_dataset)}')
    
    return train_loader, val_loader, test_loader


'''ImageNet1k 预训练loader'''
def build_loader_imagenet1k(dataset_dir, image_size, batch_size, load_into_memory, num_workers, pin_memory, persistent_workers, color_jitter, auto_augment, re_prob, re_mode, re_count, interpolation, test_crop):
    is_train = True
    transform_train = build_transform(is_train, image_size, 
                                      color_jitter, auto_augment, re_prob, re_mode, re_count, 
                                      interpolation, test_crop)
    is_train = False
    transform_val   = build_transform(is_train, image_size, 
                                      color_jitter, auto_augment, re_prob, re_mode, re_count, 
                                      interpolation, test_crop)
    # 加载数据集
    train_dataset = datasets.ImageFolder(root=f'./{dataset_dir}/train', transform=transform_train)
    val_dataset   = datasets.ImageFolder(root=f'./{dataset_dir}/val',   transform=transform_val)
    test_dataset  = datasets.ImageFolder(root=f'./{dataset_dir}/val',  transform=transform_val)

    if load_into_memory:
        # 加载整个数据集到内存
        print('Put all data into memory...')
        train_dataset_tensor = create_tensor_dataset(train_dataset)
        val_dataset_tensor   = create_tensor_dataset(val_dataset)
        test_dataset_tensor  = create_tensor_dataset(test_dataset)

        train_loader = DataLoader(train_dataset_tensor, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory, drop_last=True,  persistent_workers=persistent_workers)
        val_loader   = DataLoader(val_dataset_tensor,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, persistent_workers=persistent_workers)
        test_loader  = DataLoader(test_dataset_tensor,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, persistent_workers=persistent_workers)
        print('All data has been put into memory')
        
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory, drop_last=True,  persistent_workers=persistent_workers)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, persistent_workers=persistent_workers)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, persistent_workers=persistent_workers)

    print(f'Training   Dataset Size: {len(train_dataset)}')
    print(f'Validation Dataset Size: {len(val_dataset)}')
    print(f'Testing    Dataset Size: {len(test_dataset)}')
    
    return train_loader, val_loader, test_loader