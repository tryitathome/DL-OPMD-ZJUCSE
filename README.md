# Classify-LM-Simple-OralImages

口腔图像分类系统 - 基于深度学习的口腔病变图像分类项目

# Oral Image Classification System - Deep Learning-based Classification of Oral Lesions

## 项目简介 (Project Introduction)

该项目实现了一个用于口腔疾病图像分类的深度学习系统，支持多种现代化的深度神经网络架构，包括Swin Transformer、ConvNeXT、MetaFormer等。项目具有完整的训练、验证、测试和预测流程，适用于医学图像（尤其是口腔图像）的分类任务。本项目主要专注于口腔潜在恶性疾病(OPMD)与口腔良性疾病的二分类任务。

This project implements a deep learning system for oral disease image classification, supporting various modern deep neural network architectures including Swin Transformer, ConvNeXT, MetaFormer, etc. The project features complete training, validation, testing, and prediction workflows suitable for medical image classification tasks, particularly for oral images. The primary focus of this project is binary classification of Oral Potentially Malignant Disorders (OPMD) versus benign oral diseases.

## 环境配置

### Conda 环境安装

项目使用Conda管理环境依赖，可通过以下命令创建和配置环境：

```bash
conda env create -f environment.yml
```

主要依赖包括：
- PyTorch 2.1.0 (CUDA 12.1)
- torchvision 0.16.0
- timm 0.6.13
- numpy, pandas, matplotlib
- opencv-python
- apex (NVIDIA Apex 用于混合精度训练)

## 项目结构

```
Classify-LM-Simple-OralImages/
├── build_model/        # 模型构建模块
│   ├── build_convnext.py
│   ├── build_convnextv2.py
│   ├── build_metaformer.py
│   ├── build_my_metaformer.py
│   ├── build_replknet.py
│   └── build_swin.py
├── models/             # 模型定义
│   ├── convnext.py
│   ├── convnextv2.py
│   ├── densenet.py
│   ├── metaformer.py
│   ├── resnet18.py
│   ├── resnet50.py
│   ├── simple_cnn.py
│   ├── swin_transformer.py
│   ├── swin_transformer_v2.py
│   └── vit.py
├── utils/              # 工具函数
│   ├── build_loss.py
│   ├── build_optimizer.py
│   ├── eval_metrics.py
│   ├── final_test.py
│   ├── learning_rate.py
│   ├── load_data.py
│   ├── memmap.py
│   ├── resume.py
│   ├── train_process.py
│   └── train_vis.py
├── tools/              # 辅助工具
│   ├── cal_mean_std.py
│   ├── delete_macos_files.py
│   ├── folder_predict.py
│   ├── predict.py
│   ├── split_dataset.py
│   ├── test.py
│   └── test_catagory_save.py
├── train.py            # 主训练脚本
├── imagenet_pretrain.py # ImageNet预训练脚本
├── class_names.txt     # 类别名称列表
└── environment.yml     # 环境配置文件
```

## 数据准备

### 数据集结构

数据集应按以下结构组织：

```
dataset_directory/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── class2/
│       ├── image1.jpg
│       └── ...
├── val/
│   ├── class1/
│   │   └── ...
│   └── class2/
│       └── ...
└── test/
    ├── class1/
    │   └── ...
    └── class2/
        └── ...
```

### 数据集分割

可使用工具脚本将原始数据集分割为训练/验证/测试集：

```bash
python tools/split_dataset.py
```

在使用前，您需要在脚本中配置以下参数：
- `source_dir`: 原始数据目录路径
- `dataset_dir`: 输出数据集目录路径
- `train_ratio`: 训练集比例（默认0.6）
- `val_ratio`: 验证集比例（默认0.2）
- `test_ratio`: 测试集比例（默认0.2）

## 模型训练

### 基础训练

执行以下命令开始训练：

```bash
python train.py
```

主要训练参数在`train.py`脚本开头设置，包括：

- 基本参数：
  - `num_epochs`: 训练轮数
  - `batch_size`: 批次大小
  - `image_size`: 图像输入尺寸
  - `dataset_dir`: 数据集目录
  - `output_dir`: 输出目录

- 学习率参数：
  - `init_lr`: 初始学习率
  - `min_lr`: 最小学习率
  - `warm_epochs`: 预热轮数
  - `warm_init_lr`: 预热初始学习率

- 数据增强参数：
  - `color_jitter`: 颜色抖动
  - `auto_augment`: 自动增强策略
  - `mixup`: 是否使用mixup增强

### ImageNet预训练

对于大规模数据集预训练，可使用：

```bash
python imagenet_pretrain.py
```

## 模型评估与测试

### 模型测试

使用以下命令对模型进行测试：

```bash
python tools/test.py
```

测试结果将保存在`test_results/`目录下，包括准确率、精确度、召回率和F1分数。

### 类别分析测试

使用以下命令进行按类别的详细分析测试：

```bash
python tools/test_catagory_save.py
```

此脚本会按照类别保存预测结果，并生成详细的混淆矩阵。

## 模型预测

### 单张图像预测

使用以下命令对单张图像进行预测：

```bash
python tools/predict.py
```

在使用前，需要配置以下参数：
- `weights_path`: 模型权重路径
- `image_path`: 待预测图像路径

### 文件夹批量预测

使用以下命令对整个文件夹的图像进行批量预测：

```bash
python tools/folder_predict.py
```

在使用前，需要配置以下参数：
- `weights_path`: 模型权重路径
- `folder_path`: 待预测图像文件夹路径
- `csv_path`: 预测结果保存路径

预测结果将保存为CSV文件，包含图像名称、预测类别和置信度。

## 支持的模型架构

该项目支持多种先进的深度学习模型架构：

1. **Swin Transformer系列**
   - Swin-T/S/B/L (原始版本)
   - Swin-V2-T/S/B/L (V2版本)

2. **ConvNeXT系列**
   - ConvNeXT-T/S/B/L/XL
   - ConvNeXTv2-T/S/B/L/XL

3. **MetaFormer系列**
   - CAFormer
   - ConvFormer
   - IdentityFormer
   - PoolFormer

4. **其他经典模型**
   - ResNet18/50
   - DenseNet121
   - ViT (Vision Transformer)
   - ReplKNet

## 高级特性

### 混合精度训练

项目支持通过Nvidia Apex进行混合精度训练，可以设置不同的优化级别：

```python
amp_level = "O1"  # 可选 "O0", "O1", "O2", "O3"
```

### 内存映射加速

使用内存映射(memmap)加速数据加载：

```python
use_memmap = True
memmap_dir = 'memmap_cache'
```

### Mixup数据增强

支持Mixup/Cutmix数据增强技术：

```python
mixup = True
mixup_ratio = 0.8
cutmix = True
cutmix_ratio = 1.0
```

## 注意事项

1. 若使用混合精度训练，请确保已正确安装Nvidia Apex。
2. 对于大型模型，可能需要调整批次大小或使用梯度累积技术。
3. 在`class_names.txt`文件中定义类别名称，每行一个类别。

## 数据集可用性 (Dataset Availability)

本研究中生成或分析的数据集，为保护患者隐私，不公开发布，但可应合理要求向通讯作者索取。

The datasets generated or analyzed during this study are not publicly available due to patient privacy concerns but are available from the corresponding authors upon reasonable request.

## 预训练文件说明 (Pre-trained Files Notes)

由于体积限制，本项目仅提供一个预训练模型文件作为示例，但用户可以根据需要使用其他网络架构进行预训练，并按照项目中的方法加载预训练权重。

Due to size constraints, this project only provides one pre-trained model file as an example. However, users can perform pre-training using other network architectures and load the pre-trained weights following the methods described in this project.

## 语言说明 (Language Notes)

本项目的代码注释大部分为中文，其他语言使用者可能需要自行翻译。

Most code comments in this project are in Chinese. Users of other languages may need to translate them as needed.


## 联系方式 (Contact Information)

### 通讯作者 (Corresponding Authors)

**L.J. Shi**  
Department of Oral Medicine, Shanghai Ninth People's Hospital, College of Stomatology, Shanghai Jiao Tong University, National Center for Stomatology, National Clinical Research Center for Oral Diseases, Shanghai Key Laboratory of Stomatology, Shanghai Research Institute of Stomatology, 639 Zhizaoju Road, Huangpu District, Shanghai, China  
Email: drshilinjun@126.com

**H.X. Dan**  
State Key Laboratory of Oral Diseases & National Center for Stomatology & National Clinical Research Center for Oral Diseases & Research Unit of Oral Carcinogenesis and Management, West China Hospital of Stomatology, Sichuan University, Chengdu, No. 14, Section 3, Ren Min Nan Road, Chengdu, Sichuan 610041, China  
Email: hxdan@foxmail.com

**F.D. Zhu**  
Stomatology Hospital, School of Stomatology, Zhejiang University School of Medicine, Zhejiang Provincial Clinical Research Center for Oral Diseases, Key Laboratory of Oral Biomedical Research of Zhejiang Province, Cancer Center of Zhejiang University, Engineering Research Center of Oral Biomaterials and Devices of Zhejiang Province, 166 Qiu Tao Bei Road, Hangzhou, 310000, China  
Email: zfd@zju.edu.cn
