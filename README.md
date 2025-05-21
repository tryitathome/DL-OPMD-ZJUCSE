# DL-OPMD-JDR
Code backup for project“A Deep Learning Method for Diagnosis of Oral Mucosa Potential Malignant Disorders”
# YOLO12-simplified

这个项目是基于Ultralytics YOLOv12框架开发的目标检测解决方案，主要用于牙科图像中的口腔病变检测。系统支持训练、测试、推理和结果可视化等功能。
本研究中生成或分析的数据集，为保护患者隐私，不公开发布，但可应合理要求向通讯作者索取。

通讯作者：
L.J. Shi, Department of Oral Medicine, Shanghai Ninth People's Hospital, College of Stomatology, Shanghai Jiao Tong University, National Center for Stomatology, National Clinical Research Center for Oral Diseases, Shanghai Key Laboratory of Stomatology, Shanghai Research Institute of Stomatology, 639 Zhizaoju Road, Huangpu District, Shanghai, China.
Email: drshilinjun@126.com
H.X. Dan, State Key Laboratory of Oral Diseases & National Center for Stomatology & National Clinical Research Center for Oral Diseases & Research Unit of Oral Carcinogenesis and Management, West China Hospital of Stomatology, Sichuan University, Chengdu, No. 14, Section 3, Ren Min Nan Road, Chengdu, Sichuan 610041, China.
Email: hxdan@foxmail.com
F.D. Zhu, Stomatology Hospital, School of Stomatology, Zhejiang University School of Medicine, Zhejiang Provincial Clinical Research Center for Oral Diseases, Key Laboratory of Oral Biomedical Research of Zhejiang Province, Cancer Center of Zhejiang University, Engineering Research Center of Oral Biomaterials and Devices of Zhejiang Province, 166 Qiu Tao Bei Road, Hangzhou, 310000, China
Email: zfd@zju.edu.cn

## 环境配置

### 使用Conda安装环境

项目提供了完整的环境配置文件`environment.yml`，可以通过以下命令创建并激活环境：

```bash
# 创建环境
conda env create -f Readme/environment.yml

# 激活环境
conda activate base
```

### 主要依赖库

- ultralytics (YOLOv12)
- pytorch
- cuda
- opencv-python
- numpy
- matplotlib
- pandas

## 数据集结构

数据集采用YOLOv12标准格式组织，主要包含以下内容：

```
yolo_dataset_shengkouGen2/
├── classes.txt          # 类别名称文件
├── data.yaml            # 数据集配置文件
├── images/              # 图像文件夹
│   ├── train/           # 训练集图像
│   ├── val/             # 验证集图像
│   └── test/            # 测试集图像
└── labels/              # 标注文件夹
    ├── train/           # 训练集标注
    ├── val/             # 验证集标注
    └── test/            # 测试集标注
```

### 类别信息

当前版本支持以下3个类别：
- OLK：口腔扁平苔藓
- OLP：口腔苔藓样病变
- OSF：口腔粘膜下纤维化

## 主要功能文件说明

### 数据处理

- **DatasetSplit.py**: 数据集划分工具，用于将图像和标签按比例划分为训练集、验证集和测试集
  ```bash
  python DatasetSplit.py
  ```

- **CocoYoloConverter.py**: COCO格式转YOLOv12格式工具
  ```bash
  python CocoYoloConverter.py
  ```

### 模型训练与评估

- **Yolo12Train.py**: YOLOv12模型训练脚本
  ```bash
  python Yolo12Train.py
  ```

- **Yolo12Train_Visual.py**: 带可视化的YOLOv12模型训练脚本
  ```bash
  python Yolo12Train_Visual.py
  ```

- **Yolo12Test.py**: 模型测试评估脚本，生成评估报告和性能指标
  ```bash
  python Yolo12Test.py
  ```

- **Yolo12TestAsClassify.py**: 将检测模型作为分类器使用的测试脚本
  ```bash
  python Yolo12TestAsClassify.py
  ```

### 推理应用

- **Yolo12Inference.py**: 模型推理脚本，用于对新图像进行预测
  ```bash
  python Yolo12Inference.py --model best_155epoch_shengkouV2.pt --source MiniTestData --output MiniInferenceResults --conf 0.5 --line-thickness 10
  ```

  参数说明:
  - `--model`: 模型文件路径，默认为`./best_155epoch_shengkouV2.pt`
  - `--source`: 输入图像文件夹路径，默认为`./inference_images`
  - `--output`: 输出文件夹路径，默认为`./inference_results`
  - `--conf`: 置信度阈值，默认为0.25
  - `--iou`: NMS IoU阈值，默认为0.45
  - `--device`: 设备选择 (cuda或cpu)
  - `--save-txt`: 保存标签文件
  - `--save-conf`: 在标签文件中保存置信度
  - `--classes`: 仅检测指定类别
  - `--max-det`: 每张图像的最大检测数，默认为300
  - `--line-thickness`: 边界框线条粗细，默认为2
  - `--as-classifier`: 将检测器作为分类器使用，保存按类别分类的图片

## 预训练模型

项目提供了几个预训练模型:
- `best_155epoch_shengkouV2.pt`: 在155轮训练后的最佳模型，用于牙科口腔病变检测
- `best_120epoch.pt`: 120轮训练后的模型
- `yolo11n.pt`: YOLOv11 nano版预训练模型
- `yolo12s.pt`: YOLOv12 small版预训练模型
- `yolo12m.pt`: YOLOv12 medium版预训练模型

## 输出结果

推理后，结果会保存在指定的输出目录中，包括：
- 检测可视化图像（带有边界框和标签）
- 检测结果摘要
- 分类统计（当使用`--as-classifier`时）
- 各类别检测数量统计图
- 各类别平均置信度统计图

## 示例

检测结果示例存储在`MiniInferenceResults`目录下，包含:
- 类别平均置信度图表
- 类别检测数量图表
- 检测结果可视化图像
- 推理结果摘要

## 注意事项

* 本项目的代码注释和提示信息主要使用中文编写，非中文用户在使用时可能需要借助翻译工具。
* 在使用前请确保您的环境已正确配置，特别是GPU和CUDA依赖。
* 对于大型数据集，建议调整批处理大小以适应您的硬件配置。

# YOLO12-simplified (English Version)

This project is an object detection solution based on the Ultralytics YOLOv12 framework, primarily designed for detecting oral lesions in dental images. The system supports training, testing, inference, and result visualization functionalities.

The data sets generated or analyzed during the current study are not publicly available in order to preserve patient confidentiality but are available from the corresponding authors on reasonable request.

L.J. Shi, Department of Oral Medicine, Shanghai Ninth People's Hospital, College of Stomatology, Shanghai Jiao Tong University, National Center for Stomatology, National Clinical Research Center for Oral Diseases, Shanghai Key Laboratory of Stomatology, Shanghai Research Institute of Stomatology, 639 Zhizaoju Road, Huangpu District, Shanghai, China.
Email: drshilinjun@126.com
H.X. Dan, State Key Laboratory of Oral Diseases & National Center for Stomatology & National Clinical Research Center for Oral Diseases & Research Unit of Oral Carcinogenesis and Management, West China Hospital of Stomatology, Sichuan University, Chengdu, No. 14, Section 3, Ren Min Nan Road, Chengdu, Sichuan 610041, China.
Email: hxdan@foxmail.com
F.D. Zhu, Stomatology Hospital, School of Stomatology, Zhejiang University School of Medicine, Zhejiang Provincial Clinical Research Center for Oral Diseases, Key Laboratory of Oral Biomedical Research of Zhejiang Province, Cancer Center of Zhejiang University, Engineering Research Center of Oral Biomaterials and Devices of Zhejiang Province, 166 Qiu Tao Bei Road, Hangzhou, 310000, China
Email: zfd@zju.edu.cn

## Environment Setup

### Installing with Conda

The project provides a complete environment configuration file `environment.yml`, which can be set up using the following commands:

```bash
# Create the environment
conda env create -f Readme/environment.yml

# Activate the environment
conda activate base
```

### Main Dependencies

- ultralytics (YOLOv12)
- pytorch
- cuda
- opencv-python
- numpy
- matplotlib
- pandas

## Dataset Structure

The dataset follows the YOLOv12 standard format:

```
yolo_dataset_shengkouGen2/
├── classes.txt          # Class names file
├── data.yaml            # Dataset configuration file
├── images/              # Images folder
│   ├── train/           # Training images
│   ├── val/             # Validation images
│   └── test/            # Test images
└── labels/              # Labels folder
    ├── train/           # Training labels
    ├── val/             # Validation labels
    └── test/            # Test labels
```

### Class Information

The current version supports the following 3 classes:
- OLK: Oral Lichen Planus
- OLP: Oral Lichenoid Lesions
- OSF: Oral Submucous Fibrosis

## Main Features

### Data Processing

- **DatasetSplit.py**: Dataset splitting tool for dividing images and labels into training, validation, and test sets
  ```bash
  python DatasetSplit.py
  ```

- **CocoYoloConverter.py**: COCO format to YOLOv12 format conversion tool
  ```bash
  python CocoYoloConverter.py
  ```

### Model Training and Evaluation

- **Yolo12Train.py**: YOLOv12 model training script
  ```bash
  python Yolo12Train.py
  ```

- **Yolo12Train_Visual.py**: YOLOv12 model training script with visualization
  ```bash
  python Yolo12Train_Visual.py
  ```

- **Yolo12Test.py**: Model testing and evaluation script that generates evaluation reports and performance metrics
  ```bash
  python Yolo12Test.py
  ```

- **Yolo12TestAsClassify.py**: Script for using the detection model as a classifier
  ```bash
  python Yolo12TestAsClassify.py
  ```

### Inference Application

- **Yolo12Inference.py**: Model inference script for making predictions on new images
  ```bash
  python Yolo12Inference.py --model best_155epoch_shengkouV2.pt --source MiniTestData --output MiniInferenceResults --conf 0.5 --line-thickness 10
  ```

  Parameters:
  - `--model`: Model file path, default is `./best_155epoch_shengkouV2.pt`
  - `--source`: Input image folder path, default is `./inference_images`
  - `--output`: Output folder path, default is `./inference_results`
  - `--conf`: Confidence threshold, default is 0.25
  - `--iou`: NMS IoU threshold, default is 0.45
  - `--device`: Device selection (cuda or cpu)
  - `--save-txt`: Save label files
  - `--save-conf`: Save confidence in label files
  - `--classes`: Only detect specified classes
  - `--max-det`: Maximum number of detections per image, default is 300
  - `--line-thickness`: Boundary box line thickness, default is 2
  - `--as-classifier`: Use the detector as a classifier, saving images categorized by class

## Pre-trained Models

The project provides several pre-trained models:
- `best_155epoch_shengkouV2.pt`: Best model after 155 epochs of training, for dental oral lesion detection
- `best_120epoch.pt`: Model after 120 epochs of training
- `yolo11n.pt`: YOLOv11 nano version pre-trained model
- `yolo12s.pt`: YOLOv12 small version pre-trained model
- `yolo12m.pt`: YOLOv12 medium version pre-trained model

## Output Results

After inference, results will be saved in the specified output directory, including:
- Detection visualization images (with bounding boxes and labels)
- Detection result summary
- Classification statistics (when using `--as-classifier`)
- Detection count statistics chart by class
- Average confidence statistics chart by class

## Example

Example detection results are stored in the `MiniInferenceResults` directory, containing:
- Average confidence charts by class
- Detection count charts by class
- Detection visualization images
- Inference result summary

## Note

* The code comments and prompt information in this project are mainly written in Chinese. Non-Chinese users may need to use translation tools.
* Please ensure your environment is properly configured before use, especially GPU and CUDA dependencies.
* For large datasets, it is recommended to adjust the batch size to fit your hardware configuration.

