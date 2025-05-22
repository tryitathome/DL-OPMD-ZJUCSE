import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os
sys.path.insert(0, os.getcwd())

from models.simple_cnn import SimpleCNN
from models.resnet18 import ResNet18
from models.swin_transformer import SwinTransformer



def load_class_names(names_path):
    with open(names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names


def predict_image(image_path, weights_path, model, image_size, class_names, folder_pred=False):
    # 载入模型
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()  

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 或者根据模型训练时的设置
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if folder_pred == True:
        image = image_path
    else:
        image = Image.open(image_path).convert('RGB')  # 转换为RGB    
    image = transform(image).unsqueeze(0)              # 增加批次维度

    # 预测
    with torch.no_grad():  # 关闭梯度计算
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        predicted_class = class_names[predicted.item()]
        return predicted_class

if __name__ == '__main__':
    '''单张预测'''
    image_size = 224
    # 分类类别
    num_classes = 2
    # 权重路径
    weights_path = r'E:\MOUTH\basic_classify_model\my_classify_model\output\balanced_swin\checkpoints\best_model.pth'  
    # 预测图片路径
    image_path = r'E:\MOUTH\basic_classify_model\my_classify_model\data_tiny\Benign\DSC_0156.jpg'  
    # 类别名称路径
    class_names_path = 'class_names.txt'

    # 模型
    model = SwinTransformer(img_size=image_size,
                            patch_size=4,
                            in_chans=3,
                            num_classes=num_classes,
                            embed_dim=128,
                            depths=[2, 2, 18, 2],
                            num_heads=[4, 8, 16, 32],
                            window_size=7,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.0,
                            drop_path_rate=0.1,
                            ape=False,
                            patch_norm=True,
                            use_checkpoint=False)  
    # 类别名称映射txt
    class_names = load_class_names(class_names_path)  
    # 进行预测
    prediction = predict_image(image_path, weights_path, model, class_names)
    print(f'{image_path} predicted class: {prediction}')
