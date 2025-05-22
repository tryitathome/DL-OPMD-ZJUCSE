import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.getcwd())
from models.my_metaformer_mae import MaskedMetaformer, MAEDecoder, caformer_s18_eva02_encoder, mae_loss

# 假设您已经定义了 MaskedMetaformer 和 MAEDecoder 类

def visualize_reconstruction(model, decoder, image_path, device='cuda'):
    # 加载图像并进行预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 模型推理
    model.eval()
    decoder.eval()
    with torch.no_grad():
        features, mask, ids_restore = model(image_tensor)
        reconstructed = decoder(features, ids_restore)

    # 反归一化
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        return tensor * std + mean

    # 准备可视化
    original = denormalize(image_tensor)
    masked = denormalize(image_tensor * (1 - mask))
    reconstructed = denormalize(reconstructed)

    # 创建网格
    grid = make_grid([original[0], masked[0], reconstructed[0]], nrow=3)
    grid = grid.cpu().permute(1, 2, 0).numpy()

    # 显示图像
    plt.figure(figsize=(15, 5))
    plt.imshow(grid)
    plt.axis('off')
    plt.title('Original - Masked - Reconstructed')
    plt.show()

# 加载预训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


original_metaformer = caformer_s18_eva02_encoder(pretrained=False)
model = MaskedMetaformer(original_metaformer).to(device)
decoder = MAEDecoder(latent_dim=1024, image_size=224, patch_size=16, in_channels=3).to(device)

# 加载预训练权重
checkpoint_encoder = torch.load('output/mae_test/checkpoints/encoder_epoch_16.pth', map_location=device)
checkpoint_decoder = torch.load('output/mae_test/checkpoints/decoder_epoch_16.pth', map_location=device)

model.load_state_dict(checkpoint_encoder['model_state_dict'], strict=True)
decoder.load_state_dict(checkpoint_decoder['model_state_dict'], strict=True)

# 进行可视化
image_path = 'dataset_third_2class_balance/train/OPMD/DSC_0013.jpg'
visualize_reconstruction(model, decoder, image_path, device)