import os
import sys
sys.path.insert(0, os.getcwd())
from models.replknet import RepLKNet


def build_replknet(replknet_name):
    replknet_config = None
    if replknet_name == 'replknet_31b':
        replknet_config = {'large_kernel_sizes': [31, 29, 27, 13], 'layers': [2, 2, 18, 2], 'channels': [128, 256, 512, 1024],
                           'drop_path_rate': 0.3, 'small_kernel': 5, 'dw_ratio': 1, 'num_classes': 2, 'use_checkpoint': False,
                           'small_kernel_merged': False}
    elif replknet_name == 'replknet_31l':
        replknet_config = {'large_kernel_sizes': [31, 29, 27, 13], 'layers': [2, 2, 18, 2], 'channels': [192, 384, 768, 1536],
                           'drop_path_rate': 0.3, 'small_kernel': 5, 'dw_ratio': 1, 'num_classes': 2, 'use_checkpoint': False,
                           'small_kernel_merged': False}
    elif replknet_name == 'replknet_xl':
        replknet_config = {'large_kernel_sizes': [27, 27, 27, 13], 'layers': [2, 2, 18, 2], 'channels': [256, 512, 1024, 2048],
                           'drop_path_rate': 0.3, 'small_kernel': 5, 'dw_ratio': 1.5, 'num_classes': 2, 'use_checkpoint': False,
                           'small_kernel_merged': False}
    
    model = RepLKNet(
        large_kernel_sizes=replknet_config['large_kernel_sizes'],
        layers=replknet_config['layers'],
        channels=replknet_config['channels'],
        drop_path_rate=replknet_config['drop_path_rate'],
        small_kernel=replknet_config['small_kernel'],
        dw_ratio=replknet_config['dw_ratio'],
        ffn_ratio=4, 
        in_channels=3, 
        num_classes=replknet_config['num_classes'], 
        out_indices=None,
        use_checkpoint=replknet_config['use_checkpoint'],
        small_kernel_merged=replknet_config['small_kernel_merged'],
        use_sync_bn=True,
        norm_intermediate_features=False 
    )
    
    return model
        
        
'''
if __name__ == '__main__':
    # model = create_RepLKNet31B(small_kernel_merged=False)
    model = build_replknet('replknet_31b')
    model.eval()
    print('------------------- training-time model -------------')
    print(model)
    import torch
    x = torch.randn(2, 3, 224, 224)
    origin_y = model(x)
    model.structural_reparam()
    print('------------------- after re-param -------------')
    print(model)
    reparam_y = model(x)
    print('------------------- the difference is ------------------------')
    print((origin_y - reparam_y).abs().sum())
'''
