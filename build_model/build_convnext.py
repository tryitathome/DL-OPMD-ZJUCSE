import os
import sys
sys.path.insert(0, os.getcwd())
from models.convnext import ConvNeXt


def build_convnext(convnext_name):
    convnext_config = None
    if convnext_name == 'convnext_t':
        convnext_config = {'in_chans': 3, 'num_classes': 2, 
                           'depths': [3, 3, 9, 3],
                           'dims'  : [96, 192, 384, 768],  'drop_path_rate' : 0., 
                           'layer_scale_init_value': 1e-6, 'head_init_scale': 1.}
    elif convnext_name == 'convnext_s':
        convnext_config = {'in_chans': 3, 'num_classes': 2, 
                           'depths': [3, 3, 27, 3],
                           'dims'  : [96, 192, 384, 768],  'drop_path_rate' : 0., 
                           'layer_scale_init_value': 1e-6, 'head_init_scale': 1.}
    elif convnext_name == 'convnext_b':
        convnext_config = {'in_chans': 3, 'num_classes': 1000, 
                           'depths': [3, 3, 27, 3],
                           'dims'  : [128, 256, 512, 1024], 'drop_path_rate' : 0., 
                           'layer_scale_init_value': 1e-6,  'head_init_scale': 1.}
    elif convnext_name == 'convnext_l':
        convnext_config = {'in_chans': 3, 'num_classes': 1000, 
                           'depths': [3, 3, 27, 3],
                           'dims'  : [192, 384, 768, 1536], 'drop_path_rate' : 0., 
                           'layer_scale_init_value': 1e-6,  'head_init_scale': 1.}
    elif convnext_name == 'convnext_xl':
        convnext_config = {'in_chans': 3, 'num_classes': 1000, 
                           'depths': [3, 3, 27, 3],
                           'dims'  : [256, 512, 1024, 2048], 'drop_path_rate' : 0., 
                           'layer_scale_init_value': 1e-6,   'head_init_scale': 1.}
    
    model = ConvNeXt(in_chans=convnext_config['in_chans'],
                     num_classes=convnext_config['num_classes'], 
                     depths=convnext_config['depths'],
                     dims=convnext_config['dims'],
                     drop_path_rate=convnext_config['drop_path_rate'],
                     layer_scale_init_value=convnext_config['layer_scale_init_value'],
                     head_init_scale=convnext_config['head_init_scale']
            )
    return model


'''
model = build_convnext('convnext_t')
import torch
tensor = torch.randn(32, 3, 224, 224)
output = model(tensor)
print(output.shape)
'''