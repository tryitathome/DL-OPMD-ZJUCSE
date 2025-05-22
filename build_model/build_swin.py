import os
import sys
sys.path.insert(0, os.getcwd())
from models.swin_transformer import SwinTransformer
from models.swin_transformer_v2 import SwinTransformerV2


def build_swin_transformer(swin_name):
    if swin_name == 'swin_tiny':
        swin_transformer_config = {'image_size': 224,   'patch_size': 4,             'in_chans' : 3,              'num_classes'   : 2,
                                   'embed_dim' : 96,    'depths'    : [2, 2, 6, 2],  'num_heads': [3, 6, 12, 24], 'window_size'   : 7,
                                   'mlp_ratio' : 4.,    'qkv_bias'  : True,          'qk_scale' : None,           'drop_rate'     : 0.0, 
                                   'ape'       : False, 'patch_norm': True,          'use_checkpoint': False,     'drop_path_rate': 0.2} 
    
    elif swin_name == 'swin_small':
        swin_transformer_config = {'image_size': 224,   'patch_size': 4,             'in_chans' : 3,              'num_classes'   : 2,
                                   'embed_dim' : 96,    'depths'    : [2, 2, 18, 2], 'num_heads': [3, 6, 12, 24], 'window_size'   : 7,
                                   'mlp_ratio' : 4.,    'qkv_bias'  : True,          'qk_scale' : None,           'drop_rate'     : 0.0, 
                                   'ape'       : False, 'patch_norm': True,          'use_checkpoint': False,     'drop_path_rate': 0.3} 

    elif swin_name == 'swin_base_224':
        swin_transformer_config = {'image_size': 224,   'patch_size': 4,             'in_chans' : 3,              'num_classes'   : 2,
                                   'embed_dim' : 128,   'depths'   : [2, 2, 18, 2],  'num_heads': [4, 8, 16, 32], 'window_size'   : 7,
                                   'mlp_ratio' : 4.,    'qkv_bias'  : True,          'qk_scale' : None,           'drop_rate'     : 0.0, 
                                   'ape'       : False, 'patch_norm': True,          'use_checkpoint': False,     'drop_path_rate': 0.5} 
    
    elif swin_name == 'swin_base_384':
        swin_transformer_config = {'image_size': 384,   'patch_size': 4,             'in_chans' : 3,              'num_classes'   : 2,
                                   'embed_dim' : 128,   'depths'    : [2, 2, 18, 2], 'num_heads': [4, 8, 16, 32], 'window_size'   : 12,
                                   'mlp_ratio' : 4.,    'qkv_bias'  : True,          'qk_scale' : None,           'drop_rate'     : 0.0, 
                                   'ape'       : False, 'patch_norm': True,          'use_checkpoint': False,     'drop_path_rate': 0.1} 
    
    elif swin_name == 'swin_large_224':
        swin_transformer_config = {'image_size': 224,   'patch_size': 4,             'in_chans' : 3,               'num_classes'    : 2,
                                   'embed_dim' : 192,   'depths'    : [2, 2, 18, 2], 'num_heads': [6, 12, 24, 48], 'window_size'    : 7,
                                   'mlp_ratio' : 4.,    'qkv_bias'  : True,          'qk_scale' : None,            'drop_rate'      : 0.0, 
                                   'ape'       : False, 'patch_norm': True,          'use_checkpoint': False,      'drop_path_rate' : 0.1}
    
    elif swin_name == 'swin_large_384':
        swin_transformer_config = {'image_size': 384,   'patch_size': 4,             'in_chans' : 3,               'num_classes'   : 2,
                                   'embed_dim' : 192,   'depths'    : [2, 2, 18, 2], 'num_heads': [6, 12, 24, 48], 'window_size'   : 12,
                                   'mlp_ratio' : 4.,    'qkv_bias'  : True,          'qk_scale' : None,            'drop_rate'     : 0.0, 
                                   'ape'       : False, 'patch_norm': True,          'use_checkpoint': False,      'drop_path_rate': 0.1}
    
    model = SwinTransformer(img_size = swin_transformer_config['image_size'],
                patch_size  = swin_transformer_config['patch_size'],
                in_chans    = swin_transformer_config['in_chans'],
                num_classes = swin_transformer_config['num_classes'],
                embed_dim   = swin_transformer_config['embed_dim'],
                depths      = swin_transformer_config['depths'],
                num_heads   = swin_transformer_config['num_heads'],
                window_size = swin_transformer_config['window_size'],
                mlp_ratio   = swin_transformer_config['mlp_ratio'],
                qkv_bias    = swin_transformer_config['qkv_bias'],
                qk_scale    = swin_transformer_config['qk_scale'],
                drop_rate   = swin_transformer_config['drop_rate'],
                drop_path_rate = swin_transformer_config['drop_path_rate'],
                ape            = swin_transformer_config['ape'],
                patch_norm     = swin_transformer_config['patch_norm'],
                use_checkpoint = swin_transformer_config['use_checkpoint']
    )
    return model


def build_swin_transformer_v2(swin_v2_name):
    if swin_v2_name == 'swin_v2_tiny_w8':
        swin_transformer_v2_config = {'image_size': 256,   'patch_size': 4,             'in_chans' : 3,              'num_classes'   : 1000,
                                      'embed_dim' : 96,    'depths'    : [2, 2, 6, 2],  'num_heads': [3, 6, 12, 24], 'window_size'   : 8,
                                      'mlp_ratio' : 4.,    'qkv_bias'  : True,          'drop_rate': 0.,             'attn_drop_rate': 0., 
                                      'patch_norm': True,  'ape': False,                'drop_path_rate': 0.3,       'use_checkpoint': False,
                                      'pretrained_window_sizes': [0, 0, 0, 0]} 
    
    elif swin_v2_name == 'swin_v2_tiny_w16':
        swin_transformer_v2_config = {'image_size': 256,   'patch_size': 4,             'in_chans' : 3,              'num_classes'   : 1000,
                                      'embed_dim' : 96,    'depths'    : [2, 2, 6, 2],  'num_heads': [3, 6, 12, 24], 'window_size'   : 16,
                                      'mlp_ratio' : 4.,    'qkv_bias'  : True,          'drop_rate': 0.,             'attn_drop_rate': 0., 
                                      'patch_norm': True,  'ape': False,                'drop_path_rate': 0.3,       'use_checkpoint': False,
                                      'pretrained_window_sizes': [0, 0, 0, 0]}  

    elif swin_v2_name == 'swin_v2_small_w8':
        swin_transformer_v2_config = {'image_size': 256,   'patch_size': 4,             'in_chans' : 3,              'num_classes'   : 1000,
                                      'embed_dim' : 96,    'depths'    : [2, 2, 18, 2], 'num_heads': [3, 6, 12, 24], 'window_size'   : 8,
                                      'mlp_ratio' : 4.,    'qkv_bias'  : True,          'drop_rate': 0.,             'attn_drop_rate': 0., 
                                      'patch_norm': True,  'ape': False,                'drop_path_rate': 0.3,       'use_checkpoint': False,
                                      'pretrained_window_sizes': [0, 0, 0, 0]}  
    
    elif swin_v2_name == 'swin_v2_small_w16':
        swin_transformer_v2_config = {'image_size': 256,   'patch_size': 4,             'in_chans' : 3,              'num_classes'   : 1000,
                                      'embed_dim' : 96,    'depths'    : [2, 2, 18, 2], 'num_heads': [3, 6, 12, 24], 'window_size'   : 16,
                                      'mlp_ratio' : 4.,    'qkv_bias'  : True,          'drop_rate': 0.,             'attn_drop_rate': 0., 
                                      'patch_norm': True,  'ape': False,                'drop_path_rate': 0.1,       'use_checkpoint': False,
                                      'pretrained_window_sizes': [0, 0, 0, 0]}  
    
    elif swin_v2_name == 'swin_v2_base_w8':
        swin_transformer_v2_config = {'image_size': 256,   'patch_size': 4,             'in_chans' : 3,              'num_classes'   : 1000,
                                      'embed_dim' : 128,   'depths'    : [2, 2, 18, 2], 'num_heads': [4, 8, 16, 32], 'window_size'   : 8,
                                      'mlp_ratio' : 4.,    'qkv_bias'  : True,          'drop_rate': 0.,             'attn_drop_rate': 0., 
                                      'patch_norm': True,  'ape': False,                'drop_path_rate': 0.5,       'use_checkpoint': False,
                                      'pretrained_window_sizes': [0, 0, 0, 0]} 
    
    elif swin_v2_name == 'swin_v2_base_w16':
        swin_transformer_v2_config = {'image_size': 256,   'patch_size': 4,             'in_chans' : 3,              'num_classes'   : 1000,
                                      'embed_dim' : 128,   'depths'    : [2, 2, 18, 2], 'num_heads': [4, 8, 16, 32], 'window_size'   : 16,
                                      'mlp_ratio' : 4.,    'qkv_bias'  : True,          'drop_rate': 0.,             'attn_drop_rate': 0., 
                                      'patch_norm': True,  'ape': False,                'drop_path_rate': 0.5,       'use_checkpoint': False,
                                      'pretrained_window_sizes': [0, 0, 0, 0]} 
    elif swin_v2_name == 'swin_v2_base_384_w24':
        swin_transformer_v2_config = {'image_size': 384,   'patch_size': 4,             'in_chans' : 3,              'num_classes'   : 1000,
                                      'embed_dim' : 128,   'depths'    : [2, 2, 18, 2], 'num_heads': [4, 8, 16, 32], 'window_size'   : 24,
                                      'mlp_ratio' : 4.,    'qkv_bias'  : True,          'drop_rate': 0.,             'attn_drop_rate': 0., 
                                      'patch_norm': True,  'ape': False,                'drop_path_rate': 0.2,       'use_checkpoint': False,
                                      'pretrained_window_sizes': [12, 12, 12, 6]} 
    elif swin_v2_name == 'swin_v2_large_384_w24':
        swin_transformer_v2_config = {'image_size': 384,   'patch_size': 4,             'in_chans' : 3,               'num_classes'   : 1000,
                                      'embed_dim' : 192,   'depths'    : [2, 2, 18, 2], 'num_heads': [6, 12, 24, 48], 'window_size'   : 24,
                                      'mlp_ratio' : 4.,    'qkv_bias'  : True,          'drop_rate': 0.,              'attn_drop_rate': 0., 
                                      'patch_norm': True,  'ape': False,                'drop_path_rate': 0.2,        'use_checkpoint': False,
                                      'pretrained_window_sizes': [12, 12, 12, 6]} 
    
    model = SwinTransformerV2(img_size = swin_transformer_v2_config['image_size'],
                patch_size  = swin_transformer_v2_config['patch_size'],
                in_chans    = swin_transformer_v2_config['in_chans'],
                num_classes = swin_transformer_v2_config['num_classes'],
                embed_dim   = swin_transformer_v2_config['embed_dim'],
                depths      = swin_transformer_v2_config['depths'],
                num_heads   = swin_transformer_v2_config['num_heads'],
                window_size = swin_transformer_v2_config['window_size'],
                mlp_ratio   = swin_transformer_v2_config['mlp_ratio'],
                qkv_bias    = swin_transformer_v2_config['qkv_bias'],
                drop_rate   = swin_transformer_v2_config['drop_rate'],
                attn_drop_rate = swin_transformer_v2_config['attn_drop_rate'],
                drop_path_rate = swin_transformer_v2_config['drop_path_rate'],
                ape            = swin_transformer_v2_config['ape'],
                patch_norm     = swin_transformer_v2_config['patch_norm'],
                use_checkpoint = swin_transformer_v2_config['use_checkpoint'],
                pretrained_window_sizes = swin_transformer_v2_config['pretrained_window_sizes']
    )
    return model
