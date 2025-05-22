import os
import sys
sys.path.insert(0, os.getcwd())
from models.convnextv2 import ConvNeXtV2


def build_convnextv2(convenevtv2_name, num_classes):
    model = None
    if convenevtv2_name == 'convnextv2_atto':
        model = ConvNeXtV2(num_classes=num_classes, depths=[2, 2, 6, 2], dims=[40, 80, 160, 320])
    elif convenevtv2_name == 'convnextv2_femto':
        model = ConvNeXtV2(num_classes=num_classes, depths=[2, 2, 6, 2], dims=[48, 96, 192, 384])
    elif convenevtv2_name == 'convnext_pico':
        model = ConvNeXtV2(num_classes=num_classes, depths=[2, 2, 6, 2], dims=[64, 128, 256, 512])
    elif convenevtv2_name == 'convnextv2_nano':
        model = ConvNeXtV2(num_classes=num_classes, depths=[2, 2, 8, 2], dims=[80, 160, 320, 640])
    elif convenevtv2_name == 'convnextv2_tiny':
        model = ConvNeXtV2(num_classes=num_classes, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
    elif convenevtv2_name == 'convnextv2_base':
        model = ConvNeXtV2(num_classes=num_classes, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
    elif convenevtv2_name == 'convnextv2_large':
        model = ConvNeXtV2(num_classes=num_classes, depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
    elif convenevtv2_name == 'convnextv2_huge':
        model = ConvNeXtV2(num_classes=num_classes, depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816])
    
    return model