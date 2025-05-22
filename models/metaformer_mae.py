from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as func

from einops import rearrange

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers.helpers import to_2tuple

 
'''MetaFormer的基本默认配置'''
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }


'''每一个stage之间的降采样操作'''
class Downsampling(nn.Module):
    '''
    使用一层卷积来完成降采样
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.down_sampling = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding)
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        # LN
        x = self.pre_norm(x)
        if self.pre_permute:
            # [B, H, W, C] -> [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        x = self.down_sampling(x)
        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        # LN
        x = self.post_norm(x)
        return x


'''PatchEmbedding操作, 可以替代第一层downsampling以与MetaMAE匹配'''
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int = 4, in_c: int = 3, embed_dim: int = 96, norm_layer: nn.Module = None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=(patch_size,) * 2, stride=(patch_size,) * 2)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def padding(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            x = func.pad(x, (0, self.patch_size - W % self.patch_size,
                             0, self.patch_size - H % self.patch_size,
                             0, 0))
        return x

    def forward(self, x):
        x = self.padding(x)
        x = self.proj(x)
        x = rearrange(x, 'B C H W -> B H W C')
        x = self.norm(x)
        return x


'''在通道维度上逐元素相乘, 实现可学习的张量缩放'''
class Scale(nn.Module):
    '''
    可以用在注意力机制中调整不同头的输出
    归一化层后调整特征的尺度
    '''
    def __init__(self, dim, init_value=1.0, learnable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value*torch.ones(dim), requires_grad=learnable)
    
    def forward(self, x):
        return x * self.scale


'''平方ReLU激活函数'''
class SquaredReLU(nn.Module):
    '''
    x = ReLU(x) ** 2
    '''
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
    def forward(self, x):
        return torch.square(self.relu(x))
    

'''StarReLU'''
class StarReLU(nn.Module):
    '''
    x = s * ReLU(x)**2 + b
    加入可学习的缩放和偏移参数
    '''
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU()
        self.scale = nn.Parameter(scale_value*torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value*torch.ones(1), requires_grad=bias_learnable)
    
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias


'''原始注意力'''
class VanillaAttention(nn.Module):
    '''
    qkv
    Attention is all you need 中的原始注意力
    '''
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0, proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


'''随机混合'''
class RandomMixing(nn.Module):
    '''
    设置随机混合矩阵进行token之间的mix, 不可学习
    '''
    def __init__(self, num_tokens=196, **kwargs):
        super().__init__()
        self.random_matrix = nn.parameter.Parameter(
            data=torch.softmax(torch.rand(num_tokens, num_tokens), dim=-1),
            requires_grad=False
        )
    
    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        x = x.reshape(B, N, C)
        x = torch.einsum('mn, bnc -> bmc', self.random_matrix, x)
        x = x.reshape(B, H, W, C)
        return x


'''可分离卷积'''
class SepConv(nn.Module):
    '''
    pw + dw + pw
    '''
    def __init__(self, dim, expansion_ratio=2, act_layer1=StarReLU, act_layer2=nn.Identity,
                 bias=False, kernel_size=7, padding=3, **kwargs):
        super().__init__()
        hidden_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, hidden_channels, bias=bias)
        self.act1 = act_layer1()
        self.dwconv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                                padding=padding, groups=hidden_channels, bias=bias)
        self.act2 = act_layer2()
        self.pwconv2 = nn.Linear(hidden_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


'''池化'''
class Pooling(nn.Module):
    """
    [B, H, W, C] input
    PoolFormer
    """
    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        y = x.permute(0, 3, 1, 2)
        y = self.pool(y)
        y = y.permute(0, 2, 3, 1)
        return y - x


'''
LayerNorm
nn.LayerNoem(normalized_shape, eps = 1e-5, elementwise_affine = True, device=None, dtype=None)
normalized_shape必须等于tensor的最后一个维度的大小, 不能是中间维度的大小
代表标准化 tensor 的最后一维。
另外也可以是一个列表，但这个列表也必须是最后的 D 个维度的列表
'''
class LayerNormGeneral(nn.Module):
    r""" General LayerNorm for different situations.

    Args:
        affine_shape (int, list or tuple): The shape of affine weight and bias.
            Usually the affine_shape=C, but in some implementation, like torch.nn.LayerNorm,
            the affine_shape is the same as normalized_dim by default. 
            To adapt to different situations, we offer this argument here.
        normalized_dim (tuple or list): Which dims to compute mean and variance. 
        scale (bool): Flag indicates whether to use scale or not.
        bias (bool): Flag indicates whether to use scale or not.

        We give several examples to show how to specify the arguments.

        LayerNorm (https://arxiv.org/abs/1607.06450):
            For input shape of (B, *, C) like (B, N, C) or (B, H, W, C),
                affine_shape=C, normalized_dim=(-1, ), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, ), scale=True, bias=True.

        Modified LayerNorm (https://arxiv.org/abs/2111.11418)
            that is idental to partial(torch.nn.GroupNorm, num_groups=1):
            For input shape of (B, N, C),
                affine_shape=C, normalized_dim=(1, 2), scale=True, bias=True;
            For input shape of (B, H, W, C),
                affine_shape=C, normalized_dim=(1, 2, 3), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, 2, 3), scale=True, bias=True.

        For the several metaformer baslines,
            IdentityFormer, RandFormer and PoolFormerV2 utilize Modified LayerNorm without bias (bias=False);
            ConvFormer and CAFormer utilizes LayerNorm without bias (bias=False).
    """
    def __init__(self, affine_shape=None, normalized_dim=(-1, ), scale=True, 
        bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


'''更快的LayerNorm非Bias实现'''
class LayerNormWithoutBias(nn.Module):
    """
    Equal to partial(LayerNormGeneral, bias=False) but faster, 
    because it directly utilizes otpimized F.layer_norm
    """
    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.bias = None
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)
    

'''模板MLP实现, 被广泛用于Transformer MLP-Mixer PoolFormer MetaFormer中'''
class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_channels=None, act_layer=StarReLU, drop=0., bias=False):
        super().__init__()
        in_channels = dim
        out_channels = out_channels or in_channels
        hidden_channels = int(mlp_ratio * in_channels)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    

'''MLP分类头'''
class MlpHead(nn.Module):
    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_layer=SquaredReLU,
        norm_layer=nn.LayerNorm, head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.in_features = hidden_features
        self.fc = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.head = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)


    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.head(x)
        return x
    

'''MetaFormer基础块'''
class MetaFormerBlock(nn.Module):
    def __init__(self, dim, token_mixer=nn.Identity, mlp=Mlp, norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer  = token_mixer(dim=dim, drop=drop)
        self.drop_path1   = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) if layer_scale_init_value else nn.Identity()
        self.res_scale1   = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp   = mlp(dim=dim, drop=drop)
        self.drop_path2   = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) if layer_scale_init_value else nn.Identity()
        self.res_scale2   = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()

    def forward(self, x):
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(
                        self.norm1(x)
                    )
                )
            )
        
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(
                        self.norm2(x)
                    )
                )
            )
        return x


'''原始的MetaFormer降采样策略'''
DOWNSAMPLE_LAYERS_FOUR_STAGES = [
    partial(Downsampling, kernel_size=7, stride=4, padding=2, post_norm=partial(LayerNormGeneral, bias=False, eps=1e-6))] + \
    [partial(Downsampling, kernel_size=3, stride=2, padding=1, pre_norm=partial(LayerNormGeneral, bias=False, eps=1e-6), pre_permute=True)]*3


'''第一阶段改成4*4的patch划分, 尺寸一致'''
patch_size = 4
DOWNSAMPLE_LAYERS_FOUR_STAGES = [
    partial(Downsampling, kernel_size=patch_size, stride=patch_size, padding=0, 
            post_norm=partial(LayerNormGeneral, bias=False, eps=1e-6))] + \
    [partial(Downsampling, kernel_size=3, stride=2, padding=1, 
             pre_norm=partial(LayerNormGeneral, bias=False, eps=1e-6), pre_permute=True)]*3

'''MetaFormer'''
class MetaFormer(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000,
                 patch_size=4,
                 depths=[2, 2, 6, 2],
                 dims=[64, 128, 320, 512],
                 downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
                 token_mixers=[nn.Identity, nn.Identity, nn.Identity, nn.Identity],
                 mlps=[Mlp, Mlp, Mlp, Mlp,],
                 norm_layers=[partial(LayerNormWithoutBias, eps=1e-6), partial(LayerNormWithoutBias, eps=1e-6),
                              partial(LayerNormWithoutBias, eps=1e-6), partial(LayerNormWithoutBias, eps=1e-6),], 
                 drop_path_rate=0.,
                 head_dropout=0.0,
                 layer_scale_init_values=[None, None, None, None],
                 res_scale_init_values=[None, None, 1.0, 1.0],
                 patch_norm=True,
                 patch_norm_layer=None,
                 output_norm=partial(nn.LayerNorm, eps=1e-6), 
                 head_fn=nn.Linear
                 ):
        super().__init__()

        num_stage = len(depths)
        dims_down = [in_channels] + dims
        drop_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.num_stage = num_stage
        self.num_classes = num_classes
        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_channels, embed_dim=dims[0],
                                          norm_layer=patch_norm_layer if patch_norm else None)
        self.downsample_layers = nn.ModuleList(
            [downsample_layers[i](dims_down[i], dims_down[i+1]) for i in range(num_stage)]
        )
        self.encoder_stages = nn.ModuleList()
        current = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[MetaFormerBlock(
                    dim=dims[i],
                    token_mixer=token_mixers[i],
                    mlp=mlps[i],
                    norm_layer=norm_layers[i],
                    drop_path=drop_rates[current+j],
                    layer_scale_init_value=layer_scale_init_values[i],
                    res_scale_init_value=res_scale_init_values[i]
                ) for j in range(depths[i])]
            )
            self.encoder_stages.append(stage)
            current += depths[i]
        
        self.feature_output_norm = output_norm(dims[-1])

        if head_dropout > 0.0:
            self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout)
        else:
            self.head = head_fn(dims[-1], num_classes)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # norm层不需要权重衰减
        return {'feature_output_norm'}
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        # print(f'after patch embed shape {x.shape}')
        for i in range(self.num_stage):
            if i == 0:
                x = self.encoder_stages[i](x)
                # print(f'stage {i+1} shape {x.shape} after blocks')
            else:
                x = self.downsample_layers[i](x)
                # print(f'stage {i+1} shape {x.shape} after downsampling')
                x = self.encoder_stages[i](x)
                # print(f'stage {i+1} shape {x.shape} after blocks')
        # [B, H, W, C] -> [B, C]
        x = x.mean([1, 2])
        x = self.feature_output_norm(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        # print(f'feature output shape {x.shape}')
        x = self.head(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 3, 384, 384)
    print('input shape  ', x.shape)
    # model = MetaFormerBlock(dim=x.shape[-1])
    model = MetaFormer(depths=[3, 12, 18, 3], dims=[128, 256, 512, 1024], patch_size=4,
                    token_mixers=[SepConv, SepConv, VanillaAttention, VanillaAttention],
                    patch_norm=True, patch_norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    head_fn=MlpHead)
    y = model(x)
    print('output shape', y.shape)
