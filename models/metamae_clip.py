from metaformer_mae import *

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

import sys
import os
sys.path.insert(0, os.getcwd())

from utils.mae import PatchEmbedding, PatchExpanding, get_2d_sincos_pos_embed
# from swin_unet import PatchEmbedding, BasicBlock, PatchExpanding, BasicBlockUp
# from utils.pos_embed import get_2d_sincos_pos_embed

class MetaMAE(nn.Module):
    '''
    基于SwinMAE修改的MetaFormerMAE
    '''
    def __init__(self, img_size=224, patch_size=4, mask_ratio=0.75, in_channels=3, decoder_dim=512, 
                 encoder_depths=[2, 2, 6, 2], 
                 decoder_depths=[2, 2, 2],
                 dims=[64, 128, 320, 512], 
                 clip_dim=512,
                 downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
                 token_mixers=[nn.Identity, nn.Identity, nn.Identity, nn.Identity],
                 mlps=[Mlp, Mlp, Mlp, Mlp],
                 norm_layers=[partial(LayerNormWithoutBias, eps=1e-6), partial(LayerNormWithoutBias, eps=1e-6),
                              partial(LayerNormWithoutBias, eps=1e-6), partial(LayerNormWithoutBias, eps=1e-6),], 
                 drop_path_rate=0.,
                 layer_scale_init_values=[None, None, None, None],
                 res_scale_init_values=[None, None, 1.0, 1.0],
                 patch_norm=True,
                 encoder_output_norm=partial(nn.LayerNorm, eps=1e-6),
                 patch_norm_layer=None,
                 norm_pix_loss=False,):
        super().__init__()
        
        num_encoder_stage = len(encoder_depths)
        dims_down = [in_channels] + dims
        drop_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(encoder_depths))]
        self.mask_ratio = mask_ratio
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.norm_pix_loss = norm_pix_loss
        self.num_encoder_stage = num_encoder_stage
        self.encoder_embed_dim = dims[0]
        self.decoder_embed_dim = dims[-1]

        # encoder
        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_channels, embed_dim=self.encoder_embed_dim,
                                          norm_layer=patch_norm_layer if patch_norm else None)
        self.pos_embed  = nn.Parameter(torch.zeros(1, self.num_patches, self.encoder_embed_dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))
        
        self.downsample_layers = nn.ModuleList(
            [downsample_layers[i](dims_down[i], dims_down[i+1]) for i in range(num_encoder_stage)]
        )
        self.encoder_stages = nn.ModuleList()
        current = 0
        for i in range(num_encoder_stage):
            stage = nn.Sequential(
                *[MetaFormerBlock(
                    dim=dims[i],
                    token_mixer=token_mixers[i],
                    mlp=mlps[i],
                    norm_layer=norm_layers[i],
                    drop_path=drop_rates[current+j],
                    layer_scale_init_value=layer_scale_init_values[i],
                    res_scale_init_value=res_scale_init_values[i]
                ) for j in range(encoder_depths[i])]
            )
            self.encoder_stages.append(stage)
            current += encoder_depths[i]

        # 用于与CLIP对齐的部分
        self.encoder_output_norm = encoder_output_norm(dims[-1])
        self.clip_proj = nn.Linear(dims[-1], clip_dim)

        # decoder
        self.num_decoder_stage = num_encoder_stage - 1
        self.upsample_layers = nn.ModuleList(
            [PatchExpanding(dims[-i-1]) for i in range(self.num_decoder_stage)]
        )
        self.decoder_stages = nn.ModuleList()
        for i in range(self.num_decoder_stage):
            stage = nn.Sequential(
                *[MetaFormerBlock(
                    dim=dims[-i-2],
                    token_mixer=VanillaAttention,
                    mlp=mlps[-i],
                    norm_layer=norm_layers[-i-2],
                    drop_path=0.,
                    layer_scale_init_value=layer_scale_init_values[-i-2],
                    res_scale_init_value=res_scale_init_values[-i-2]
                ) for j in range(decoder_depths[-i])]
            )
            self.decoder_stages.append(stage)

        self.decoder_norm = patch_norm_layer(self.encoder_embed_dim)
        self.decoder_proj = nn.Linear(self.decoder_embed_dim // 8, patch_size ** 2 * in_channels, bias=True)

        self.initialize_weights()
    
    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**0.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear): 
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def patchify(self, imgs):
        '''
        imgs: [B, 3, H, W]
        x:    [B, L, patch_size**2*3]
        '''
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h*w, p**2*3)
        return x
    
    def unpatchify(self, x):
        '''
        x:    [B, L, patch_size**2*3]
        imgs: [B, 3, H, W]
        '''
        p = self.patch_size
        h = w = int(x.shape[1]**0.5)
        assert h*w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 3, h*p, w*p)
        return imgs
    
    def adjacent_masking(self, x, r=4, remove=False, mask_len_sparse=False):
        # [B, H, W, C] -> [B, H*W, C]
        x = rearrange(x, 'B H W C -> B (H W) C')
        B, L, C = x.shape
        assert int(L ** 0.5 / r) == L ** 0.5 / r
        # 得到相邻patch进行mask, 得到了一条边有多少个mask组合
        # d*d为使用相邻mask策略下总的token数目
        d = int(L ** 0.5 // r)

        # 长度为d*d的随机噪声序列用来排序
        noise = torch.rand(B, d**2, device=x.device)
        # 排序操作
        sparse_shuffle = torch.argsort(noise, dim=1)
        # 逆排序操作, 保存以用来重建mask
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        # 获取保留的token索引, 共d*d*(1-mask_ratio)个
        sparse_keep = sparse_shuffle[:, :int(d**2*(1-self.mask_ratio))]

        # 计算保留的token索引
        # 得到一个window中第一个patch的索引
        # [B, L-d*d*(1-mask_ratio)]
        index_keep_part = torch.div(sparse_keep, d, rounding_mode='floor') * d * r ** 2 \
                        + sparse_keep % d * r
        index_keep = index_keep_part
        # 将一个window中其他r*r-1个patch索引保留
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat([index_keep, index_keep_part + int(L**0.5)*i+j], dim=1)

        # 所有patch的索引, [B, L] = [B, H*W]
        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)
        # 初始化所有mask掉的window索引, [B, L-d*d*(1-mask_ratio)]
        index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int32)
        # 对所有batch中, 找出 index_all[i] 中不在 index_keep.cpu().numpy()[i] 中的元素。
        # 即所有mask掉的patch的索引
        for i in range(B):
            index_mask[i] = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        # 生成随机打乱的index
        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        # 恢复打乱的方法
        index_restore = torch.argsort(index_shuffle, dim=1)

        if mask_len_sparse:
            mask = torch.ones([B, d ** 2], device=x.device)
            mask[:, :sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            mask = torch.ones([B, L], device=x.device)
            mask[:, :index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)
            for i in range(B):
                x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask
        
    def forward_encoder(self, x):
        x = self.patch_embed(x)
        # print(f'after patch embed shape {x.shape}')

        x, mask = self.adjacent_masking(x, remove=False, mask_len_sparse=False)
        # print(f'after adjacent masking shape {x.shape}')

        for i in range(self.num_encoder_stage):
            if i == 0:
                x = self.encoder_stages[i](x)
                # print(f'stage {i+1} shape {x.shape} after blocks')
            else:
                x = self.downsample_layers[i](x)
                # print(f'stage {i+1} shape {x.shape} after downsampling')
                x = self.encoder_stages[i](x)
                # print(f'stage {i+1} shape {x.shape} after encoder blocks')
        return x, mask
    
    def forward_clip_adapter(self, x):
        # [B, H, W, C] -> [B, C]
        x = self.encoder_output_norm(x.mean([1, 2]))
        # [B, C] -> [B, 512]
        x = self.clip_proj(x)
        return x
    
    def forward_decoder(self, x):
        for i in range(self.num_decoder_stage):
            x = self.upsample_layers[i](x)
            # print(f'stage {i+1} shape {x.shape} after upsampling')
            x = self.decoder_stages[i](x)
            # print(f'stage {i+1} shape {x.shape} after decoder blocks')

        x = self.decoder_norm(x)
        x = rearrange(x, 'B H W C -> B (H W) C')
        x = self.decoder_proj(x)
        # print(f'after decoder proj shape {x.shape}')
        return x
    
    def forward_loss(self, imgs, pred, mask):
        '''
        imgs: [B, 3, H, W]
        pred: [B, L, patch_size**2*3]
        mask: [B, L], 0 is keep, 1 is remove
        L = (H/patch_size)**2
        '''
        target = self.patchify(imgs)
        # print(f'target shape {target.shape}')
        # print(f'pred   shape {pred.shape}')
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) 

        loss = (loss * mask).sum() / mask.sum() 
        return loss
    
    def forward(self, x):
        latent, mask = self.forward_encoder(x)
        latent_clip  = self.forward_clip_adapter(latent)
        pred = self.forward_decoder(latent)
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask, latent_clip


def metamae_clip(**kwargs):
    model = MetaMAE(img_size=224, patch_size=4, mask_ratio=0.75, in_channels=3, decoder_dim=512, 
                 encoder_depths=[3, 12, 18, 3], 
                 decoder_depths=[2, 2, 2],
                 dims=[128, 256, 512, 1024], 
                 clip_dim=512,
                 downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
                 token_mixers=[SepConv, SepConv, VanillaAttention, VanillaAttention],
                 mlps=[Mlp, Mlp, Mlp, Mlp],
                 norm_layers=[partial(LayerNormWithoutBias, eps=1e-6), partial(LayerNormWithoutBias, eps=1e-6),
                              partial(LayerNormWithoutBias, eps=1e-6), partial(LayerNormWithoutBias, eps=1e-6),], 
                 drop_path_rate=0.,
                 layer_scale_init_values=[None, None, None, None],
                 res_scale_init_values=[None, None, 1.0, 1.0],
                 encoder_output_norm=partial(nn.LayerNorm, eps=1e-6),
                 patch_norm=True,
                 patch_norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_pix_loss=False,
                 **kwargs)
    return model

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    print(f'input shape {x.shape}')
    model = metamae_clip()
    loss, pred, mask, latent_clip = model(x)
    print('output shape', pred.shape)
    print('clip feature shape', latent_clip.shape)
    
    import torch
    import clip
    from PIL import Image
    import torch.nn as nn
    device = "cpu"
    # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    model, preprocess = clip.load("ViT-B/32", device=device)
    image = preprocess(Image.open("dog1.jpg")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
                                        
    loss_func = nn.CosineSimilarity(dim=-1)
    loss = loss_func(image_features.float(), latent_clip.float())
    loss = -loss.mean()
    print(loss)