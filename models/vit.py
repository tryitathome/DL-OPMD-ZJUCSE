import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 自注意力机制
        x2 = self.norm1(x)
        x2 = x2.transpose(0, 1)  # 转换为 (seq_len, batch_size, dim)
        attn_output, _ = self.attn(x2, x2, x2)
        attn_output = attn_output.transpose(0, 1)  # 转换回 (batch_size, seq_len, dim)
        x = x + attn_output

        # 前馈网络
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1):
        super(ViT, self).__init__()

        assert image_size % patch_size == 0, '图像尺寸必须能被 patch 大小整除。'

        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        # Patch Embedding
        self.patch_embeddings = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)

        # 位置编码
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, dim))

        # 分类令牌
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer 编码层
        self.transformer = nn.ModuleList([
            TransformerEncoderLayer(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        # MLP 分类头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.position_embeddings, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # 输入 x 的形状: (batch_size, channels, height, width)
        x = self.patch_embeddings(x)  # 输出形状: (batch_size, dim, num_patches_sqrt, num_patches_sqrt)
        x = x.flatten(2)  # 展平为 (batch_size, dim, num_patches)
        x = x.transpose(1, 2)  # 转换为 (batch_size, num_patches, dim)

        batch_size = x.size(0)

        # 添加分类令牌
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, dim)

        # 添加位置编码
        x = x + self.position_embeddings[:, :(x.size(1))]

        x = self.dropout(x)

        # Transformer 编码层
        for layer in self.transformer:
            x = layer(x)

        # 分类头
        x = self.mlp_head(x[:, 0])

        return x
