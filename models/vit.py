import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math


class PatchEmbed(nn.Module):
    """
    将图像切分成patches并嵌入
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"
        
        # (B, embed_dim, H/P, W/P) -> (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    """
    Multi-head Self Attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        # (B, N, 3*C) -> (B, N, 3, num_heads, C//num_heads) -> (3, B, num_heads, N, C//num_heads)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention: (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # (B, num_heads, N, C//num_heads) -> (B, N, num_heads, C//num_heads) -> (B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP with GELU activation
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Block(nn.Module):
    """
    Transformer Block: Attention + MLP
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, drop=drop
        )
        
        def forward(self, x):
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=None, act_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        
        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # CLS token & Position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)
        
        # Classification: use CLS token
        x = x[:, 0]
        x = self.head(x)
        
        return x


# ============================================================
# Model Factory Functions
# ============================================================

def vit_tiny_patch16_224(num_classes=1000, **kwargs):
    """ViT-Tiny (5M params)"""
    model = VisionTransformer(
        img_size=224, 
        patch_size=16, 
        embed_dim=192, 
        depth=12, 
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True, 
        num_classes=num_classes, 
        **kwargs
    )
    return model


def vit_small_patch16_224(num_classes=1000, **kwargs):
    """ViT-Small (22M params)"""
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4, qkv_bias=True, num_classes=num_classes, **kwargs
    )
    return model


def vit_base_patch16_224(num_classes=1000, **kwargs):
    """ViT-Base (86M params)"""
    model = VisionTransformer(
        img_size=224, 
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        num_heads=12,
        mlp_ratio=4, 
        qkv_bias=True, 
        num_classes=num_classes, 
        **kwargs
    )
    return model


def vit_large_patch16_224(num_classes=1000, **kwargs):
    """ViT-Large (307M params)"""
    model = VisionTransformer(
        img_size=224, 
        patch_size=16, 
        embed_dim=1024, 
        depth=24, 
        num_heads=16,
        mlp_ratio=4, 
        qkv_bias=True, 
        num_classes=num_classes, 
        **kwargs
    )
    return model


# 适配不同图像尺寸的版本
def vit_small_patch16_512(num_classes=1000, **kwargs):
    """ViT-Small for 512x512 images"""
    model = VisionTransformer(
        img_size=512, 
        patch_size=16, 
        embed_dim=384, 
        depth=12, 
        num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True, 
        num_classes=num_classes, 
        **kwargs
    )
    return model


def vit_base_patch16_512(num_classes=1000, **kwargs):
    """ViT-Base for 512x512 images"""
    model = VisionTransformer(
        img_size=512, 
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        num_heads=12,
        mlp_ratio=4, 
        qkv_bias=True, 
        num_classes=num_classes, 
        **kwargs
    )
    return model


def vit_micro_patch16_512(num_classes=1000, **kwargs):
    """ViT-Micro for 512x512 images - 专为小数据集设计 (~3M params)"""
    model = VisionTransformer(
        img_size=512, 
        patch_size=32,  # 更大的 patch，减少序列长度
        embed_dim=256,  # 更小的 embedding
        depth=8,        # 更少的层数
        num_heads=4,    # 更少的注意力头
        mlp_ratio=3.0,  # 更小的 MLP
        qkv_bias=True, 
        num_classes=num_classes, 
        **kwargs
    )
    return model


def vit_nano_patch32_512(num_classes=1000, **kwargs):
    """ViT-Nano for 512x512 images - 超轻量级 (~1M params)"""
    model = VisionTransformer(
        img_size=512, 
        patch_size=32,  
        embed_dim=192,  # 非常小的 embedding
        depth=6,        # 很少的层数
        num_heads=3,    
        mlp_ratio=3.0,  
        qkv_bias=True, 
        num_classes=num_classes, 
        **kwargs
    )
    return model


def vit_rs_small_patch16_512(num_classes=1000, **kwargs):
    """
    ViT-RS-Small for 512x512 remote sensing:
      - patch_size=16
      - embed_dim=384
      - depth=10
      - num_heads=6
      - mlp_ratio=3.0
      - drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.2
    """
    model = VisionTransformer(
        img_size=512,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=384,
        depth=10,
        num_heads=6,
        mlp_ratio=3.0,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.2,
        **kwargs
    )
    return model


# 创建自定义 ViT 的便捷函数
def create_vit_model(img_size=224, patch_size=16, num_classes=1000, 
                     embed_dim=768, depth=12, num_heads=12, 
                     drop_rate=0., drop_path_rate=0.1):
    """
    创建自定义配置的 ViT 模型
    """
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate
    )
    return model
