import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat
from torch.nn.init import trunc_normal_
from .DCNv4.modules.dcnv4 import DCNv4


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class DropPath(nn.Module):
    """
    Stochastic Depth / DropPath, Drop the residual branch randomly.
    Compatible with any dimension input, only sample mask in batch dimension.
    """
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0. or (not self.training):
            return x
        keep_prob = 1.0 - self.drop_prob
        # mask 只在 batch 维采样，其余维度共享
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x / keep_prob * random_tensor


class PreNorm(nn.Module):
    """LayerNorm + submodule, used for Transformer Pre-LN structure"""
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn   = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """MLP submodule in Transformer"""
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Standard multi-head self-attention (scaled dot-product attention)"""
    def __init__(self,
                 dim: int,
                 heads: int = 4,
                 dim_head: int = 64,
                 dropout: float = 0.):
        super().__init__()
        inner_dim  = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, N, C)
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)   # 3 × (B, N, inner_dim)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = self.attend(dots)
        out  = einsum("b h i j, b h j d -> b h i d", attn, v)
        out  = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """Stacked Transformer Encoder with DropPath"""
    def __init__(self,
                 dim: int,
                 depth: int,
                 heads: int,
                 dim_head: int,
                 mlp_dim: int,
                 dropout: float = 0.,
                 drop_path_rate: float = 0.1):
        super().__init__()
        layers = []
        dpr = torch.linspace(0, drop_path_rate, steps=depth).tolist()
        for i in range(depth):
            attn = PreNorm(dim, Attention(dim, heads, dim_head, dropout))
            ff   = PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            drop = DropPath(dpr[i]) if dpr[i] > 0 else nn.Identity()
            layers.append(nn.ModuleList([attn, ff, drop]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for attn, ff, drop in self.layers:
            x = x + drop(attn(x))
            x = x + drop(ff(x))
        return x


# DCNv4 Block Lite
class DCNv4BlockLite(nn.Module):
    """
    Single DCNv4 + MLP residual block (simplified version of InternImageLayer):
    Input/output are both (B, N, C), where N = H*W.

      x -> LN -> DCNv4 -> (optional LayerScale) -> DropPath -> residual
        -> LN -> MLP -> (optional LayerScale) -> DropPath -> residual
    """
    def __init__(self,
                 channels: int,
                 groups: int,
                 mlp_ratio: float = 4.0,
                 drop_path: float = 0.0,
                 offset_scale: float = 1.0,
                 layer_scale_init: float | None = 1e-6):
        super().__init__()
        self.channels = channels
        self.groups   = groups

        # DCNv4 main body
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.dcn   = DCNv4(
            channels     = channels,
            kernel_size  = 3,
            stride       = 1,
            pad          = 1,
            group        = groups,
            offset_scale = offset_scale
        )

        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        hidden_dim = int(channels * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels)
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if layer_scale_init is not None and layer_scale_init > 0:
            self.gamma1 = nn.Parameter(layer_scale_init * torch.ones(channels),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init * torch.ones(channels),
                                       requires_grad=True)
        else:
            self.gamma1 = None
            self.gamma2 = None

    def forward(self, x, hw_shape):
        """
        x: (B, N, C)
        hw_shape: (H, W)
        """
        H, W = hw_shape
        B, N, C = x.shape
        assert N == H * W, f"N={N} does not match H*W={H*W}"

        # DCNv4 branch
        shortcut = x
        x = self.norm1(x)
        x = self.dcn(x, shape=(H, W))   # still (B, N, C)

        if self.gamma1 is not None:
            x = self.gamma1 * x
        x = shortcut + self.drop_path(x)

        # MLP branch
        shortcut2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = shortcut2 + self.drop_path(x)

        return x


class DCNv4Stage(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 depth: int,
                 groups: int,
                 mlp_ratio: float,
                 drop_path_rates: list[float],
                 downsample: bool = True,
                 offset_scale: float = 1.0,
                 layer_scale_init: float | None = 1e-6):
        super().__init__()

        # Downsample layer: 3x3 Conv, stride=2 or 1 (only change channels)
        if downsample or in_channels != out_channels:
            stride = 2 if downsample else 1
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
        else:
            self.downsample = nn.Identity()

        self.blocks = nn.ModuleList([
            DCNv4BlockLite(
                channels        = out_channels,
                groups          = groups,
                mlp_ratio       = mlp_ratio,
                drop_path       = drop_path_rates[i],
                offset_scale    = offset_scale,
                layer_scale_init= layer_scale_init
            )
            for i in range(depth)
        ])

        self.out_channels = out_channels

    def forward(self, x):
        # x: (B, C_in, H_in, W_in)
        x = self.downsample(x)
        B, C, H, W = x.shape
        # reshape to (B, N, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        for blk in self.blocks:
            x_flat = blk(x_flat, hw_shape=(H, W))

        # reshape to feature map
        x = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x, (H, W)


class DCNv4HierarchicalBackbone(nn.Module):
    """
    Hierarchical DCNv4 Backbone:
    Stem: Conv3x3(stride=2) * 2  => /4
    Stage0: channels base, no downsample (keep /4)
    Stage1: channels base * 2, downsample => /8
    Stage2: channels base * 4, downsample => /16
    Stage3: channels base * 8, downsample => /32

    Output:
    feature: (B, C_last, H_out, W_out), where H_out = image_size / 32
    """
    def __init__(self,
                 in_channels: int = 3,
                 base_channels: int = 48,
                 depths: list[int] = [2, 2, 4, 2],
                 mlp_ratio: float = 4.0,
                 drop_path_rate: float = 0.1,
                 offset_scale: float = 1.0,
                 layer_scale_init: float | None = 1e-6):
        super().__init__()

        assert base_channels % 16 == 0, "base_channels 必须是 16 的倍数，保证 channels/groups=16 整除"

        stem_out = base_channels    # Assume base_channels = 80
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_out // 2, kernel_size=3, stride=2, padding=1, bias=False), # (B,3,512,512) -> (B, 40, 256, 256)
            nn.BatchNorm2d(stem_out // 2),
            nn.GELU(),
            nn.Conv2d(stem_out // 2, stem_out, kernel_size=3, stride=2, padding=1, bias=False),  # (B, 40, 256, 256) -> (B, 80, 128, 128)
            nn.BatchNorm2d(stem_out),
            nn.GELU()
        )

        self.stage_channels = [
            base_channels,       # Stage0  input: 80 channels, output: 80 channels
            base_channels * 2,   # Stage1  input: 80 channels, output: 160 channels
            base_channels * 4,   # Stage2  input: 160 channels, output: 320 channels
            base_channels * 8    # Stage3  input: 320 channels, output: 640 channels
        ]

        # Assume groups[i] = base_channels // 16 * (2 ** i)
        base_group = base_channels // 16
        self.stage_groups = [base_group * (2 ** i) for i in range(4)]  # [5, 10, 20, 40]

        # Check if the channels and groups are divisible by 16
        for c, g in zip(self.stage_channels, self.stage_groups):
            assert c % g == 0 and (c // g) % 16 == 0, \
                f"Channels={c}, groups={g} does not satisfy (C/group) % 16 == 0"

        # Progressive DropPath
        total_blocks = sum(depths)   # Assume large version, depths = [3, 4, 16, 6], total_blocks = 3 + 4 + 16 + 6 = 29
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()   # drop path rate for each block
        dp_offset = 0

        stages = []
        for i in range(4):
            in_ch  = stem_out if i == 0 else self.stage_channels[i - 1]
            out_ch = self.stage_channels[i]
            depth  = depths[i]
            down   = (i > 0)  # Stage0: no downsample, other stages downsample

            stage = DCNv4Stage(
                in_channels   = in_ch,
                out_channels  = out_ch,
                depth         = depth,
                groups        = self.stage_groups[i],
                mlp_ratio     = mlp_ratio,
                drop_path_rates = dpr[dp_offset: dp_offset + depth],
                downsample    = down,
                offset_scale  = offset_scale,
                layer_scale_init = layer_scale_init
            )
            stages.append(stage)
            dp_offset += depth

        self.stages = nn.ModuleList(stages)
        self.out_channels = self.stage_channels[-1]

    def forward(self, x):
        x = self.stem(x)   # (B, base, H/4, W/4)
        H, W = x.shape[2:]
        for i, stage in enumerate(self.stages):
            x, (H, W) = stage(x)
        return x  # 最终 (B, C_last, H_out, W_out)


# DCNv4 + ViT Hybrid classifier
class DViTDCNv4Hybrid(nn.Module):
    """
    Deformable Vision Transformer (DCNv4 + ViT) 混合架构：
      - 分层 DCNv4 主干（类似 FlashInternImage / InternImage）
      - 轻量 ViT 头（few layers，多头自注意力）
    """
    def __init__(self,
                 image_size: int = 512,
                 num_classes: int = 8,
                 # backbone config
                 in_channels: int = 3,
                 base_channels: int = 48,
                 depths: list[int] = [2, 2, 4, 2],
                 mlp_ratio: float = 4.0,
                 drop_path_rate_backbone: float = 0.1,
                 # transformer head config
                 vit_dim: int = 256,
                 vit_depth: int = 4,
                 vit_heads: int = 4,
                 vit_dim_head: int = 64,
                 vit_mlp_ratio: float = 4.0,
                 vit_dropout: float = 0.1,
                 vit_drop_path_rate: float = 0.1,
                 pool: str = "hybrid",    # {"cls", "mean", "hybrid"}
                 layer_scale_init: float | None = 1e-6):
        super().__init__()

        assert pool in {"cls", "mean", "hybrid"}
        self.pool       = pool
        self.image_size = image_size

        # 1. DCNv4 Backbone
        self.backbone = DCNv4HierarchicalBackbone(
            in_channels   = in_channels,
            base_channels = base_channels,
            depths        = depths,
            mlp_ratio     = mlp_ratio,
            drop_path_rate= drop_path_rate_backbone,
            offset_scale  = 1.0,
            layer_scale_init = layer_scale_init
        )

        # 2. Calculate the number of tokens in the backbone output
        img_h, img_w = pair(image_size)
        down = 32
        assert img_h % down == 0 and img_w % down == 0, \
            f"image_size ({img_h},{img_w}) must be divisible by 32"
        fh, fw      = img_h // down, img_w // down
        num_patches = fh * fw
        patch_dim   = self.backbone.out_channels

        self.num_patches = num_patches
        self.fh, self.fw = fh, fw

        # 3. Patch embedding：LN + Linear -> vit_dim
        self.patch_norm = nn.LayerNorm(patch_dim, eps=1e-6)
        self.patch_proj = nn.Linear(patch_dim, vit_dim)

        # 4. CLS + Pos Embedding
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, vit_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, vit_dim))
        self.pos_dropout   = nn.Dropout(vit_dropout)

        # 5. Transformer head
        vit_mlp_dim = int(vit_dim * vit_mlp_ratio)
        self.transformer = Transformer(
            dim           = vit_dim,
            depth         = vit_depth,
            heads         = vit_heads,
            dim_head      = vit_dim_head,
            mlp_dim       = vit_mlp_dim,
            dropout       = vit_dropout,
            drop_path_rate= vit_drop_path_rate
        )

        # 6. Hybrid Head
        latent_in_dim = vit_dim * 2 if pool == "hybrid" else vit_dim
        self.to_latent = nn.Sequential(
            nn.LayerNorm(latent_in_dim),
            nn.Linear(latent_in_dim, vit_dim),
            nn.GELU()
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize ViT related parameters
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)
        # Other Linear / LayerNorm use default initialization, or supplement as needed

    def forward(self, img):
        # 1. DCNv4 Backbone
        feat = self.backbone(img)                  # (B, C_out, H_out, W_out)
        B, C, H, W = feat.shape
        assert H == self.fh and W == self.fw, \
            f"backbone output size ({H},{W}) does not match expected ({self.fh},{self.fw})"

        # 2. Patch Embedding
        x = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, N, C)
        x = self.patch_norm(x)
        x = self.patch_proj(x)                             # (B, N, vit_dim)
        B, N, D = x.shape

        # 3. CLS + Pos
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=B)
        x = torch.cat([cls_tokens, x], dim=1)              # (B, N+1, D)
        x = x + self.pos_embedding[:, : N + 1, :]
        x = self.pos_dropout(x)

        # 4. Transformer
        x = self.transformer(x)

        # 5. Pooling
        cls_token    = x[:, 0]         # (B, D)
        patch_tokens = x[:, 1:]        # (B, N, D)

        if self.pool == "cls":
            feat_vec = cls_token
        elif self.pool == "mean":
            feat_vec = x.mean(dim=1)
        else:  # "hybrid"
            patch_mean = patch_tokens.mean(dim=1)
            feat_vec   = torch.cat([cls_token, patch_mean], dim=-1)

        feat_vec = self.to_latent(feat_vec)

        # 6. Classifier head
        logits = self.mlp_head(feat_vec)
        return logits



def create_dvit_dcnv4_small(num_classes: int = 8,
                            image_size: int = 512) -> DViTDCNv4Hybrid:
    """
    Small version:
      - base_channels=48 -> stages=[48,96,192,384]
      - depths=[2,2,4,2]
      - ViT: dim=256, depth=4, heads=4
      - DropPath smaller (0.1), more generalizable
    """
    model = DViTDCNv4Hybrid(
        image_size  = image_size,
        num_classes = num_classes,
        in_channels = 3,
        base_channels = 48,
        depths      = [2, 2, 4, 2],
        mlp_ratio   = 4.0,
        drop_path_rate_backbone = 0.1,
        vit_dim     = 256,
        vit_depth   = 4,
        vit_heads   = 4,
        vit_dim_head= 64,
        vit_mlp_ratio = 4.0,
        vit_dropout   = 0.1,
        vit_drop_path_rate = 0.1,
        pool        = "hybrid",
        layer_scale_init = 1e-6
    )
    return model


def create_dvit_dcnv4_medium(num_classes: int = 8,
                             image_size: int = 512) -> DViTDCNv4Hybrid:
    """
    Medium version
      - base_channels=64 -> stages=[64,128,256,512]
      - depths=[3,3,9,3]
      - ViT: dim=320, depth=6, heads=5
    """
    model = DViTDCNv4Hybrid(
        image_size  = image_size,
        num_classes = num_classes,
        in_channels = 3,
        base_channels = 64,
        depths      = [3, 3, 9, 3],
        mlp_ratio   = 4.0,
        drop_path_rate_backbone = 0.15,
        vit_dim     = 320,
        vit_depth   = 6,
        vit_heads   = 5,
        vit_dim_head= 64,
        vit_mlp_ratio = 4.0,
        vit_dropout   = 0.1,
        vit_drop_path_rate = 0.1,
        pool        = "hybrid",
        layer_scale_init = 1e-6
    )
    return model


def create_dvit_dcnv4_large(num_classes: int = 8,
                             image_size: int = 512) -> DViTDCNv4Hybrid:
    """
    Large version:
      - base_channels=80 -> stages=[80,160,320,640]
      - depths=[3,4,18,4]
      - ViT: dim=384, depth=8, heads=6
    """
    model = DViTDCNv4Hybrid(
        image_size  = image_size,
        num_classes = num_classes,
        in_channels = 3,
        base_channels = 80,
        depths      = [3, 4, 16, 6],
        mlp_ratio   = 4.0,
        drop_path_rate_backbone = 0.2,
        vit_dim     = 384,
        vit_depth   = 7,
        vit_heads   = 8,
        vit_dim_head= 48,
        vit_mlp_ratio = 4.0,
        vit_dropout   = 0.1,
        vit_drop_path_rate = 0.15,
        pool        = "hybrid",
        layer_scale_init = 1e-6
    )
    return model


def create_vit_dcnv4_model(num_classes: int = 8, image_size: int = 512) -> DViTDCNv4Hybrid:
    return create_dvit_dcnv4_large(num_classes=num_classes, image_size=image_size)
