

def create_cytofm() -> nn.Module:
    """
    Create a CytoFM model for binary classification.
    """
    """
CytoFM backbone architecture (ViT-Base/16) for feature extraction.

This matches the architecture used in the CytoFM repo's example notebook:
- Loads ckpt['teacher'] backbone weights into iBOT's vit_base(patch_size=16)
- Produces a CLS embedding of shape [B, 768] for input [B, 3, 224, 224]

Source alignment:
- CytoFM loading notebook: load_cytofm_weights.ipynb
- Backbone definition is based on iBOT's VisionTransformer implementation.

Notes:
- This file defines the architecture only. Weight loading can be done elsewhere.
- We keep module/parameter names consistent with iBOT so state_dict loading works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, List

import math
import torch
import torch.nn as nn

try:
    # Used for trunc_normal_ init, same behavior as common ViT codepaths.
    from timm.layers import trunc_normal_
except ImportError as e:
    raise ImportError(
        "timm is required for CytoFM ViT-Base backbone. Install it with: pip install timm"
    ) from e


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Stochastic depth per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale if qk_scale is not None else head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_prob: float = 0.0,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = lambda d: nn.LayerNorm(d, eps=1e-6),
        init_values: float = 0.0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0.0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn  # type: ignore[return-value]

        if self.gamma_1 is None:
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class VisionTransformer(nn.Module):
    """
    ViT backbone with iBOT/DINO-style naming so CytoFM/iBOT backbone weights load cleanly.
    """
    def __init__(
        self,
        img_size: List[int] = [224],
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 0,              # keep 0 for backbone (head = Identity)
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,             # IMPORTANT: iBOT vit_base sets qkv_bias=True
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Callable[[int], nn.Module] = lambda d: nn.LayerNorm(d, eps=1e-6),
        return_all_tokens: bool = False,
        init_values: float = 0.0,
        use_mean_pooling: bool = False,
        masked_im_modeling: bool = False,  # kept for compatibility; not needed for CytoFM feature extraction
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.return_all_tokens = return_all_tokens

        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path_prob=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None

        # classifier head (Identity when num_classes == 0)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        # masked image modeling (kept for naming compatibility)
        self.masked_im_modeling = masked_im_modeling
        if masked_im_modeling:
            self.masked_embed = nn.Parameter(torch.zeros(1, embed_dim))

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def mask_model(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x is [B, C, H, W] after patch_embed conv
        x = x.permute(0, 2, 3, 1)  # [B, H', W', C]
        x[mask, :] = self.masked_embed.to(x.dtype)
        return x.permute(0, 3, 1, 2)

    def prepare_tokens(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        if mask is not None:
            x = self.mask_model(x, mask)

        x = x.flatten(2).transpose(1, 2)  # [B, N, C]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward(
        self,
        x: torch.Tensor,
        return_all_tokens: Optional[bool] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.masked_im_modeling:
            if mask is None:
                raise ValueError("masked_im_modeling=True requires a non-None mask.")
            x = self.prepare_tokens(x, mask=mask)
        else:
            x = self.prepare_tokens(x, mask=None)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.fc_norm is not None:
            x[:, 0] = self.fc_norm(x[:, 1:, :].mean(1))

        rat = self.return_all_tokens if return_all_tokens is None else return_all_tokens
        if rat:
            return x
        return x[:, 0]


def vit_base(patch_size: int = 16, **kwargs) -> VisionTransformer:
    # iBOT vit_base: embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        **kwargs,
    )


def create_cytofm_vit_base(
    device: torch.device,
    img_size: int = 224,
    patch_size: int = 16,
    return_all_tokens: bool = False,
) -> nn.Module:
    """
    Create the CytoFM backbone architecture (ViT-Base/16).

    Args:
        device: Device to move the model to (caller owns device resolution)
        img_size: Input image size (default 224 to match common CytoFM patch usage)
        patch_size: Patch size (default 16; matches CytoFM/iBOT usage)
        return_all_tokens: If True, forward() returns [B, N+1, 768] tokens. Otherwise returns [B, 768] CLS.

    Returns:
        ViT-Base backbone model moved to device, ready for weight loading and inference.
    """
    model = vit_base(
        patch_size=patch_size,
        img_size=[img_size],
        in_chans=3,
        num_classes=0,               # backbone-only (head is Identity)
        return_all_tokens=return_all_tokens,
        use_mean_pooling=False,      # iBOT default in their ViT backbone path
        masked_im_modeling=False,
    ).to(device)

    return model
