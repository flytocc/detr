import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from util.misc import NestedTensor


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class FishEyePatchEmbedding(nn.Module):
    def __init__(self, input_size=736, patch_size=16, patch_length=768, embed_dim=768, transformer_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_length = patch_length
        assert patch_length % patch_size == 0

        self.input_size = input_w, input_h = _pair(input_size)
        assert input_w == input_h
        assert input_w % (2 * patch_size) == 0, input_h % (2 * patch_size) == 0

        degrees, radius = self.polar()  # shape: (input_w, input_h)
        # degrees_intervals = self.degrees_intervals(interval=45)  # shape: (M, 2), item: (low, high)
        radius_intervals = self.radius_intervals()  # shape: (N, 2), item: (low, high]

        masks = []
        # for low_degrees, high_degrees in degrees_intervals:
        #     keep_degrees = (low_degrees < degrees) & (degrees < high_degrees)
        offset = torch.tensor([(input_w - 1) / 2, (input_h - 1) / 2])
        for low_radius, high_radius in radius_intervals:
            mask = (low_radius < radius) & (radius <= high_radius)
            x, y = (mask.nonzero() - offset).unbind(dim=-1)
            deg = torch.atan2(-y, x) * 180. / math.pi
            mask = mask.flatten().nonzero().squeeze()
            masks.append(mask[deg.argsort()])  # shape: (n,); item: index
        self.masks_size = tuple(map(len, masks))  # shape: (N,)
        self.register_buffer('masks', torch.cat(masks))  # shape: (num_pixels,)

        self.embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)

        grid_r = np.arange(input_w // 2 // patch_size) * patch_size
        grid_a = np.arange(patch_length // patch_size) / (patch_length // patch_size) * 2 * math.pi
        grid = np.meshgrid(grid_r, grid_a)  # here r goes first
        grid = np.stack(grid, axis=0)
        pos_embed = get_2d_sincos_pos_embed_from_grid(transformer_dim, grid)
        pos_embed = torch.as_tensor(pos_embed, dtype=torch.float32)
        pos_embed = pos_embed.view(1, len(grid_a), len(grid_r), transformer_dim).permute(0, 3, 1, 2)
        self.register_buffer('pos_embed', pos_embed)

    def degrees_intervals(self, interval):
        assert 360 % interval == 0
        degrees = torch.arange(-180, 180 + interval, interval)
        assert len(degrees) == 360 // interval + 1
        return torch.stack((degrees[:-1], degrees[1:]), dim=-1).tolist()

    def radius_intervals(self, mode='equal_interval', data=None):
        input_size = min(self.input_size)
        R = input_size // 2
        if mode == 'equal_interval':
            # r_n = n * interval; n = 0, 1, 2, ..., R / interval
            interval = data or 1
            assert R % interval == 0
            radius = torch.arange(0, R + interval, interval)
            assert len(radius) == R // interval + 1
        elif mode == 'equal_area':
            # r_n = √n * (R / √N); n = 0, 1, 2, ..., N
            # N = (H // 2 // 16)**2 * math.pi // 8
            N = data
            assert N is not None
            radius = torch.arange(N + 1)**.5 * (R / N**.5)
        else:
            raise NotImplementedError
        return torch.stack((radius[:-1], radius[1:]), dim=-1).tolist()

    def polar(self):
        input_w, input_h = self.input_size
        y = torch.arange(input_h) - (input_h - 1) / 2  # [-1.5, -0.5, 0.5, 1.5]
        x = torch.arange(input_w) - (input_w - 1) / 2  # [-1.5, -0.5, 0.5, 1.5]
        y, x = torch.meshgrid(y, x)
        degrees = torch.atan2(-y, x) * 180. / math.pi  # shape: (input_w, input_h); -180 < item <= 180
        radius = (x**2 + y**2)**.5  # shape: (input_w, input_h)
        return degrees, radius

    def forward(self, x: torch.Tensor):
        N, C, H, W = x.shape
        assert self.input_size == (W, H)
        assert C == 3

        with torch.no_grad():
            patches = x.view(N, C, -1)[:, :, self.masks]
            patches = patches.split(self.masks_size, dim=-1)
            patches = [
                F.interpolate(patch, size=self.patch_length, mode='linear', align_corners=True)  # shape: (N, 3, patch_length)
                for patch in patches
            ]
            patches = torch.stack(patches, dim=2)  # shape: (N, 3, num_patches, patch_length)

        patches = self.embed(patches)  # shape: (N, embed_dim, num_patches//patch_size, patch_length//patch_size)

        return patches


class Backbone(FishEyePatchEmbedding):

    def forward(self, tensor_list: NestedTensor):
        x = super().forward(tensor_list.tensors)
        pos_embed = self.pos_embed.expand(x.size(0), -1, -1, -1)

        m = tensor_list.mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]

        return [NestedTensor(x, mask)], [pos_embed]


def build_backbone(args):
    model = Backbone(input_size=736, patch_size=16, patch_length=768, embed_dim=768, transformer_dim=args.hidden_dim)
    model.num_channels = model.embed_dim
    return model


if __name__ == '__main__':
    from PIL import Image

    image = Image.open('fisheye_BUPT@xuebaduimian_h3.65_10.30_08985.jpg').convert("RGB")
    image = np.array(image)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

    patch_embed = FishEyePatchEmbedding()
    params = sum(p.numel() for p in patch_embed.parameters() if p.requires_grad)
    print(params / 1e6)

    patches = patch_embed(image)
