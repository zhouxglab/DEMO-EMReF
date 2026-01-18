import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import sys
sys.path.append("/jiel/DEMO-EMReF")
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple,to_3tuple, trunc_normal_
# from model.Loss import CombinedLoss
from einops import rearrange

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0. or not training:
        return x
    
    keep_prob = 1 - drop_prob
    # Generate a random tensor for drop path, with the same shape as x but keeping the batch and channel dimensions.
    shape = (x.shape[0], x.shape[1]) + (1,) * (x.ndim - 2)  # keep batch and channel, expand for spatial dims
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize (0 or 1)
    
    # Apply drop path by masking and scaling
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    """MLP with two fully connected layers and activation."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):

        # Flatten x except for batch and channel dimensions
        batch_size, channels = x.shape[0], x.shape[2]
        # x = x.flatten(start_dim=1)  # Flatten height, width, depth into a single dimension

        # Forward pass through the MLP
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

def window_partition(x, window_size):
    """
    Partition the input tensor into windows.
    x: (b, c, h, w, d)  - 输入的5D张量
    window_size: (wd, wh, ww)  - 每个维度的窗口大小
    """
    if isinstance(window_size, int):
        window_size = (window_size, window_size, window_size)
    b, h, w, d,c = x.shape 

    wd, wh, ww = window_size  
 
    x = x.view(b, d // wd, wd, h // wh, wh, w // ww, ww, c)  # b, c, h//wh, wh, w//ww, ww, d//wd, wd
  
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()  
    
    windows = windows.view(-1, wd, wh, ww, c)

    return windows
def window_reverse(windows, window_size, h, w, d):
    """
    Reconstruct the input tensor from windows.
    windows: (num_windows, window_size * window_size * window_size, c)
    window_size: (wh, ww, wd)
    h, w, d: height, width, depth of original input
    """
   
    b = int(windows.shape[0] // (h * w * d/ window_size / window_size/ window_size))  # Get batch size

    x = windows.reshape(b, h // window_size, w // window_size, d // window_size, window_size, window_size, window_size, -1)

    # Permute the dimensions to match the original order
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()

    # Reshape the tensor to the final output shape
    x = x.view(b, d, h, w, -1)

    return x

def grid_shuffle(x, h, w, d, c, interval_size):
    """
    Shuffle the 3D tensor grid.

    Args:
        x: (b, c, h, w, d) - Input tensor
        h (int): Height of the image
        w (int): Width of the image
        d (int): Depth of the image
        c (int): Channel of the feature map
        interval_size (int): Interval size for grid shuffling

    Returns:
        shuffled: (b, h // interval_size, w // interval_size, d // interval_size, c)
    """
    x = x.view(-1, h // interval_size, interval_size, w // interval_size, interval_size, d // interval_size, interval_size, c)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    shuffled = x.view(-1, h // interval_size, w // interval_size, d // interval_size, c)
    return shuffled

def grid_unshuffle(x, b, h, w, d, interval_size):
    """
    Reverse the grid shuffle operation for 3D tensor.

    Args:
        x: (b * h * w * d // (interval_size * interval_size * interval_size), h // interval_size, w // interval_size, d // interval_size, c) - Shuffled tensor
        b: Batch size
        h (int): Height of the image
        w (int): Width of the image
        d (int): Depth of the image
        interval_size (int): Interval size for grid unshuffling

    Returns:
        x: (b, c, h, w, d) - Unshuffled tensor
    """
    x = x.view(b, interval_size, interval_size, interval_size, h // interval_size, w // interval_size, d // interval_size, -1)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(b, h, w, d,-1)
    return x

class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4  # Same as 2D but adapted to 3D
        self.pos_proj = nn.Linear(3, self.pos_dim)  # Changed to 3 for 3D position projection
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias for 3D.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height, width, and depth of the window (Wh, Ww, Wd).
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        # Ensure window_size is a tuple (Wh, Ww, Wd)
        if isinstance(window_size, int):
            self.window_size = (window_size, window_size, window_size)
        else:
            self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # Define a parameter table of relative position bias for 3D
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1), num_heads))  # 3D bias table

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        """
        Args:
            x: input features with shape of (num_windows * b, n, c)
            rpi: relative position index (3D)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww*Wd, Wh*Ww*Wd) or None
        """
        b_, n, c = x.shape
        qkv = x.reshape(b_, n, 3, self.num_heads, c // self.num_heads // 3).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Separate q, k, v

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2], self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # 3D relative position
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, (Wh*Ww*Wd), (Wh*Ww*Wd)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c // 3)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FAB(nn.Module):
    r""" Fused Attention Block for 3D data.
    
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution (h, w, d).
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size (Wh, Ww, Wd).
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=(7, 7, 7),
                 shift_size=(0, 0, 0),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (h, w, d)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # Make sure shift_size < window_size
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
        
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        
        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size, rpi_sa, attn_mask):
        h, w, d = x_size  # 3D sizes

        b, _, c = x.shape


        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, d, c)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
            attn_mask = attn_mask
        else:
            shifted_x = x
            attn_mask = None

        # partition windows (3D)
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size*window_size*window_size, c
        
        x_windows = x_windows.view(-1, self.window_size * self.window_size* self.window_size, c)  # nw*b, window_size^3, c

        # W-MSA/SW-MSA
        attn_windows = self.attn(self.qkv(x_windows), rpi=rpi_sa, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w, d)  # b h' w' d' c

        # reverse cyclic shift
        if self.shift_size >0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(b, h * w * d, c)

        # FFN
        x = shortcut + self.drop_path(attn_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SEModule(nn.Module):
    def __init__(self, channels, rd_channels=None, bias=True):
        super(SEModule, self).__init__()
        self.fc1 = nn.Conv3d(channels, rd_channels, kernel_size=1, bias=bias)  # 3D卷积
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv3d(rd_channels, channels, kernel_size=1, bias=bias)  # 3D卷积
        self.gate = nn.Sigmoid()

    def forward(self, x):
        # 全局池化
        x_se = x.mean((2, 3, 4), keepdim=True)  # 对深度、高度、宽度进行池化
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


class FusedConv(nn.Module):
    """ Fused Conv Block for 3D data.
        Args:
            num_feat (int): Number of input channels.
            expand_size (int): expand size
            attn_ratio (int): Ratio of attention hidden dim to embedding dim.
    """

    def __init__(self, num_feat, expand_size=4, attn_ratio=4):
        super(FusedConv, self).__init__()
        mid_feat = num_feat * expand_size
        rd_feat = int(mid_feat / attn_ratio)
        
        # 3D层
        self.pre_norm = nn.LayerNorm(num_feat)  # 归一化层
        self.fused_conv = nn.Conv3d(num_feat, mid_feat, 3, 1, 1)  # 3D卷积
        self.norm1 = nn.LayerNorm(mid_feat)
        self.act1 = nn.GELU()
        self.se = SEModule(mid_feat, rd_feat, bias=True)
        self.conv3_1x1 = nn.Conv3d(mid_feat, num_feat, 1, 1)  # 3D卷积

    def forward(self, x, x_size, rpi, mask):
        shortcut = x
        d, h, w = x_size  # 深度、高度、宽度
        b, _, c = x.shape  # batch_size, channels, depth, height, width
        
        # 调整输入张量形状为 (b, c, d, h, w)
        x = x.view(b, d, h, w,c)

        x = self.pre_norm(x)
        x = x.permute(0, 4, 1, 2, 3)

        x = self.fused_conv(x)

        x = x.permute(0, 2, 3, 4, 1)

        # 激活并应用归一化
        x = self.act1(self.norm1(x))
        x = x.permute(0, 4, 1, 2, 3)

        # Squeeze-and-Excitation模块
        x = self.se(x)
        
        # 1x1卷积操作
        x = self.conv3_1x1(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = x.view(b, d*h*w, c)
        # print(f"x.shape: {x.shape}, shortcut.shape: {shortcut.shape}")

        return x + shortcut  # 残差连接



class AffineTransform(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias for 3D data.
    It supports both shifted and non-shifted windows.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height, width, and depth of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim,
                 window_size,
                 num_heads,
                 qk_scale=None,
                 attn_drop=0.,
                 position_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww, Wd)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, h, w, d):
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        group_size = (h, w, d)

        if self.position_bias:
            # Generate relative position biases in 3D
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            position_bias_d = torch.arange(1 - group_size[2], group_size[2], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w, position_bias_d]))  # 3, 2Gh-1, 2Gw-1, 2Gd-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()

            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords_d = torch.arange(group_size[2], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_d]))  # 3, Gh, Gw, Gd
            coords_flatten = torch.flatten(coords, 1)  # 3, Gh*Gw*Gd
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Gh*Gw*Gd, Gh*Gw*Gd
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw*Gd, Gh*Gw*Gd, 3
            relative_coords[:, :, 0] += group_size[0] - 1
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 2] += group_size[2] - 1
            relative_coords = relative_coords.sum(-1)  # Gh*Gw*Gd, Gh*Gw*Gd

            pos = self.pos(biases)  # 2Gh-1 * 2Gw-1 * 2Gd-1, heads
            relative_position_bias = pos[relative_coords.view(-1)].view(
                group_size[0] * group_size[1] * group_size[2], group_size[0] * group_size[1] * group_size[2], -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw*Gd, Gh*Gw*Gd
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        x = self.attn_drop(attn)
        x = x @ v

        return x


class GridAttention(nn.Module):
    r""" Grid based multi-head self-attention (G-MSA) module with relative position bias for 3D data.
        It supports both shifted and non-shifted windows.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
            attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0.
            proj_drop (float, optional): Dropout ratio of output. Default: 0.0.
        """
    
    def __init__(self, window_size, dim, num_heads, qk_scale=None, attn_drop=0., position_bias=True):
        super().__init__()
        self.window_size = window_size
        self.dim = dim
        self.num_heads = num_heads
        self.attn_transform1 = AffineTransform(dim,
                                               window_size=to_3tuple(self.window_size),
                                               num_heads=num_heads,
                                               qk_scale=qk_scale,
                                               attn_drop=attn_drop,
                                               position_bias=position_bias)
        self.attn_transform2 = AffineTransform(dim,
                                               window_size=to_3tuple(self.window_size),
                                               num_heads=num_heads,
                                               qk_scale=qk_scale,
                                               attn_drop=attn_drop,
                                               position_bias=position_bias)

    def forward(self, qkv, grid, h, w, d):
        """
        Args:
            qkv: input features with shape of (num_windows*b, n, c).
            grid: 3D grid information.
            h, w, d: dimensions of the input (height, width, depth).
        """
        b_, n, c = grid.shape
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        grid = grid.reshape(b_, n, self.num_heads, -1).permute(0, 2, 1, 3)

        q, k, v = qkv[0], qkv[1], qkv[2]  # Split into query, key, and value
        x = self.attn_transform1(grid, k, v, h, w, d)
        x = self.attn_transform2(q, grid, x, h, w, d)
        x = x.transpose(1, 2).reshape(b_, n, c)

        return x


class GAB(nn.Module):
    r""" Grid Attention Block for 3D data.

        Args:
            dim (int): Number of input channels.
            grid_size (int): Grid size.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
            attn_drop (float, optional): Attention dropout rate. Default: 0.0.
            drop (float, optional): Dropout rate. Default: 0.0.
            drop_path (float, optional): Stochastic depth rate. Default: 0.0.
        """
    
    def __init__(self,
                 window_size,
                 interval_size,
                 dim,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 drop=0.,
                 drop_path=0.,
                 mlp_ratio=2):
        super().__init__()
        self.window_size = window_size
        self.interval_size = interval_size
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.grid_proj = nn.Linear(dim, dim // 2)
        self.shift_size = window_size // 2

        self.grid_attn = GridAttention(
            window_size,
            dim // 2,
            num_heads=num_heads // 2,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
        )
        self.window_attn = WindowAttention(
            dim // 4,
            window_size=to_3tuple(self.window_size),
            num_heads=num_heads // 2,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        self.window_attn_s = WindowAttention(
            dim // 4,
            window_size=to_3tuple(self.window_size),
            num_heads=num_heads // 2,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.fc = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        mip_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mip_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x, x_size, rpi_sa, mask):
        h, w, d = x_size

        b, _, c = x.shape
        shortcut = x

        qkv = self.qkv(x)
        x_window, x_qkv = torch.split(qkv, c * 3 // 2, dim=-1)

        x = x.view(b, h, w, d, c)
        Gh, Gw, Gd = h // self.interval_size, w // self.interval_size, d // self.interval_size
        x_grid = self.grid_proj(grid_shuffle(x, h, w, d, c, self.interval_size).view(-1, Gh * Gw * Gd, c))
        x_qkv = grid_shuffle(x_qkv, h, w, d, c * 3 // 2, self.interval_size).view(-1, Gh * Gw * Gd, c * 3 // 2)

        # GSA
        x_grid_attn = self.grid_attn(x_qkv, x_grid, Gh, Gw, Gd).view(-1, Gh, Gw, Gd, c // 2)
        x_grid_attn = grid_unshuffle(x_grid_attn, b, h, w, d, self.interval_size).view(b, h * w * d, c // 2)

        x_window, x_window_s = torch.split(x_window.view(b, h, w, d, c * 3 // 2), c * 3 // 4, dim=-1)
        x_window = window_partition(x_window, self.window_size)  # nw*b, window_size, window_size, window_size, c
        x_window = x_window.view(-1, self.window_size * self.window_size * self.window_size, c * 3 // 4)

        x_window_s = torch.roll(x_window_s, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
        x_window_s = x_window_s.view(-1, self.window_size * self.window_size * self.window_size, c * 3 // 4)
        
        # MSA
        x_win_attn = self.window_attn(x_window, rpi=rpi_sa, mask=None).view(-1, self.window_size, self.window_size,
                                                                            self.window_size, c // 4)
        x_win_attn = window_reverse(x_win_attn, self.window_size, h, w, d).view(b, h * w * d, c // 4)

        x_win_s_attn = self.window_attn_s(x_window_s, rpi=rpi_sa, mask=None).view(-1, self.window_size,
                                                                                  self.window_size,
                                                                                  self.window_size,
                                                                                  c // 4)
        x_win_s_attn = window_reverse(x_win_s_attn, self.window_size, h, w, d).view(b, h * w * d, c // 4)

        x_win_s_attn = torch.roll(x_win_s_attn, shifts=(self.shift_size, self.shift_size), dims=(1, 2))


        x_win_attn = torch.cat([x_win_attn, x_win_s_attn], dim=-1)
        x = torch.cat([x_win_attn, x_grid_attn], dim=-1)
        x = self.norm1(self.fc(x))

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


class AttenBlocks(nn.Module):
    """ A series of attention blocks for one RHAG, adapted for 3D data.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution (height, width, depth).
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        interval_size (int): Size of intervals for local/global attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 interval_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (height, width, depth)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        blk = []
        for i in range(depth):
            if i % 2 == 0:
                blk.append(
                    FusedConv(  # Modified to 3D convolution
                        num_feat=dim,
                        expand_size=6,
                        attn_ratio=2
                    )
                )
                blk.append(
                    FAB(
                        dim=dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0,  # Shift size remains 0 for first block
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer
                    )
                )
            else:
                blk.append(
                    FAB(
                        dim=dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=window_size // 2,  # Shift size is half of window_size for others
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer
                    )
                )
        
        self.blocks = nn.ModuleList(blk)

        # Global attention block (modified for 3D)
        self.gab = GAB(
            window_size=window_size,
            interval_size=interval_size,
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            drop=drop,
            drop_path=0.,  # No stochastic depth here
            mlp_ratio=mlp_ratio
        )

        # Learnable scale parameter
        self.scale = nn.Parameter(torch.empty(dim))
        trunc_normal_(self.scale, std=.02)

        # Patch merging layer (if downsample is provided)
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, params):
        """ Forward pass through all blocks and the global attention block.
        
        Args:
            x (tensor): Input tensor of shape (batch_size, channels, height, width, depth).
            x_size (tuple): A tuple (height, width, depth) of the input size.
            params (dict): Dictionary containing additional parameters such as attention masks.
        """

        for blk in self.blocks:
            x = blk(x, x_size, params['rpi_sa'], params['attn_mask'])

        # Apply global attention block (GAB)
        y = self.gab(x, x_size, params['rpi_sa'], params['attn_mask'])
        x = x + y * self.scale  # Scale the output and add residual

        # Apply downsample if present
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class RHTB3D(nn.Module):
    """Residual Hybrid Transformer Block for 3D data.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution (height, width, depth).
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 interval_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv'):
        super(RHTB3D, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = AttenBlocks(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            interval_size=interval_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv3d(dim, dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv = nn.Identity()

        self.patch_embed = PatchEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size, params):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, params), x_size))) + x

class PatchEmbed3D(nn.Module):
    r""" 3D Image to Patch Embedding
    Args:
        img_size (tuple): 3D image size (height, width, depth).
        patch_size (int): Patch size.
        in_chans (int): Number of input image channels.
        embed_dim (int): Embedding dimension.
        norm_layer (nn.Module): Normalization layer.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_3tuple(img_size)  # Adjust to 3D tuple
        patch_size = to_3tuple(patch_size)  # Adjust to 3D tuple
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # Flatten the tensor for 3D (b, Ph*Pw*Pd, c)
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchUnEmbed3D(nn.Module):
    r""" 3D Patch Unembedding
    Args:
        img_size (tuple): 3D image size.
        patch_size (int): Patch size.
        in_chans (int): Number of input channels.
        embed_dim (int): Embedding dimension.
        norm_layer (nn.Module): Normalization layer.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_3tuple(img_size)  # Adjust to 3D tuple
        patch_size = to_3tuple(patch_size)  # Adjust to 3D tuple
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, x_size[0], x_size[1], x_size[2])  # (b, c, h, w, d)
        return x
    

class PixelShuffle3D(nn.Module):
    """3D PixelShuffle with convolutional layer."""
    def __init__(self, upscale_factor: int):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, channels, depth, height, width = input.size()

        # Ensure that the number of channels is divisible by upscale_factor^3
        upscale_factor_squared = self.upscale_factor ** 3
        if channels % upscale_factor_squared != 0:
            raise ValueError(f"Input channels ({channels}) must be divisible by {upscale_factor_squared}.")
        
        # Apply a 3D convolution (could be any Conv3d layer)
        # Here we directly multiply channels by upscale_factor^3
        output = F.conv3d(input, weight=torch.ones(channels, channels, 1, 1, 1).to(input.device))  # Placeholder conv

        # Reshape the output tensor to split channels into C * upscale_factor^3
        output_reshaped = output.view(batch_size, channels // upscale_factor_squared,
                                      self.upscale_factor, self.upscale_factor, self.upscale_factor,
                                      depth, height, width)
        
        # Permute dimensions to rearrange the channels and spatial dimensions
        output = output_reshaped.permute(0, 1, 5, 6, 7, 2, 3, 4)
        
        # Reshape the output to get the final result with the expanded spatial dimensions
        output = output.contiguous().view(batch_size, channels // upscale_factor_squared, 
                                          depth * self.upscale_factor, 
                                          height * self.upscale_factor, 
                                          width * self.upscale_factor)
        
        return output

class Upsample3D(nn.Module):
    """Upsample module for 3D data."""
    def __init__(self, scale, num_feat):
        super(Upsample3D, self).__init__()

        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                # Step 1: Increase the channel dimension
                m.append(nn.Conv3d(num_feat, 4 * num_feat, 3, 1, 1))  # Increase channels to 4 * num_feat
                # Step 2: Apply PixelShuffle to increase spatial dimensions
                m.append(PixelShuffle3D(2))  # Upsample the spatial dimensions by a factor of 2
        elif scale == 3:
            # Step 1: Increase the channel dimension
            m.append(nn.Conv3d(num_feat, 9 * num_feat, 3, 1, 1))  # Increase channels to 9 * num_feat
            # Step 2: Apply PixelShuffle to increase spatial dimensions
            m.append(PixelShuffle3D(3))  # Upsample the spatial dimensions by a factor of 3
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')

        self.upsample = nn.Sequential(*m)

    def forward(self, x):

        x = self.upsample(x)

        return x


@ARCH_REGISTRY.register()

class HMANet3D(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=(7,7,7),
                 interval_size=4,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs):
        super(HMANet3D, self).__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 48
        self.img_range = img_range
        self.mean = torch.zeros(1, 1, 1, 1, 1)  # 对应于3D输入

        self.upscale = upscale
        self.upsampler = upsampler

        # relative position index (3D)
        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv3d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths) # num_layers = 4
        self.embed_dim = embed_dim # embed_dim = 32
        self.ape = ape #ape = False
        self.patch_norm = patch_norm   # patch_norm = true
        self.num_features = embed_dim   # num_features = 32
        self.mlp_ratio = mlp_ratio      # mlp_ratio = 2

        # split image into non-overlapping patches (3D)
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image (3D)
        self.patch_unembed = PatchUnEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding (3D)
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build Residual Hybrid Attention Groups (RHAG) for 3D
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHTB3D(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1], patches_resolution[2]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                interval_size=interval_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv3d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv_after_body = nn.Identity()

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv3d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample3D(upscale, num_feat)  # 需要定义3D的Upsample模块
            self.conv_last = nn.Conv3d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def calculate_rpi_sa(self):
        # 计算三维相对位置索引
        coords_d = torch.arange(self.window_size)
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w]))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 2] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_coords[:, :, 1] *= 2 * self.window_size - 1
        relative_coords[:, :, 2] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        return relative_position_index

    def calculate_mask(self, x_size):
        h, w, d = x_size  
        img_mask = torch.zeros((1, h, w, d, 1))  # 1, c, h, w, d, 1
        if isinstance(self.window_size, tuple):
            window_size_depth = self.window_size[0]
            window_size_height = self.window_size[1]
            window_size_width = self.window_size[2]
        else:
            window_size_depth = self.window_size
            window_size_height = self.window_size
            window_size_width = self.window_size

        shift_size_depth = window_size_depth // 2
        shift_size_height = window_size_height // 2
        shift_size_width = window_size_width // 2

        # 创建深度、高度、宽度的切片
        d_slices = (slice(0, -window_size_depth), slice(-window_size_depth, -shift_size_depth), slice(-shift_size_depth, None))
        h_slices = (slice(0, -window_size_height), slice(-window_size_height, -shift_size_height), slice(-shift_size_height, None))
        w_slices = (slice(0, -window_size_width), slice(-window_size_width, -shift_size_width), slice(-shift_size_width, None))


        cnt = 0
        # 遍历深度、高度、宽度的切片
        for d in d_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, d, :] = cnt  # 注意调整顺序
                    cnt += 1
        
        if isinstance(self.window_size, int):
            self.window_size = (self.window_size, self.window_size, self.window_size)

        # 调用 window_partition 处理掩码
        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, window_size, 1

        # 展开并生成 attention mask
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def set_loss(self):
        self.loss_func = None
    def forward_features(self, x):
        # x = x['density']
        x_size = (x.shape[2],x.shape[3], x.shape[4])

        attn_mask = self.calculate_mask(x_size).to(x.device)

        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}
        x = self.patch_embed(x)
      
        if self.ape:
            x = x + self.absolute_pos_embed

        x = self.pos_drop(x)
   
        for layer in self.layers:
            x = layer(x, x_size, params)
            
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x
    
    def forward(self, x,istrain=True):
        # print("x: ", type(x))
        if istrain:
            label = x['label']
            x = x["density"]

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)

            x = self.conv_after_body(self.forward_features(x)) + x

            x = self.conv_before_upsample(x)

            x = self.conv_last(x)

        if istrain:

            total_loss,mse_loss,ssim_loss = self.loss_func(x, label)

            diff = torch.abs(x - label)
            num_large_diff = (diff > 0.5).sum().item()

            return total_loss,mse_loss,ssim_loss, x, label  
        else:
            return x 


