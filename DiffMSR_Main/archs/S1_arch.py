import sys
sys.path.append('/disk1/lgy/CVPR2024/DiffMSR_Main/DiffMSR_Main/archs/')
import DiffMSR_Main.archs.common as common
import DiffMSR_Main.archs.CATL as CATL
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from basicsr.utils.registry import ARCH_REGISTRY
from torch.nn import functional as F
from einops import rearrange
from .arch_util import to_2tuple, trunc_normal_


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class PG_FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(PG_FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.kernel = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )
    def forward(self, x,k_v):
        b,c,h,w = x.shape
        k_v=self.kernel(k_v).view(-1,c*2,1,1)
        k_v1,k_v2=k_v.chunk(2, dim=1)
        x = x*k_v1+k_v2  
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class PL_MSA(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size# Wh, Ww
        self.permuted_window_size = (window_size[0] // 2,window_size[1] // 2 )
        # self.permuted_window_size = (window_size[0] // 4, window_size[1] // 4)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.permuted_window_size[0] - 1) * (2 * self.permuted_window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise aligned relative position index for each token inside the window
        coords_h = torch.arange(self.permuted_window_size[0])
        coords_w = torch.arange(self.permuted_window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.permuted_window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.permuted_window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.permuted_window_size[1] - 1
        aligned_relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        aligned_relative_position_index = aligned_relative_position_index.reshape\
            (self.permuted_window_size[0],self.permuted_window_size[1],1,1,self.permuted_window_size[0]*self.permuted_window_size[1]).repeat(1,1,2,2,1)\
            .permute(0,2,1,3,4).reshape(4*self.permuted_window_size[0]*self.permuted_window_size[1],self.permuted_window_size[0]*self.permuted_window_size[1]) #  FN*FN,WN*WN
        self.register_buffer('aligned_relative_position_index', aligned_relative_position_index)
        # compresses the channel dimension of KV
        self.kv = nn.Linear(dim, dim//2, bias=qkv_bias)
        # self.kv = nn.Linear(dim, dim // 4, bias=qkv_bias)
        self.q = nn.Linear(dim,dim,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        self.kernel = nn.Sequential(
            nn.Linear(256, dim * 2, bias=False),
        )
        #############################
        ########## ORG ##############
        #############################
        # self.dim = dim
        # self.window_size = window_size  # Wh, Ww
        # self.num_heads = num_heads
        # head_dim = dim // num_heads
        # self.scale = qk_scale or head_dim ** -0.5
        #
        # # define a parameter table of relative position bias
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        #
        # # get pair-wise relative position index for each token inside the window
        # coords_h = torch.arange(self.window_size[0])
        # coords_w = torch.arange(self.window_size[1])
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        # relative_coords[:, :, 1] += self.window_size[1] - 1
        # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # self.register_buffer("relative_position_index", relative_position_index)
        #
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        #
        # self.proj_drop = nn.Dropout(proj_drop)
        #
        # trunc_normal_(self.relative_position_bias_table, std=.02)
        # self.softmax = nn.Softmax(dim=-1)
        # self.kernel = nn.Sequential(
        #     nn.Linear(256, dim * 2, bias=False),
        # )

    def forward(self, x, k_v, b, windows_size, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape

        x = x.view(b, -1, windows_size, windows_size, c)
        k_v = self.kernel(k_v).view(-1, 1, 1, 1, c * 2)
        k_v1, k_v2 = k_v.chunk(2, dim=4)
        x = x * k_v1 + k_v2
        x = x.view(b_, n, c)



        # compress the channel dimension of KV :(num_windows*b, num_heads, n//4, c//num_heads)
        # print(b_, n, c, x.shape, self.kv(x).shape)
        # print(self.q(x).shape)
        kv = self.kv(x).reshape(b_, self.permuted_window_size[0], 2, self.permuted_window_size[1], 2, 2, c//4).permute(0, 1, 3, 5, 2, 4, 6).reshape(b_, n//4, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        # kv = self.kv(x).reshape(b_, self.permuted_window_size[0], 4, self.permuted_window_size[1], 4, 4, c // 16).permute(0, 1, 3, 5, 2, 4, 6).reshape(b_, n // 16, 4, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # keep the channel dimension of Q: (num_windows*b, num_heads, n, c//num_heads)
        q = self.q(x).reshape(b_, n, 1, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))   # (num_windows*b, num_heads, n, n//4)

        relative_position_bias = self.relative_position_bias_table[self.aligned_relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.permuted_window_size[0] * self.permuted_window_size[1], -1)  # (n, n//4)
        # print(relative_position_bias.shape)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, n, n//4)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n//4) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n//4)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

        #############################
        ########## ORG ##############
        #############################

        # B_, N, C = x.shape
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        #
        # q = q * self.scale
        # attn = (q @ k.transpose(-2, -1))
        #
        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # attn = attn + relative_position_bias.unsqueeze(0)
        #
        # if mask is not None:
        #     nW = mask.shape[0]
        #     attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        #     attn = attn.view(-1, self.num_heads, N, N)
        #     attn = self.softmax(attn)
        # else:
        #     attn = self.softmax(attn)
        #
        # attn = self.attn_drop(attn)
        #
        # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        # return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.permuted_window_size}, num_heads={self.num_heads}'

    # def flops(self, n):
    #     # calculate flops for 1 window with token length of n
    #     flops = 0
    #     # qkv = self.qkv(x)
    #     flops += n * self.dim * 1.5 * self.dim
    #     # attn = (q @ k.transpose(-2, -1))
    #     flops += self.num_heads * n * (self.dim // self.num_heads) * n/4
    #     #  x = (attn @ v)
    #     flops += self.num_heads * n * n/4 * (self.dim // self.num_heads)
    #     # x = self.proj(x)
    #     flops += n * self.dim * self.dim
    #     return flops

class TransformerBlock(nn.Module):
    def __init__(self, dim,
                 num_heads,
                 ffn_expansion_factor,
                 bias,
                 LayerNorm_type,
                 window_size=16,
                 shift_size=0,
                 mlp_ratio=4.,
                 input_resolution=[64,64],
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(TransformerBlock, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.permuted_window_size = window_size//2
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.attn = PL_MSA(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        # self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = PG_FeedForward(dim, ffn_expansion_factor, bias)

    def calculate_mask(self, x_size):
        # calculate mask for original windows
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        # calculate mask for permuted windows
        h, w = x_size
        permuted_window_mask = torch.zeros((1, h // 2, w // 2, 1))  # 1 h w 1
        h_slices = (slice(0, -self.permuted_window_size), slice(-self.permuted_window_size,
                                                                -self.shift_size // 2),
                    slice(-self.shift_size // 2, None))
        w_slices = (slice(0, -self.permuted_window_size), slice(-self.permuted_window_size,
                                                                -self.shift_size // 2),
                    slice(-self.shift_size // 2, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                permuted_window_mask[:, h, w, :] = cnt
                cnt += 1

        permuted_windows = window_partition(permuted_window_mask, self.permuted_window_size)
        permuted_windows = permuted_windows.view(-1, self.permuted_window_size * self.permuted_window_size)
        # calculate attention mask
        attn_mask = mask_windows.unsqueeze(2) - permuted_windows.unsqueeze(1)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, y):
        x = y[0]
        k_v = y[1]

        b, c, h, w = x.shape
        shortcut = x
        x_size = [h, w]

        x = self.norm1(x)
        x = x.view(b, h, w, c)


        # cyclic shift
        if self.shift_size > 0:

            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, k_v, b, self.window_size, mask=self.attn_mask)  # nw*b, window_size*window_size, c
        else:
            attn_windows = self.attn(x_windows, k_v, b, self.window_size, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(b, c, h, w)
        x = shortcut + x
        # x = x + self.attn(self.norm1(x),k_v)
        x = x + self.ffn(self.norm2(x),k_v)

        return [x, k_v]


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=2, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class REFOverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=2, embed_dim=48, bias=False):
        super(REFOverlapPatchEmbed, self).__init__()

        self.proj_1 = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.proj_2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=bias)
        self.proj_3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj_3(self.proj_2(self.proj_1(x)))

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class BasicLayer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, num_blocks):

        super().__init__()

        # build blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in
             range(num_blocks)])

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, y):
        x = y[0]
        res = x
        prior = y[1]
        for blk in self.blocks:
            x, k_v = blk([x, prior])

        x = self.conv(x)
        x = res + x
        return [x, k_v]

class PLWformer(nn.Module):
    def __init__(self, 
        inp_channels=2,
        out_channels=2,
        scale=4,
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        ):

        super(PLWformer, self).__init__()
        self.scale=scale
        if self.scale == 2:
            inp_channels =8
            self.pixel_unshuffle=nn.PixelUnshuffle(2)
        elif self.scale == 1:
            inp_channels =48
            self.pixel_unshuffle=nn.PixelUnshuffle(4)
        else:
            inp_channels =2
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim*2)
        self.ref_patch_embed = REFOverlapPatchEmbed(inp_channels, dim*2)

        self.level1 = BasicLayer(dim=dim*2, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                         bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[0])

        self.level2 = BasicLayer(dim=dim*2, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                         bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[1])

        self.level3 = BasicLayer(dim=dim*2, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                                 bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[2])

        self.level4 = BasicLayer(dim=dim*2, num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                                 bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[3])
        
        self.refinement = BasicLayer(dim=dim*2, num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                                 bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[3])

        self.CA = CATL.BasicLayer(dim=dim*2, num_heads=4, ffn_expansion_factor=ffn_expansion_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=2)
        
        modules_tail = [common.Upsampler(common.default_conv, 4, int(dim*2**1), act=False),
                        common.default_conv(int(dim*2**1), out_channels, 3)]
        self.tail = nn.Sequential(*modules_tail)
        self.conv_cat = nn.Conv2d(dim * 2 * 2, dim * 2, kernel_size=3, stride=1, padding=1, bias=bias)
        
    def forward(self, inp_img, ref, k_v):
        if self.scale == 2:
            feat = self.pixel_unshuffle(inp_img)
        elif self.scale == 1:
            feat = self.pixel_unshuffle(inp_img)
        else:
            feat = inp_img 

        inp_enc_level1 = self.patch_embed(feat)
        ref_enc = self.ref_patch_embed(ref)

        fused_enc = self.CA(inp_enc_level1, ref_enc)

        out_enc_level1, _ = self.level1([fused_enc, k_v])
        out_enc_level2, _ = self.level2([out_enc_level1, k_v])
        out_enc_level3, _ = self.level3([out_enc_level2, k_v])
        out_enc_level4, _ = self.level4([out_enc_level3, k_v])
        
        out_dec, _ = self.refinement([out_enc_level4, k_v])

        out_dec = self.tail(out_dec) + F.interpolate(inp_img, scale_factor=self.scale, mode='nearest')


        return out_dec

class PE(nn.Module):
    def __init__(self,n_feats = 64, n_encoder_res = 6,scale=4):
        super(PE, self).__init__()
        self.scale=scale
        if scale == 2:
            E1=[nn.Conv2d(40, n_feats, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, True)]
        elif scale == 1:
            E1=[nn.Conv2d(96, n_feats, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, True)]
        else:
            E1=[nn.Conv2d(34, n_feats, kernel_size=3, padding=1), #10
                nn.LeakyReLU(0.1, True)]
        E2=[
            common.ResBlock(
                common.default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ]
        E3=[
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ]
        E=E1+E2+E3
        self.E = nn.Sequential(
            *E
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True)
        )
        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.pixel_unshufflev2 = nn.PixelUnshuffle(2)
    def forward(self, x,gt):
        # print(x.shape, gt.shape)

        gt0 = self.pixel_unshuffle(gt)
        if self.scale == 2:
            feat = self.pixel_unshufflev2(x)
        elif self.scale == 1:
            feat = self.pixel_unshuffle(x)
        else:
            feat = x  

        x = torch.cat([feat, gt0], dim=1)
        fea = self.E(x).squeeze(-1).squeeze(-1)
        S1_IPR = []
        fea1 = self.mlp(fea)
        S1_IPR.append(fea1)
        return fea1,S1_IPR

@ARCH_REGISTRY.register()
class DiffMSR_S1(nn.Module):
    def __init__(self, 
        n_encoder_res=6,         
        inp_channels=3, 
        out_channels=3, 
        scale=4,
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
):
        super(DiffMSR_S1, self).__init__()

        # Generator
        self.G = PLWformer(
        inp_channels=inp_channels, 
        out_channels=out_channels,
        scale = scale, 
        dim = dim,
        num_blocks = num_blocks, 
        num_refinement_blocks = num_refinement_blocks,
        heads = heads,
        ffn_expansion_factor = ffn_expansion_factor,
        bias = bias,
        LayerNorm_type = LayerNorm_type,   ## Other option 'BiasFree'
)

        self.E = PE(n_feats=64, n_encoder_res=n_encoder_res, scale=scale)


    def forward(self, x, ref, gt):

        if self.training:          
            IPRS1, S1_IPR = self.E(x,gt)

            sr = self.G(x, ref, IPRS1)

            return sr, S1_IPR
        else:
            IPRS1, _ = self.E(x,gt)

            sr = self.G(x, ref, IPRS1)

            return sr

# if __name__ == '__main__':
#     model = DiffMSR_S1(n_encoder_res=9, inp_channels=2, out_channels=2, dim=32, num_blocks=[6,6,6,6], num_refinement_blocks=6,
#                      heads=[4,4,4,4],ffn_expansion_factor=2.2, bias=False, LayerNorm_type='BiasFree')
#     # print(model)
#     img = torch.randn(1, 2, 64, 64)
#     # print(img)
#     ref = torch.randn(1, 2, 256, 256)
#     gt = torch.randn(1, 2, 256, 256)
#     #
#     # sr = model(img, ref, gt)
#     # print(sr[0].shape, sr[1][0].shape)
#