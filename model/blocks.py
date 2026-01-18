import torch
from torch import nn
from abc import abstractmethod
from einops import rearrange, repeat
from functools import partial
import einops

class StructEmbedSequential(nn.Sequential):
    """
    A sequential module that passes structure embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, struc_feat=None):
        for layer in self:
            if isinstance(layer, CrossAttentionBlock):
                x = layer(x, struc_feat)
            else:
                x = layer(x)
        return x


class CrossAttentionBlock(nn.Module):
    '''
    A module that takes structure embeddings as a second argument.
    '''
    @abstractmethod
    def forward(self, x, struc_feat=None):
        '''
        Apply the module to `x` given optional `struc_feat` structure embeddings.
        '''



def apply_norm(channels, num_groups=32, norm='group'):
    """
    Group normalization.
    """
    if norm == 'group':
        return nn.GroupNorm(num_groups, channels)
    if norm == 'instance':
        return nn.InstanceNorm3d(channels)




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            apply_norm(in_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.conv2 = nn.Sequential(
            apply_norm(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        '''
        'x': input tensor, [batch_size, in_channels, depth, height, width]
        '''
        out = self.conv1(x)
        out = self.conv2(out)
        return out + self.shortcut(x)
    
    
    
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4, dim_head=32, num_mem_kv=4, flash=True):
        super(AttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.hidden_dim = num_heads * dim_head  
        
        self.norm = apply_norm(channels)
        self.attend = Attend(flash=flash)
        
        self.mem_kv = nn.Parameter(torch.randn(2, num_heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv3d(channels, self.hidden_dim*3, kernel_size=1, bias=False)
        self.to_out = nn.Conv3d(self.hidden_dim, channels, kernel_size=1)
        
    def forward(self, x):
        b, c, d, h, w = x.shape
        
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) d x y -> b h (d x y) c', h=self.num_heads), qkv)
        
        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))
        
        out = self.attend(q, k, v)
        out = rearrange(out, 'b h (d x y) c -> b (h c) d x y', d=d, x=h, y=w)
        return self.to_out(out)
        

class LinearAttention(nn.Module):
    def __init__(self, channels,num_heads=4, dim_head=32, num_mem_kv=4):
        super(LinearAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.hidden_dim = num_heads * dim_head
        self.scale = dim_head ** -0.5
        
        self.norm = apply_norm(channels)
        
        self.mem_kv = nn.Parameter(torch.randn(2, num_heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv3d(channels, self.hidden_dim*3, kernel_size=1, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Conv3d(self.hidden_dim, channels, kernel_size=1),
            apply_norm(channels)
        )
        
    def forward(self, x):
        b, c, d, h, w = x.shape
        
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) d x y -> b h c (d x y)', h=self.num_heads), qkv)
        
        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))
        
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        
        q = q * self.scale
        
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (d x y) -> b (h c) d x y', h=self.num_heads, d=d, x=h, y=w)
        return self.to_out(out)
    

class TransformerCrossAttentionBlock(CrossAttentionBlock):
    def __init__(self, image_dim, struc_dim, num_heads=4):
        """
        Cross attention block
        Args:
            image_dim (int): Dimension of query tensor
            struc_dim (int): Dimension of structure embedding (key/value) tensor
            num_heads (int): Number of attention heads
        """
        super(TransformerCrossAttentionBlock, self).__init__()
        
        self.image_dim = image_dim
        self.struc_dim = struc_dim
        self.num_heads = num_heads
        
        # Initialize the multi-head attention module
        self.attention = nn.MultiheadAttention(
            embed_dim=self.image_dim, 
            num_heads=self.num_heads,
            kdim=self.struc_dim,
            vdim=self.struc_dim,
            batch_first=True # [batch, seq, feature]
        )

        # Normalization layers
        self.norm_q = nn.LayerNorm(self.image_dim)
        self.norm_kv = nn.LayerNorm(self.struc_dim)
        self.norm_ff = nn.LayerNorm(self.image_dim)
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(self.image_dim, self.image_dim * 4),
            nn.SiLU(),
            nn.Linear(self.image_dim * 4, self.image_dim)
        )        
        
    def forward(self, img_feat, struc_feat=None):
        """
        Forward pass of cross attention
        Args:
            img_feat: [B, C, D, H, W]
            struc_feat: [B, target_len, emb_dim]. If None, skip cross attention                               
        Returns:
            Output: torch.Tensor with the dimensionas as x
            Attention weights: torch.Tensor
        """
        
        # Flatten image features
        # img_feat_flat = einops.rearrange(img_feat, 'b c d h w -> b (d h w) c')
        img_feat_flat = img_feat
        if struc_feat is None:
            attn_norm = self.norm_ff(img_feat_flat)
            ff_out = self.ff(attn_norm)
            output = img_feat_flat + ff_out

        else: 
            # # Flatten structure features
            # struc_feat_flat = einops.rearrange(struc_feat, 'b n l d -> b (n l) d')   
            # First residual block: cross attention
            img_feat_norm = self.norm_q(img_feat_flat)        
            struc_feat_norm = self.norm_kv(struc_feat)
                    
            attn, _ = self.attention(
                query=img_feat_norm,
                key=struc_feat_norm,
                value=struc_feat_norm,
                need_weights=False
            )
            attn = attn + img_feat_flat

            # Second residual block: feedforward
            attn_norm = self.norm_ff(attn)
            ff_out = self.ff(attn_norm)
            output = attn + ff_out
        
        # Reshape the output
        # output = einops.rearrange(output, 'b (d h w) c -> b c d h w', 
        #                           d = img_feat.shape[2],
        #                           h = img_feat.shape[3],
        #                           w = img_feat.shape[4]
        #                           )
        return output