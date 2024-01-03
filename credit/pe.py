import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat, pack, unpack


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # attn = self.attend(dots)
        # attn = self.dropout(attn)

        # out = torch.matmul(attn, v)

        with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=True
        ):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0
            )

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TokenizationAggregation(nn.Module):
    def __init__(self, channels, patch_height, patch_width, emb_dim):
        super(TokenizationAggregation, self).__init__()

        self.patch_height = patch_height
        self.patch_width = patch_width

        self.tokenization = nn.Linear(patch_height * patch_width, emb_dim)
        self.aggregation = Attention(emb_dim)
        self.query = nn.Parameter(torch.randn(emb_dim))

    def forward(self, x):
        B, V, H, W = x.shape

        token_height = H // self.patch_height
        token_width = W // self.patch_width

        x = x.unfold(2, self.patch_height, self.patch_height).unfold(3, self.patch_width, self.patch_width)
        x = x.contiguous().view(B, V, token_height, token_width, -1)

        x = x.permute(0, 2, 3, 1, 4)  # Adjust permutation
        x = x.reshape(B, token_height, token_width, V, -1)

        # Linearly embed each variable independently
        x = self.tokenization(x)

        # Reshape to include token dimensions
        x = x.reshape(B, token_height, token_width, V, -1)

        B, N, M, V, D = x.shape

        # Average over the color channels
        x = x.mean(dim=3)

        # Flatten the token and variable dimensions for the attention mechanism
        x = x.reshape(B, N * M, D)

        # Attention mechanism using the Attention class
        x = self.aggregation(x)

        return x


class SurfacePosEmb2D(nn.Module):
    def __init__(self, image_height, image_width, patch_height, patch_width, dim, temperature=10000, cls_token=False):
        super(SurfacePosEmb2D, self).__init__()
        y, x = torch.meshgrid(
            torch.arange(image_height // patch_height),
            torch.arange(image_width // patch_width),
            indexing="ij"
        )

        assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
        omega = torch.arange(dim // 4) / (dim // 4 - 1)
        omega = 1.0 / (temperature ** omega)

        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)

        # Add an additional row for the CLS token
        if cls_token:
            cls_pe = torch.zeros(1, pe.size(1))
            pe = torch.cat((cls_pe, pe), dim=0)

        self.embedding = nn.Parameter(pe)

    def forward(self, x):
        return x + self.embedding[:, :x.size(1)].to(dtype=x.dtype, device=x.device)
