"""ModernBERT-backbone architecture copy — weight-compatible with
ModernBERT4Rec in datawarehouse-ai/models/career_path_transformer_modernbert.py.

Pre-norm residual blocks, RoPE (no learned absolute positions — pos_emb is
kept zeroed/frozen purely for the max_len introspection contract), GeGLU
feed-forward, bias-free Linear/LayerNorm, final LayerNorm after the last
block. The head stays weight-tied to item_emb, so every demo feature that
projects hidden states through the head (logit lens, ablation, ranking)
works unchanged.
"""

import torch
from torch import nn
from torch.nn import functional as F

from demo.bert4rec_model import BERT4Rec


class RotaryEmbedding(nn.Module):
    """GPT-NeoX-style half-split RoPE, cos/sin cached to max_len."""

    def __init__(self, head_dim: int, max_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos', freqs.cos(), persistent=False)
        self.register_buffer('sin', freqs.sin(), persistent=False)

    def forward(self, x):
        """x: (B, H, L, hd) -> same shape, positions rotated."""
        L = x.size(-2)
        cos, sin = self.cos[:L], self.sin[:L]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class ModernBertBlock(nn.Module):
    """Pre-norm block: x + Attn(LN(x)), x + GeGLU(LN(x)). Bias-free."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, rope: RotaryEmbedding):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope
        self.attn_norm = nn.LayerNorm(d_model, bias=False)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.mlp_norm = nn.LayerNorm(d_model, bias=False)
        ff = 4 * d_model
        self.mlp_in = nn.Linear(d_model, 2 * ff, bias=False)   # GeGLU: gate ‖ value
        self.mlp_out = nn.Linear(ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

    def _attend(self, x, attn_bias):
        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (B, L, self.n_heads, self.head_dim)
        q = self.rope(q.view(shape).transpose(1, 2))
        k = self.rope(k.view(shape).transpose(1, 2))
        v = v.view(shape).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.dropout_p if self.training else 0.0)
        return self.attn_out(out.transpose(1, 2).reshape(B, L, -1))

    def forward(self, x, attn_bias):
        x = x + self.dropout(self._attend(self.attn_norm(x), attn_bias))
        gate, val = self.mlp_in(self.mlp_norm(x)).chunk(2, dim=-1)
        x = x + self.dropout(self.mlp_out(F.gelu(gate) * val))
        return x


class ModernBERT4Rec(BERT4Rec):
    """BERT4Rec with the ModernBERT encoder recipe — same state-dict layout as
    the training module."""

    def __init__(self, **kw):
        super().__init__(**kw)
        d_model  = self.item_emb.embedding_dim
        n_heads  = self.encoder.layers[0].self_attn.num_heads
        n_layers = len(self.encoder.layers)
        dropout  = self.dropout.p
        max_len  = self.pos_emb.num_embeddings

        rope = RotaryEmbedding(d_model // n_heads, max_len)
        self.encoder = nn.ModuleList(
            [ModernBertBlock(d_model, n_heads, dropout, rope) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model, bias=False)
        self.final_norm = nn.LayerNorm(d_model, bias=False)
        nn.init.zeros_(self.pos_emb.weight)
        self.pos_emb.weight.requires_grad_(False)

    def _attn_bias(self, ids, dtype):
        pad = (ids == self.pad_id)
        bias = torch.zeros(ids.size(0), 1, 1, ids.size(1),
                           device=ids.device, dtype=dtype)
        return bias.masked_fill_(pad[:, None, None, :], float('-inf'))

    def _encode(self, ids):
        h = self.dropout(self.norm(self._input_embeddings(ids)))   # RoPE, no abs pos
        attn_bias = self._attn_bias(ids, h.dtype)
        for blk in self.encoder:
            h = blk(h, attn_bias)
        return self.final_norm(h)
