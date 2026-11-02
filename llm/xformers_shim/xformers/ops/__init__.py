import torch
import torch.nn.functional as F
from .fmha.attn_bias import (AttentionBias, BlockDiagonalCausalMask,
                             LowerTriangularMaskWithTensorBias)

__all__ = ["AttentionBias", "memory_efficient_attention_forward"]


def memory_efficient_attention_forward(query, key, value, attn_bias=None, p=0.0, scale=None):
    """query/key/value: [B, N, H, D]. Supports no bias, causal block-diagonal bias and (lower-triangular + tensor) bias."""
    if p != 0.0:
        raise NotImplementedError("dropout is not supported in the shim")
    q, k, v = (t.transpose(1, 2) for t in (query, key, value))  # [B, H, N, D]
    if attn_bias is None:
        out = F.scaled_dot_product_attention(q, k, v, scale=scale)
    elif isinstance(attn_bias, BlockDiagonalCausalMask):
        if query.shape[0] != 1:
            raise ValueError(f"block-diagonal bias expects batch 1, got shape {tuple(query.shape)} on {query.device}")
        if getattr(attn_bias, "window", None) is not None:
            raise NotImplementedError("sliding window is not supported in the shim")
        outs, s = [], 0
        for n in attn_bias.seqlens:
            outs.append(F.scaled_dot_product_attention(q[:, :, s:s + n], k[:, :, s:s + n], v[:, :, s:s + n],
                                                       is_causal=True, scale=scale))
            s += n
        if s != q.shape[2]:
            raise ValueError(f"seqlens sum {s} != tokens {q.shape[2]} (dtype {q.dtype}, device {q.device})")
        out = torch.cat(outs, dim=2)
    elif isinstance(attn_bias, LowerTriangularMaskWithTensorBias):
        n = q.shape[2]
        causal = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
        mask = attn_bias.bias.masked_fill(~causal, float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.to(q.dtype), scale=scale)
    else:
        raise NotImplementedError(f"unsupported attn_bias {type(attn_bias)}")
    return out.transpose(1, 2).contiguous()
