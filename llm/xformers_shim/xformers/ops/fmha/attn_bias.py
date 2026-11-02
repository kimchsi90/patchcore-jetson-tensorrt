from typing import List, Optional


class AttentionBias:
    pass


class BlockDiagonalCausalMask(AttentionBias):
    def __init__(self, seqlens: List[int], window: Optional[int] = None):
        self.seqlens = list(seqlens)
        self.window = window

    @classmethod
    def from_seqlens(cls, seqlens):
        return cls(seqlens)

    def make_local_attention(self, window_size: int):
        return BlockDiagonalCausalMask(self.seqlens, window=window_size)


class LowerTriangularMaskWithTensorBias(AttentionBias):
    def __init__(self, bias):
        self.bias = bias
