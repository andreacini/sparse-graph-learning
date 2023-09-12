import torch
from torch import nn

from lib.utils.utils import soft_clip

from einops import repeat


class AdjEmb(nn.Module):
    """
    """

    def __init__(self,
                 num_nodes,
                 learnable=True,
                 clamp_at=5.):
        super(AdjEmb, self).__init__()
        self.clamp_value = clamp_at
        self.logits = nn.Parameter(torch.rand(num_nodes, num_nodes) - 0.5, requires_grad=learnable)

    def forward(self, x, *args, **kwargs):
        """"""
        b, *_ = x.size()
        scores = soft_clip(self.logits, self.clamp_value)
        return repeat(scores, '... -> b ...', b=b)
