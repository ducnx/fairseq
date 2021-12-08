import math
from typing import List

from fairseq.modules import Fp32GroupNorm, Fp32LayerNorm, TransposeLast
from torch import nn


def norm_block(is_layer_norm, dim, affine=True):
    if is_layer_norm:
        mod = nn.Sequential(
            TransposeLast(),
            Fp32LayerNorm(dim, elementwise_affine=affine),
            TransposeLast(),
        )
    else:
        mod = Fp32GroupNorm(1, dim, affine=affine)

    return mod


class MLPFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List,
        output_size: int
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.net = self.build_model()

    def build_model(self):
        if len(self.hidden_sizes) == 0:
            net = nn.Sequential(
                nn.Linear(self.input_size, self.output_size)
            )
        else:
            modules = []
            inp_size = self.input_size
            out_size = self.hidden_sizes[0]
            for i in range(len(self.hidden_sizes)):
                modules.append(
                    nn.Linear(inp_size, out_size, bias=True)
                )
                if i < len(self.hidden_sizes):
                    modules.append(nn.ReLU(True))
            modules.append(
                nn.Linear(self.hidden_sizes[-1], self.output_size, bias=True)
            )
            net = nn.Sequential(*modules)
        return net

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): BxTxC1

        Returns:
            torch.Tensor: BxTxC2
        """
        out = self.net(x)
        out = out.transpose(1, 2)
        return out
