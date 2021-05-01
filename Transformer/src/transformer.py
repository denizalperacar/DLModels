""" Here is the implementation of the transformer 
architecture used in vanilla transformer and the 
Linformer.

Deniz A. ACAR
"""

from torch import Tensor, bmm
from torch.nn import Module, ModuleList, Linear
from torch.nn.functional import softmax, leaky_relu, relu
from copy import deepcopy

from math import sqrt
import torch


def clone_module(module, n):
    "Clones the module n times."
    return  ModuleList([deepcopy(module) for _ in range(n)])


def scaledDotProductAttention(
        q, k, v, mask=None, 
        dropout=None, minusinf=-1e9,
        scaling=softmax
        ):
    
        "Calculates Scaled Dot Product Attention"
        d_k = q.size(-1)
        scores = bmm(q, k.permute(0,2,1)) / sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, minusinf)
        scores = scaling(scores, dim=-2)
        # apply dropout to the output of each sublayer
        if dropout is not None:
            return dropout(bmm(scores, v))
        else:
            return bmm(scores, v)


class LinearProject(Module):
    "Linearly project the input n times."


    def __init__(
            self, in_features, out_features, 
            n=1, bias=True, activation=None) -> Tensor:
        super().__init__()
        self.__slots__ = ["in_features", "out_features", 
            "n", "bias", "activation", "projectors"]

        self.in_features = in_features
        self.out_features = out_features
        self.n = n
        self.bias = bias
        self.activation = activation

        self.projectors = clone_module(
            Linear(
                in_features=in_features, 
                out_features=out_features,
                bias=bias
                ), n
            )

    def forward(self, x):
        out = []
        for ele in self.projectors:
            out.append(ele(x))
        return out

    def extra_repr(self) -> str:
            return 'in_features={}, out_features={}, bias={}, n={}'.format(
                self.in_features, self.out_features, 
                self.bias is not None, self.n
            )


class KQV(Module):
    """Projects the input to either or all of the 
    Query key and Value spaces.
    
    dm: int model dimension
    dq: int query dimension
    dk: int key dimension
    dv: int value dimension
    """

    def __init__(self, dm:int, dq:int, dk:int, dv:int) -> list:
        super().__init__()
        self.__slots__ = ["dm", "dq", 'dk', "dv", 
            "q_linear", "k_linear", "v_linear"]

        self.dm = dm
        self.dq = dq
        self.dk = dk
        self.dv = dv
        self.q_linear = None if dq == 0 else LinearProject(dm, dq)
        self.k_linear = None if dk == 0 else LinearProject(dm, dk)
        self.v_linear = None if dv == 0 else LinearProject(dm, dv)

    def forward(self, x):
        out = [1,2,3]
        out[0] = self.q_linear(x)[0] if self.q_linear is not None else None
        out[1] = self.k_linear(x)[0] if self.k_linear is not None else None
        out[2] = self.v_linear(x)[0] if self.v_linear is not None else None
        return out

    def extra_repr(self) -> str:
            return 'dm={}, dq={}, dk={}, dv={}'.format(
                self.dm, self.dq, self.dk, self.dv)    


class MultiHeadAttention(Module):
    "Implements different varients of MultiHead Attention"

    def __init__(self, dm, dq, dk, dv, h=2, ) -> None:
        super().__init__()




if __name__ == "__main__":

    device = torch.device("cuda:0")

    x = torch.randn(1,2,50).to(device)
    a = KQV(50, 256,256,10).to(device)
    print(a)
    b = a(x)
    print(scaledDotProductAttention(*b))