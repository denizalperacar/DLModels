""" Here is the implementation of the transformer 
architecture used in vanilla transformer and the 
Linformer.

Deniz A. ACAR
"""

from torch import (
    Tensor, bmm, dropout, cat,
    transpose, ones, zeros
    )   
from torch.nn import (
    Module, ModuleList, Linear, Dropout,
    Parameter
    )
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
        scores = scaling(scores, dim=-1)
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


class LayerNorm(Module):
    "Layer Norm Implementation"

    def __init__(self, features, eps=1e-6) -> None:
        super().__init__()
        self.a2 = Parameter(ones(features))
        self.b2 = Parameter(zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


class PositionWiseFF(Module):
    "Position-wise feed forward element implementation."

    def __init__(self, d_model, dff, dropout=0.1) -> None:
        super().__init__()
        self.w1 = Linear(d_model, dff)
        self.w2 = Linear(dff, d_model)
        self.dropout = Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(relu(self.w1(x))))


class MultiHeadAttention(Module):
    "Implementation of the MultiHead Attention"

    def __init__(self, dm, dq, dk, dv, h=2, dropout=None) -> None:
        super().__init__()
        self.h = h
        # project q, k, v
        self.linear_q = LinearProject(dq, dq//h, n=h)
        self.linear_k = LinearProject(dk, dk//h, n=h)
        self.linear_v = LinearProject(dv, dv//h, n=h)

        self.W0 = Linear(dm, dm)
        self.dropout = dropout

    def forward(self, q, k, v, mask=None):
        out = []
        qs = self.linear_q(q)
        ks = self.linear_k(k)
        vs = self.linear_v(v)
        for ele in range(self.h):
            out.append(
                scaledDotProductAttention(
                    qs[ele], ks[ele], vs[ele], mask, self.dropout
                )
            )
        output = cat(out, 2)
        return self.W0(output)

 
class MultiHeadAttentionLinformer(Module):
    "Implementation of the MultiHead Attention"

    def __init__(self, dm, dq, dk, dv, dl, h=2, dropout=None) -> None:
        super().__init__()
        self.h = h
        # project q, k, v
        self.linear_q = LinearProject(dq, dq//h, n=h)
        self.linear_k = LinearProject(dk, dk//h, n=h)
        self.linear_v = LinearProject(dv, dv//h, n=h)
        self.proj_k = LinearProject(dk, dl, n=h)
        self.proj_v = LinearProject(dv, dl, n=h)

        self.W0 = Linear(dm, dm)
        self.dropout = dropout

    def forward(self, q, k, v, mask=None):
        out = []
        qs = self.linear_q(q)
        ks = self.linear_k(k)
        es = self.proj_k(k)
        vs = self.linear_v(v)
        fs = self.proj_v(v)
        for ele in range(self.h):
            out.append(
                scaledDotProductAttention(
                    qs[ele], 
                    bmm(transpose(es[ele], 1, 2), ks[ele]), 
                    bmm(transpose(fs[ele], 1, 2), vs[ele]), 
                    mask, self.dropout
                )
            )
        output = cat(out, 2)
        return self.W0(output)


class TransformerEncoderBase(Module):
    "One layer of transformer encoder."

    def __init__(
            self, dm, dq, dk, dv, dl, h, 
            dropout, linformer=False, 
            norm=True, dff=512, dropout_r=0.1
            ):
        super().__init__()

        self.norm = LayerNorm(dm) if norm else None
        self.MHA = (
            MultiHeadAttentionLinformer(
                dm=dm, dq=dq, dk=dk, dv=dv,
                dl=dl, h=h, dropout=dropout
                ) if linformer else
            MultiHeadAttention(
                dm=dm, dq=dq, dk=dk, dv=dv,
                h=h, dropout=dropout
                ) 
            )
        self.PWFF = PositionWiseFF(
            d_model=dm, dff=dff, 
            dropout=dropout_r
            )
    
    def forward(self, x, q, k, v, q_enc, k_enc, mask=None):
        y = self.MHA(q,k,v, mask)
        if self.norm is not None:
            x = self.norm(x + y)
        else:
            x = x + y
        y = self.PWFF(x)
        if self.norm is not None:
            x = self.norm(x + y)
        else:
            x = x + y        
        return x


class TransformerEncoder(Module):
    "Implementation of the transformer Encoder."

    def __init__(
            self, N, dm, dq, dk, dv, dl, h, 
            dropout, linformer=False, 
            norm=True, dff=512, dropout_r=0.1):
        super().__init__()
        self.N = N
        self.QKV = clone_module(
            KQV(dm, dq, dk, dv), N
            )
        self.model = clone_module(
            TransformerEncoderBase(
                dm=dm, dq=dq, dk=dk, dv=dv, dl=dl, 
                h=h, dropout=dropout, 
                linformer=linformer, 
                norm=norm, dff=dff, 
                dropout_r=dropout_r
                ), N
            )
        self.output = KQV(dm, dq, dk, 0)
    
    def forward(self, x, mask=None):
        for ele in range(self.N):
            q, k, v = self.QKV[ele](x)
            x = self.model(x, q, k, v, mask)
        q, k, _ = self.output(x)
        return x, q, k


class TransformerDecoderBase(Module):
    "Implementation of one layer of the transformer decoder."

    def __init__(
            self, dm, dq, dk, dv, dl, h, 
            dropout, linformer=False, 
            norm=True, dff=512, dropout_r=0.1
            ):
        super().__init__()

        self.norm = LayerNorm(dm) if norm else None
        self.v = KQV(dm=dm, dq=0,dk=0, dv=dv)
        self.MHA_first = (
            MultiHeadAttentionLinformer(
                dm=dm, dq=dq, dk=dk, dv=dv,
                dl=dl, h=h, dropout=dropout
                ) if linformer else
            MultiHeadAttention(
                dm=dm, dq=dq, dk=dk, dv=dv,
                h=h, dropout=dropout
                ) 
            )
        self.MHA_second = (
            MultiHeadAttentionLinformer(
                dm=dm, dq=dq, dk=dk, dv=dv,
                dl=dl, h=h, dropout=dropout
                ) if linformer else
            MultiHeadAttention(
                dm=dm, dq=dq, dk=dk, dv=dv,
                h=h, dropout=dropout
                ) 
            )
        self.PWFF = PositionWiseFF(
            d_model=dm, dff=dff, 
            dropout=dropout_r
            )
    
    def forward(self, x, q, k, v, q_enc, k_enc, mask=None):
        y = self.MHA_first(q,k,v, mask)
        if self.norm is not None:
            x = self.norm(x + y)
        else:
            x = x + y
        y = self.MHA_second(q_enc,k_enc,self.v(x)[2], mask)
        if self.norm is not None:
            x = self.norm(x + y)
        else:
            x = x + y
        y = self.PWFF(x)
        if self.norm is not None:
            x = self.norm(x + y)
        else:
            x = x + y        
        return x


class TransformerDecoder(Module):
    "Implementation of the transformer decoder."

    def __init__(
            self, N, dm, dq, dk, dv, dl, h, 
            dropout, linformer=False, 
            norm=True, dff=512, dropout_r=0.1):
        super().__init__()
        self.N = N
        self.QKV = clone_module(
            KQV(dm, dq, dk, dv), N
            )
        self.model = clone_module(
            TransformerDecoderBase(
                dm=dm, dq=dq, dk=dk, dv=dv, dl=dl, 
                h=h, dropout=dropout, 
                linformer=linformer, 
                norm=norm, dff=dff, 
                dropout_r=dropout_r
                ), N
            )
    
    def forward(self, x, q_enc, k_enc, mask=None):
        for ele in range(self.N):
            q, k, v = self.QKV[ele](x)
            x = self.model(x, q, k, v, q_enc, k_enc, mask)
        return x


if __name__ == "__main__":

    device = torch.device("cuda:0")
    for i in range(1000):
        x = torch.randn(10,10000,256).to(device)
        a = KQV(256, 256,256,256).to(device)
        b = TransformerEncoder(8, 256, 256, 256, 256, 100, 8, Dropout(0.1), True, True, 512, 0.1).to(device)
        c = a(x)
        print(c[0].shape)

    