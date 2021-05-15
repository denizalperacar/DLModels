""" Here is the implementation of the transformer 
architecture used in vanilla transformer and the 
Linformer.

Deniz A. ACAR
"""

from torch import (
    Tensor, bmm, cat, randn,
    transpose, ones, zeros,
    device, from_numpy
    )   
from torch.nn import (
    Module, ModuleList, Linear, Dropout,
    Parameter, Conv2d, LeakyReLU, Sequential,
    ConvTranspose2d
    )
from torch.nn.functional import softmax
from copy import deepcopy
from math import sqrt
from numpy import triu, ones as npones


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


def subsequent_mask(size):
    attn_shape = (1, size, size)
    np_mask = triu(npones(attn_shape), k=1).astype('uint8')
    return from_numpy(np_mask) == 0


class LinearProject(Module):
    "Linearly project the input n times."


    def __init__(
            self, in_features, out_features, 
            n=1, bias=True) -> Tensor:
        super().__init__()
        self.__slots__ = ["in_features", "out_features", 
            "n", "bias", "activation", "projectors"]

        self.in_features = in_features
        self.out_features = out_features
        self.n = n
        self.bias = bias

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

    def __init__(self, d_model, dff, dropout=0.1, activation=LeakyReLU(0.1)) -> None:
        super().__init__()
        self.activation = activation
        self.w1 = Linear(d_model, dff)
        self.w2 = Linear(dff, d_model)
        self.dropout = Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(self.activation(self.w1(x))))


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
    
    def forward(self, x, q, k, v, mask=None):
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
        self.output = KQV(dm, 0, dk, dv)
    
    def forward(self, x, mask=None):
        for ele in range(self.N):
            q, k, v = self.QKV[ele](x)
            x = self.model[ele](x, q, k, v, mask)
        _, k, v = self.output(x)
        return k, v


class TransformerDecoderBase(Module):
    "Implementation of one layer of the transformer decoder."

    def __init__(
            self, dm, dq, dk, dv, dl, h, 
            dropout, linformer=False, 
            norm=True, dff=512, dropout_r=0.1
            ):
        super().__init__()

        self.norm = LayerNorm(dm) if norm else None
        self.q = KQV(dm=dm, dq=dq,dk=0, dv=0)
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
    
    def forward(self, x, q, k, v, k_enc, v_enc, mask=None):
        y = self.MHA_first(q,k,v, mask)
        if self.norm is not None:
            x = self.norm(x + y)
        else:
            x = x + y
        y = self.MHA_second(self.q(x)[0], k_enc, v_enc, mask)
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
            x = self.model[ele](x, q, k, v, q_enc, k_enc, mask)
        return x


class Embedd(Module):
    "Embedd the inputs"

    def __init__(self, in_channels=6, out_dim=256, activation=LeakyReLU(0.2)) -> None:
        super().__init__()

        self.out_dim = out_dim
        self.block = Sequential()
        self.block.add_module(
            "conv_1", Conv2d(in_channels, int(out_dim//2), 4, 2)
        )
        self.block.add_module("a1", activation)
        self.block.add_module(
            "conv_2", Conv2d(int(out_dim//2), int(out_dim//2), 4, 3)
        )
        self.block.add_module("a2", activation)
        self.block.add_module(
            "conv_3", Conv2d(int(out_dim//2), out_dim, 4, 1)
        )
        self.block.add_module("a3", activation)

    def forward(self, x):
        out = []
        x_s = x.shape
        for ele in x:
            out.append(self.block(ele).reshape(1, x_s[1], self.out_dim))
        y = cat(out, 0)
        return y


class Translate(Module):
    "Translate the output."

    def __init__(self, in_channels=256, out_dim=6, activation=LeakyReLU(0.2)):
        super().__init__()

        self.out_dim = out_dim
        self.block = Sequential()
        self.block.add_module(
            "tconv_1", Conv2d(in_channels, int(out_dim//2), 4, 2)
        )
        self.block.add_module("a1", activation)
        self.block.add_module(
            "tconv_2", Conv2d(int(out_dim//2), int(out_dim//2), 4, 3)
        )
        self.block.add_module("a2", activation)
        self.block.add_module(
            "tconv_3", Conv2d(int(out_dim//2), out_dim, 4, 1)
        )
        self.block.add_module("a3", activation) 

if __name__ == "__main__":

    dev = device("cuda:0")
    k = 100
    a = Embedd(6, 256).to(dev)
    d = TransformerEncoder(4, 256, 256, 256, 256, k, 8, Dropout(0.1), False, True, 256, 0.1).to(dev)
    e = TransformerDecoder(4, 256, 256, 256, 256, k, 8, Dropout(0.1), False, True, 256, 0.1).to(dev)

    for i in range(1000):

        n = 256
        l = 256

        m = subsequent_mask(n).to(dev)        


        out = []
        x1 = randn(1, n, 6, 32, 32).to(dev)
        x2 = randn(1, n, 6, 32, 32).to(dev)
        
        y1 = a(x1)
        y2 = a(x2)
        k, v = d(y1)
        o2 = e(y2, k, v, m)
        print(o2.shape)
    
    