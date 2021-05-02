from typing import Sequence
import torch
from torch.nn import (
    Parameter, Linear, Module, ModuleList, Dropout
    )
from torch import Tensor, randn, bmm, cat, device, stack, ones, zeros
from torch.nn.functional import softmax, leaky_relu, relu
from torch.cuda import memory_allocated
from math import sqrt
from time import time
from copy import deepcopy
from collections import OrderedDict

import numpy as np 

from torch.nn.modules.container import Sequential


def clone(module, n):
    return ModuleList([deepcopy(module) for _ in range(n)])


class Z2QKV(Module):
    "Implements the transformation from z to QKV"
    
    def __init__(self, ENCDIM, QDIM, KDIM, VDIM):
        super().__init__()
        __slots__ = [
            "ENCDIM", "QDIM", "KDIM", "VDIM", 
            "Q_Linear", "K_Linear", "V_Linear"]

        self.ENCDIM = ENCDIM
        self.QDIM = QDIM
        self.KDIM = KDIM
        self.VDIM = VDIM
        
        self.Q_Linear = Linear(ENCDIM, QDIM, bias=True)
        self.K_Linear = Linear(ENCDIM, KDIM, bias=True)
        self.V_Linear = Linear(ENCDIM, VDIM, bias=True)
        
    def forward(self, x):
        return self.Q_Linear(x), self.K_Linear(x), self.V_Linear(x)


class MultiHeadedAtention(Module):
    "Implements a Multi-Head Ateention layer."

    def __init__(self, dk, dv, h=8, d_model=256, dropout=None) -> Tensor:
        super().__init__()
        self.dk = dk//h
        self.dv = dv//h
        self.h = h
        self.linear_q = clone(Linear(dk, self.dk),self.h)
        self.linear_k = clone(Linear(dk, self.dk),self.h)
        self.linear_v = clone(Linear(dv, self.dv),self.h)

        self.w_o = Linear(int(h*self.dv), dv)
        self.dropout = dropout

    def attention(
            self, 
            Query:Tensor, Key:Tensor, Value:Tensor, 
            mask=None, dropout=None, minusinf=-1e9
            ) -> Tensor:
        "Calculates Scaled Dot Product Attention"
        d_k = Query.size(-1)
        scores = bmm(Query, Key.permute(0,2,1)) / sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, minusinf)
        scores = softmax(scores, dim=-1)
        # apply dropout to the output of each sublayer
        if dropout is not None:
            return dropout(bmm(scores, Value))
        else:
            return bmm(scores, Value)

    def forward(self, Query:Tensor, Key:Tensor, Value:Tensor, mask=None):

        out = []
        for head in range(0, self.h):
            out.append(
                self.attention(self.linear_q[head](Query),
                self.linear_k[head](Key), 
                self.linear_v[head](Value), 
                mask, self.dropout)
                )
        output = cat(out, 2)
        return self.w_o(output)


class PositionWiseFF(Module):
    "Position-wise feed forward element implementation."

    def __init__(self, d_model, dff, dropout=0.1) -> None:
        super().__init__()
        self.w1 = Linear(d_model, dff)
        self.w2 = Linear(dff, d_model)
        self.dropout = Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(relu(self.w1(x))))


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
        return self.a_2 * (x - mean) / (std + self.eps) + self.b2


class TransformerEncoderBase(Module):
    "Implementation of Transformers Encoder."

    def __init__(
            self, q_dim, model_dim, h=4, 
            dropout=None, mask=None, norm=LayerNorm,
            dff=512, dropout_r=0.1
            ) -> None:
        super().__init__()

        self.__slots__ = [
            "enc_dim", "q_dim", "model_dim", "h", 
            "dropout", "mask", "norm", "dff", "dropout_r"
            ]
        self.q_dim = q_dim
        self.model_dim = model_dim
        self.h = h
        self.dropout = dropout
        self.mask = mask
        self.norm = norm
        self.dff = dff
        self.dropout_r = dropout_r

        self.QKV = Z2QKV(model_dim, q_dim, q_dim, model_dim)
        self.MHA = MultiHeadedAtention(
            q_dim, model_dim, h=h, 
            d_model=model_dim, dropout=dropout
            )
        self.PWFF = PositionWiseFF(model_dim, dff, dropout_r)

    def forward(self, x, k, q, v, mask=None):
        # multi head attention sublayer
        y = self.MHA(q,k,v,mask)
        x = x + y
        y = self.PWFF(x)
        x = x + y
        q, k, v = self.QKV(x)
        return x, q, k, v


class TransformerEncoder(Module):
    "Encoder layer of transformer module."

    def __init__(
            self, N, q_dim, model_dim, h=4, 
            dropout=None, mask=None, norm=LayerNorm,
            dff=512, dropout_r=0.1
        ) -> None:
        super().__init__()


        self.__slots__ = [
            "N", "enc_dim", "q_dim", "model_dim", "h", 
            "dropout", "mask", "norm", "dff", "dropout_r"
            ]
        self.N = N
        self.params_dict = OrderedDict()
        self.params_dict["q_dim"] = q_dim
        self.params_dict["model_dim"] = model_dim 
        self.params_dict["h"] = h 
        self.params_dict["dropout"] = dropout 
        self.params_dict["mask"] = mask 
        self.params_dict["norm"] = norm
        self.params_dict["dff"] = dff 
        self.params_dict["dropout_r"] = dropout_r


        self.qkv = Z2QKV(model_dim, q_dim, q_dim, model_dim)
        self.layers = clone(
            TransformerEncoderBase(*self.params_dict.values()), N
            )
        
    def forward(self, x, mask=None):

        q, k, v = self.qkv(x)

        for layer in self.layers:
            x, q, k, v = layer(x, q, k, v, mask)

        return q, k


class TransformerDecoderBase(Module):
    "Implementation of the Transformer Decoder."

    def __init__(
            self, q_dim, model_dim, h=4, 
            dropout=None, mask=None, norm=LayerNorm,
            dff=512, dropout_r=0.1
            ) -> None:
        super().__init__()

        self.__slots__ = [
            "enc_dim", "q_dim", "model_dim", "h", 
            "dropout", "mask", "norm", "dff", "dropout_r"
            ]
        self.q_dim = q_dim
        self.model_dim = model_dim
        self.h = h
        self.dropout = dropout
        self.mask = mask
        self.norm = norm
        self.dff = dff
        self.dropout_r = dropout_r

        self.QKV = Z2QKV(model_dim, q_dim, q_dim, model_dim)
        self.MHA_1 = MultiHeadedAtention(
            q_dim, model_dim, h=h, 
            d_model=model_dim, dropout=dropout
            )
        self.get_v = Linear(model_dim, model_dim, bias=True)
        self.MHA_2 = MultiHeadedAtention(
            q_dim, model_dim, h=h, 
            d_model=model_dim, dropout=dropout
            )
        self.PWFF = PositionWiseFF(model_dim, dff, dropout_r)

    def forward(self, x, k, q, v, q_enc, k_enc, mask=None):
        # multi head attention sublayer
        y = self.MHA_1(q,k,v,mask)
        x = x + y
        v = self.get_v(x)
        y = self.MHA_1(q_enc,k_enc,v,mask)
        x = x + y
        y = self.PWFF(x)
        x = x + y
        q, k, v = self.QKV(x)
        return x, q, k, v


class TransformerDecoder(Module):
    "Implementation of the Transformer Decoder"

    def __init__(
            self, N, q_dim, model_dim, h=4, 
            dropout=None, mask=None, norm=LayerNorm,
            dff=512, dropout_r=0.1
        ) -> None:
        super().__init__()

        self.__slots__ = [
            "N", "enc_dim", "q_dim", "model_dim", "h", 
            "dropout", "mask", "norm", "dff", "dropout_r"
            ]
        
        self.N = N
        self.params_dict = OrderedDict()
        self.params_dict["q_dim"] = q_dim
        self.params_dict["model_dim"] = model_dim 
        self.params_dict["h"] = h 
        self.params_dict["dropout"] = dropout 
        self.params_dict["mask"] = mask 
        self.params_dict["norm"] = norm
        self.params_dict["dff"] = dff 
        self.params_dict["dropout_r"] = dropout_r


        self.qkv = Z2QKV(model_dim, q_dim, q_dim, model_dim)
        self.layers = clone(
            TransformerDecoderBase(*self.params_dict.values()), N
            )
        
    def forward(self, x, q_enc, k_enc, mask=None):

        q, k, v = self.qkv(x)

        for layer in self.layers:
            x, q, k, v = layer(x, q, k, v, q_enc, k_enc, mask)    

        return v

    
if __name__ == "__main__":

    KDIM = 256
    QDIM = 256
    VDIM = 256
    batch = 1
    n = 100
    h = 4


    DEVICE = device("cuda:0")


    tt = []
    for i in range(1000):
        t = time()
        x = randn(batch, n, VDIM).to(DEVICE)
        
        enc = TransformerEncoder(1, QDIM, VDIM, h).to(DEVICE)
        dec = TransformerDecoder(1, QDIM, VDIM, h).to(DEVICE)

        q_enc, k_enc = enc(x)
        out = dec(x, q_enc, k_enc, None)

        # print(out.shape)
        # print(memory_allocated(DEVICE))
        tt.append(time()-t)
    print(tt)
    print(np.array(tt)[1:].mean())

