""" Here is the implementation of the transformer 
architecture used in vanilla transformer and the 
Linformer.

Deniz A. ACAR
"""

from torch import (
    Tensor, bmm, cat, randn, 
    transpose, ones, zeros,
    device, from_numpy, save,
    load, sin, atan, unsqueeze,
    acos, cos, tanh, cross
    )   
from torch.nn import (
    Module, ModuleList, Linear, Dropout,
    Parameter, Conv2d, LeakyReLU, Sequential,
    ConvTranspose2d, BCELoss, Sigmoid
    )
from torch.optim import Adam, RMSprop
from torch.nn.functional import sigmoid, softmax, leaky_relu
from copy import deepcopy
from math import cosh, sqrt, pi
from numpy import triu, ones as npones

# create a polygon
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
import pickle


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
        # scores = leaky_relu(scores, 0.5)
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
        # self.q = KQV(dm=dm, dq=dq,dk=0, dv=0)
        """ self.MHA_first = (
            MultiHeadAttentionLinformer(
                dm=dm, dq=dq, dk=dk, dv=dv,
                dl=dl, h=h, dropout=dropout
                ) if linformer else
            MultiHeadAttention(
                dm=dm, dq=dq, dk=dk, dv=dv,
                h=h, dropout=dropout
                ) 
            ) """
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
    
    def forward(self, x, q, k_enc, v_enc, mask=None):
        """         y = self.MHA_first(q,k,v, mask)
        if self.norm is not None:
            x = self.norm(x + y)
        else:
            x = x + y """
        y = self.MHA_second(q, k_enc, v_enc, mask)
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
            KQV(dm, dq, 0, 0), N
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
            q, _, _ = self.QKV[ele](x)
            x = self.model[ele](x, q, q_enc, k_enc, mask)
        return x


"""

class Model(Module):

    def __init__(self, d_model=256, k=80, inp_size=1, dr=0.0, dff=512) -> None:
        super().__init__()


        self.enc_moderator = Conv2d(1, d_model, kernel_size=2, stride=1, padding=1)
        self.enc_moderator2 = Conv2d(d_model, d_model, kernel_size=3, stride=1)
        self.dec_moderator = Linear(2, d_model)
        self.encoder = TransformerEncoder(
            5, d_model, d_model, d_model, d_model, k, 2, 
            Dropout(dr), False, True, dff, dr
            )
        self.decoder = TransformerDecoder(
            5, d_model, d_model, d_model, d_model, k, 2, 
            Dropout(dr), False, True, dff, dr
            )
        self.linear1 = Linear(d_model, 1)
        self.act_1 = LeakyReLU(0.1)

    def forward(self, x_enc, x_dec):
        xe = unsqueeze(x_enc, 1)
        xe = self.enc_moderator(xe)
        xe = self.enc_moderator2(xe)
        xe = xe.squeeze(-1).permute(0,2,1)
        k, v = self.encoder(xe)
        output = self.decoder(self.dec_moderator(x_dec), k, v)
        
        return output

    def save_net(self, name):
        save(self, '{}'.format(name))
        return 'Current network is saved.\n'


if __name__ == "__main__":
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    address = "/home/deniz/Desktop/DLModels/geometry_learner/data"
    def create_data(ns, hh=5):
        output = []
        for k in range(hh):
            output.append([])
            r = np.random.randint(1, 100)
            n = np.random.randint(20, 40)
            out_p = np.random.randint(2000, 3000)
            enc = []
            dec = []
            out = []
            rs = []
            for i in range(ns):
                e, d, o = get_input(n, out_p, r)
                enc.append(e)
                dec.append(d)
                out.append(o)
                rs.append(r)
            
            output[k].append([np.array(enc), np.array(dec), np.array(out), np.array(r)])

        with open(f"{address}/data.pickle", 'wb') as fid:
            pickle.dump(output, fid)
        # print(time()-t)
    
    def getdata(data, ind):
        d = data[ind]
        # r = d[0][3]
        x_enc = from_numpy(d[0][0]).float()
        x_dec = from_numpy(d[0][1]).float()
        y = from_numpy(d[0][2]).float().unsqueeze(2)
        return x_enc, x_dec, y

    DEVICE = device('cuda:0') 
    NUMELE = 1

    net = Model().to(DEVICE)
    net = load('net')

    create_data(NUMELE, 1)
    with open(f"{address}/data.pickle", 'rb') as fid:
        data = pickle.load(fid)  
    x_enc, x_dec, y= getdata(data, 0)
    x_e = x_enc.to(DEVICE)
    x_d = x_dec.to(DEVICE)
    y = y.to(DEVICE)
    # print(x_dec.max())
    o = net(x_e, x_d) 
    y_p = o >=0.5
    plt.plot(x_enc[0,:,0], x_enc[0,:,1])
    plt.scatter(x_dec[0,:,0], x_dec[0,:,1], s=1, c=np.array(y_p.tolist()) / 10 + 1, cmap="Accent")
    plt.show()   


if __name__ == "__main__":
    
    t = time()
    address = "/home/deniz/Desktop/DLModels/geometry_learner/data"
    def create_data(ns):
        output = []
        for k in range(5):
            output.append([])
            r = np.random.randint(1, 100)
            n = np.random.randint(5, 40)
            out_p = np.random.randint(100, 1000)
            enc = []
            dec = []
            out = []
            rs = []
            for i in range(ns):
                e, d, o = get_input(n, out_p, r)
                enc.append(e)
                dec.append(d)
                out.append(o)
                rs.append(r)
            
            output[k].append([np.array(enc), np.array(dec), np.array(out), np.array(r)])

        with open(f"{address}/data.pickle", 'wb') as fid:
            pickle.dump(output, fid)
        # print(time()-t)
    
    def getdata(data, ind):
        d = data[ind]
        # r = d[0][3]
        x_enc = from_numpy(d[0][0]).float()
        x_dec = from_numpy(d[0][1]).float()
        y = from_numpy(d[0][2]).float().unsqueeze(2)
        return x_enc, x_dec, y
    
    print_string = 'epoch {:08d} |mean_loss {:16.08f} |max_loss {:16.08f} | min_loss{:16.08f} |time {:16.08f}s'
    DEVICE = device('cuda:0') 
    NUMELE = 50

    net = Model().to(DEVICE)
    net = load('net')
    network_opimizer = Adam(net.parameters(), lr=1.e-5)
    loss_fn = BCELoss()
    # prepare dataset
    # convertToArray(address)
    # prepare_dataset(address, NUMELE, 1)
    create_data(NUMELE)
    e_loss = 0.6
    with open(f"{address}/data.pickle", 'rb') as fid:
        data = pickle.load(fid)
        indices = np.array(range(len(data)))
    for epoch in range(1, 1000000):

        if epoch != 1 and epoch%1 == 0:
            create_data(NUMELE)
            with open(f"{address}/data.pickle", 'rb') as fid:
                data = pickle.load(fid)
            indices = np.array(range(len(data)))
        t = time()
        element_loss = []
        np.random.shuffle(indices) # shuffle the inputs
        for i in indices:   
            x_enc, x_dec, y= getdata(data, i)
            x_enc = x_enc.to(DEVICE)
            x_dec = x_dec.to(DEVICE)
            y = y.to(DEVICE)
            # print(x_dec.max())
            o = net(x_enc, x_dec)
            network_opimizer.zero_grad()
            
            current_loss = loss_fn(o, y)
            current_loss.backward()
            network_opimizer.step()

            element_loss.append(current_loss.tolist())

        element_loss = np.array(element_loss)

        if element_loss.mean() < e_loss:
            e_loss = element_loss.mean()
            net.save_net('net')
        if epoch % 10000 == 0:
            net.save_net('net_{}'.format(epoch))
        if epoch % 1 == 0:
            print(print_string.format(epoch, 
                                    element_loss.mean(), 
                                    element_loss.max(), 
                                    element_loss.min(),
                                    time()-t))
"""
