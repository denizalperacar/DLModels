import torch
import torch.nn as nn
from math import pi


from .transformer_learner import TransformerEncoder as TE 
from .transformer_learner import TransformerDecoder as TD

class Encoder(nn.Module):


    def __init__(
            self, d_model=256, k=80, 
            layers=5, inp_size=2, 
            dr=0.0, dff=512
        ):
        super(Encoder, self).__init__()
        self.enc_moderator = nn.Linear(inp_size, d_model)
        self.encoder = TE(
            layers, d_model, d_model, d_model, d_model, k, 2, 
            nn.Dropout(dr), False, True, dff, dr
            )
        
    def forward(self, x):
        x = self.enc_moderator(x)
        return self.encoder(x)

class Decoder(nn.Module):

    def __init__(
            self, d_model=256, k=80, 
            layers=5, inp_size=1, 
            out_size=4,
            dr=0.0, dff=512
        ):
        super(Decoder, self).__init__()
        self.dec_moderator = nn.Linear(inp_size, d_model)
        self.decoder = TD(
            layers, d_model, d_model, d_model, d_model, k, 2, 
            nn.Dropout(dr), False, True, dff, dr
            )
        self.linear1 = nn.Linear(d_model, out_size)
        self.act_1 = nn.LeakyReLU(0.1)
    
    def forward(self, x, k, v):
        x = self.decoder(self.dec_moderator(x), k, v)
        x = torch.atan(self.linear1(self.act_1(x))) / pi + 0.5
        return x
    

class Model(nn.Module):

    def __init__(
            self, d_model=256, k=80, 
            layers=5, enc_inp_size=2, 
            dec_inp_size=1, out_size=4,
            dr=0.0, dff=512
        ):
    
        self.enc = Encoder(
            d_model, k, layers, enc_inp_size, dr, dff
            )
        self.dec = Decoder(
            d_model, k, layers, dec_inp_size, out_size,dr, dff
            )

    def forward(self, x_enc, x_dec):

        k, v = self.enc(x_enc)
        return self.dec(x_dec, k[:,-1,:], v[:,-1,:])

    def save_net(self, name):
        torch.save(self, '{}'.format(name))
        return 'Current network is saved.\n'
