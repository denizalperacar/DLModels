import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from torch.nn import LSTM, Module, LeakyReLU, Sequential, Linear
import torch


class Point:

    def __init__(self, p):
        self.p = np.array(p)


class Curve:
    
    def __init__(self, points, order=3):
        
        points = np.array(points)
        self.x = points[:, 0]
        self.y = points[:, 1]
        self.order = 3
        self.spline, _ = interpolate.splprep(
            [self.x,self.y],
            k=3,
            )
        
    def interpret(self, t):
        assert(t.max()<=1 and t.min()>= 0), "t out of renge"
        return interpolate.splev(t, self.spline)

    def plot(self, t, ax):
        out = self.interpret(t)
        plt.plot(out[0], out[1], 'b')
    

class Encoder(Module):

    def __init__(
            self, 
            input_dim=2, 
            hidden_size=256, 
            num_layers=1, 
            bias=True, 
            batch_first=True):
        super().__init__()

        self.lstm = LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first)

    def forward(self, x, h0, c0):
        output, _ = self.lstm(x, (h0, c0))
        return output


class ARCLENDecoder(Module):

    def __init__(
            self,
            input_size=3,
            output_size=4, 
            hidden_size=256):
        super().__init__()

        self.model = Sequential()
        self.model.add_module(
            "Linear_1", Linear(hidden_size + input_size, hidden_size)
        )
        self.model.add_module(
            "act_1", LeakyReLU(0.1)
        )
        self.model.add_module(
            "Linear_2", Linear(hidden_size, hidden_size)
        )
        self.model.add_module(
            "act_2", LeakyReLU(0.1)
        )        
        self.model.add_module(
            "Linear_3", Linear(hidden_size, output_size)
        )
        self.model.add_module(
            "act_3", LeakyReLU(0.1)
        )          
    
    def forward(self, x, h):
        y = torch.cat((x, h.expand(-1,x.shape[0])))
        return self.model(y)





if __name__ == "__main__":

    ctr =np.array( [(3 , 1), (2.5, 4), (0, 1), (-2.5, 4),
                (-3, 0), (-2.5, -4), (0, -1), (2.5, -4), (3, -1)])    
    c = Curve(ctr, 3)
    ax = plt.subplot()
    t = np.linspace(0,1,num=50,endpoint=True)  
    c.plot(t, ax)
    plt.show()
