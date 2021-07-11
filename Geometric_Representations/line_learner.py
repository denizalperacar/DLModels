import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from torch.nn import LSTM, Module, LeakyReLU, Sequential, Linear
import torch
import math, random
from time import time
import multiprocessing as mp
import pickle


def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float64)
    return np.array(*args, **kwargs)


def generatePoint( p1, p2, irregularity, spikeyness, numVerts ) :  
    
    if p2[0] == p1[0] and p2[1] == p1[1]:
        print("ERROR p1=p2")
        return

    '''    
    Params:
    p1, p2 - coordinates of the ends of the line
    aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
    numVerts - self-explanatory

    Returns a list of vertices, in CCW order.
    '''
    length = ( ( p1[0] - p2[0] )**2+ ( p1[1] - p2[1] )**2 )**0.5
    irregularity = clip( irregularity, 0,1 ) * length / numVerts
    spikeyness = clip( spikeyness, 0,1 ) * length

    # generate n  steps
    LengthSteps = []
    lower = (length / numVerts) - irregularity
    upper = (length / numVerts) + irregularity
    sum = 0
    for i in range(numVerts) :
        tmp = random.uniform(lower, upper)
        LengthSteps.append( tmp )
        sum = sum + tmp

    # normalize the steps 
    k = sum / (length)
    for i in range(numVerts) :
        LengthSteps[i] = LengthSteps[i] / k

    # now generate the points
    points = [(p1[0],p1[1])]
    t = 0

    #slope of the line through p1-p2
    m = ( p2[1] - p1[1]) / ( p2[0] - p1[0] )
    angle_n = math.pi /2 + math.atan(m)
    slope_ang = math.atan(m)
    if p1[0] > p2[0]: angle_n += math.pi
    if p1[0] > p2[0]: slope_ang += math.pi

    ctrX = p1[0]
    ctrY = p1[1]

    for i in range(numVerts) :

        if p2[0] == p1[0]:
             ctrY += LengthSteps[i]

        else:             
             ctrX += LengthSteps[i] * math.cos(slope_ang)
             ctrY += LengthSteps[i] * math.sin(slope_ang)
            
        r_i = clip( random.gauss((length / numVerts), spikeyness), 0, 2*(length / numVerts) )-(length / numVerts)
        x = +ctrX + r_i * math.cos(angle_n)
        y = +ctrY + r_i * math.sin(angle_n)
        points.append([x,y])
        
    points.append([p2[0],p2[1]])


    return np.array(points)


def clip(x, min, max) :
    if( min > max ) :  return x    
    elif( x < min ) :  return min
    elif( x > max ) :  return max
    else :             return x


def calculate_normal(vect, normalized=True):

    r2 = (vect[2] - vect[1]) / 2.
    r1 = (vect[1] - vect[0]) / 2.
    dx, dy = (r2 + r1)
    norm = (dx**2. + dy**2.)**0.5
    if normalized:
        return np.array([-dy/norm, dx/norm])
    else:
        return np.array([-dy, dx])


class Point:

    def __init__(self, p):
        self.p = np.array(p)


class Curve:
    
    def __init__(self, points, order=3, rot=1, numpts=1000):
        """rot indicates the normal direction 
        in the frenet frame on the curve in 2d it 
        is the direction of the k vector
        """ 

        points = np.array(points)[::rot]
        self.length = (((points[-1] - points[0]) ** 2.).sum())**0.5
        self.npts = numpts
        self.x = points[:, 0]
        self.y = points[:, 1]
        self.order = order
        self.rot = rot
        self.spline, _ = interpolate.splprep(
            [self.x,self.y],
            k=3,
            )
        ts = np.linspace(0,1,num=numpts+1, endpoint=True)
        self.normal = self.get_normal(ts)
        self.curve_length, self.green = self.get_curve_data(numpts)

        
    def interpret(self, t):
        t = np.array(t)
        assert (t.max()<=1. and t.min()>= 0.), "t out of range"
        return np.array(interpolate.splev(t, self.spline)).T

    def plot(self, t, ax):
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)
        out = self.interpret(t)
        ax.plot(out[:,0], out[:,1], 'b', c=color)
    
    def get_normal(self, t, eps=1e-8, normalize=True):
        n = []
        t = np.array(t)
        pts = []
        for ts in t:
            low = ts - eps if (ts - eps) > 0. else ts
            upp = ts + eps if (ts + eps) < 1. else ts 
            n.append(
                calculate_normal(
                    self.interpret([low, ts, upp]), 
                    normalize
                    )
                )
            pts.append(self.interpret([low, ts, upp]).T)
        return np.array(n)

    def plot_normal(self, t, ax):
        # norm = self.length if self.length > 1 else 1./ self.length
        n = self.get_normal(t)
        v = self.interpret(t)
        pts = v + n * 0.1
        ax.scatter(v[:,0], v[:,1], s=1)
        ax.scatter(pts[:,0], pts[:, 1], s=1)

    def get_curve_data(self, n=1000):
        t = np.linspace(0,1,n+1)
        p = self.interpret(t)
        ns = self.get_normal(t, normalize=False, eps=1/n)
        length = ((((p[1:,:] - p[:-1,:])**2.).sum(axis=1))**0.5).sum()
        green = -(p * ns).sum() / 2.
        return length, green


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


def return_data(num=1000):
    data = {}
    data['params'] = [
        np.random.randint(0,10000, 2) * np.random.randn(2) / 1000.,
        np.random.randint(0,10000, 2) * np.random.randn(2) / 1000.,
        abs(0.4), # * np.random.randn()),
        abs(0.4), # * np.random.randn()),
        np.random.randint(3,50)
        ]
    kk = data['params']
    data['vetices']  = generatePoint(
        kk[0], kk[1], irregularity=kk[2], 
        spikeyness=kk[3], numVerts=kk[4]
        )
    curve = Curve(
        np.array(data['vetices']).astype(np.float64), 3
        )
    data['curve'] = curve.__dict__
    data['query'] = {}
    data['query']['points'] = np.random.uniform(0,1, num+1)
    data['query']['out'] = np.concatenate([
        curve.interpret(data['query']['points']),
        curve.get_normal(data['query']['points']),
    ], axis=-1)
    return data


def get_data(n=1, num=1000):

    l = list(range(n))

    pool = mp.Pool(3)
    l[:] = pool.map(return_data, (num for _ in range(n)))

    return l


t = time()
ls = []
start = 0
for kk in range(1000):
    ls.append(get_data(100, np.random.randint(500, 1000)))
    if kk % 10 == 0 and kk != 0:
        with open(f'./dataset/data_{kk//10+start}.pickle', 'wb') as fid:
            pickle.dump(ls, fid)
        ls = []
        print(time()-t)
        t = time()

"""
ax = plt.subplot()
kk = [
    np.array([1,1]),
    np.array([2,3]),
    abs(0.6), # * np.random.randn()),
    abs(0.6), # * np.random.randn()),
    20
]

# verts = generatePoint(kk[0], kk[1], irregularity=kk[2], spikeyness=kk[3], numVerts=kk[4])
verts = np.array([np.linspace(5, 1,10), np.linspace(1,3,10)]).T
c = Curve(np.array(verts).astype(np.float64), 3, numpts=10000)


t = np.linspace(0,1,num=2001, endpoint=True)
t1 = np.linspace(0,1,1001)
c.plot(t, ax)
c.plot_normal(t1, ax)
print(c.get_curve_data())
print(((verts[-1] - verts[0])**2.).sum()**0.5)
plt.axis('equal')
plt.show()
"""