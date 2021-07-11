import torch
import pickle
import torch.nn as nn
import numpy as np
from math import pi
from .model import Model

with open('data.pickle', 'rb') as fid:
    pts, inps, outs = pickle.load(fid)

with open('data_ids.pickle', 'rb') as fid:
    ids = pickle.load(fid)

