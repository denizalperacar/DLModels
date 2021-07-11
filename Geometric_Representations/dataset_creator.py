from glob import glob
import pickle
import numpy as np
from time import time

ADDRESS = "dataset/"
t = time()

def get_data(address):
    data_address = {i:j for i, j in enumerate(glob(f"{address}*.pickle"))}
    return data_address


dataset = get_data(ADDRESS)

def read_data(ids):
    with open(dataset[ids], 'rb') as fid:
        data = pickle.load(fid)
    return data

data_pts = []
data_query_inp = []
data_query_out = []

for ind in dataset.keys():
    for dts in read_data(ind):
        for grp in dts:
            s = grp['curve']['x'].shape
            a = np.zeros((s[0], 2))
            a[:,0] = grp['curve']['x'].copy()
            a[:,1] = grp['curve']['y'].copy()
            data_pts.append(a.copy())
            data_query_inp.append(grp['query']['points'].copy())
            data_query_out.append(grp['query']['out'].copy())

with open("data.pickle", 'wb') as fid:
    pickle.dump([data_pts, data_query_inp, data_query_out], fid)


inds = np.zeros((1000)).astype(int)
for i in data_query_inp:
    inds[i.shape[0]] += 1

inds_dict = {}
for i in range(1000):
    if inds[i] != 0:
        inds_dict[i] = []

for i in range(len(data_query_inp)):
    inds_dict[data_query_inp[i].shape[0]].append(i)

with open('data_ids.pickle', 'wb') as fid:
    pickle.dump(inds_dict, fid)

print(time()-t)