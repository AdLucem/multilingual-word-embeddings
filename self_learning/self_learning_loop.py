
from sample_dict import random_dict

import json
import numpy as np
from sklearn.neighbors import NearestNeighbors

# number of iterations
num_iter = 100

# vo: vocab, mat: matrix
# d: dictionary in the format described in the paper
d, vo_x, vo_z, mat_x, mat_z = random_dict()

def test_print_map(d):
    for index, ls in enumerate(d):
        print(vo_x[index])
        mapping = ls.index(1)
        print(vo_z[mapping])
        print("-------------------")


embedding_size = 100

# nvx= number of items in vocab of language x
nvx = len(vo_x)
nvz = len(vo_z)

# convert the embedding matrices to numpy arrays
x = np.array(mat_x)
z = np.array(mat_z)

# randomly initialize weights
wx = np.random.rand(1, embedding_size)
wz = np.random.rand(1, embedding_size)

for i in range(num_iter):

    # get optimal wx and wz
    m = (x.T.dot(d)).dot(z)
    wx2, s, wz2_t = np.linalg.svd(m)
    wx = wx.dot(wx2)
    wz = wz.dot(wz2_t.T)

    # reweigh the word vectors
    for i in range(x.shape[0]):
        x[i] = x[i].dot(wx.T)
    for j in range(z.shape[0]):
        z[j] = z[j].dot(wz.T)

    # get nearest neighbour of each element in matrix x
    nbrs = NearestNeighbors(n_neighbors=1).fit(z)
    nns = nbrs.kneighbors(x, 1, return_distance=False)

    # recalculate dictionary
    d = [[0 for i in range(nvz)] for j in range(nvx)]

    for i in range(nvx):
        nn = nns[i]
        d[i][nn[0]] = 1

# save mappings as a list of tuples
lmap = []
for index, ls in enumerate(d):
        a = vo_x[index].encode("utf-8")
        mapping = ls.index(1)
        b = vo_z[mapping].encode("utf-8")
        lmap.append((a, b))

with open("mappings.json", "w+") as f:
    data = json.dumps(lmap)
    f.write(data)


