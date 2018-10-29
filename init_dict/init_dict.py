from numpy import *
import numpy as np
import json
from sklearn.neighbors import NearestNeighbors
#from sklearn.preprocessing import normalize

datafile = "/home/atreyee/Academics/multilingual-word-embeddings/data.json"

def main():

    n = 2000

    with open(datafile) as f:
        d = json.load(f)
    # taking the same no of words in each language
    X = array(d["matrix_english"][:n])
    Y = array(d["matrix_hindi"][:n])
    vocab_x = X.shape[0]
    vocab_y = Y.shape[0]
    #print (X)
    #norm = abs(X).sum(axis=1)
    #normed_X = X/(norm[:, newaxis])

    norm = X.sum(axis=1)
    normed_X = X/norm[:, newaxis]
    norm =Y.sum(axis=1)
    normed_Y = Y/norm[:, newaxis]

    meanX = mean(X.T, axis=1)
    new_X = normed_X - meanX
    meanY = mean(Y.T, axis=1)
    new_Y = normed_Y - meanY

    #print (new_X)
    new_X /= new_X.sum(axis=1)[:, newaxis]
    new_Y /= new_Y.sum(axis=1)[:, newaxis]
    #print (new_X)
    #print (new_Y)

    Mx = dot(X, X.T)
    My = dot(Y, Y.T)

    # sort the matrices
    for i in range(Mx.shape[0]):
        Mx[i] = np.sort(Mx[i])
    for j in range(My.shape[1]):
        My[j] = np.sort(My[j])

    # get nearest neighbour of each element in matrix Mx
    nbrs = NearestNeighbors(n_neighbors=1).fit(My)
    nns = nbrs.kneighbors(Mx, 1, return_distance=False)

    # calculate initialization dictionary
    d = [[0 for i in range(vocab_y)] for j in range(vocab_x)]

    for i in range(vocab_x):
        nn = nns[i]
        d[i][nn[0]] = 1

    # save initialization dictionary
    with open("../init_dict.json", "w+") as f:
        data = json.dumps(d)
        f.write(data)

main()
