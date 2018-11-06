frim gensim.models import Word2Vec
import json
import numpy as np
import sys

# treat it as a simple classification problem

train_file = "../data/train.txt"
test_file = "../data/test.txt"

x_model = "../data/" + sys.argv[1]
y_model = "../data/" + sys.argv[2]

def load_data(filename):

    with open(filename, "r+") as f:
        s = f.read()
    data = []
    for i in s[:10]:
        data.append(i.split(" "))

    objs = list(map(lambda x: x[0], data))
    cls = list(map(lambda x: x[1]))

    return objs, cls

def load_model(filename):

    


    
