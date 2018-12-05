from gensim.models import KeyedVectors
import json
import numpy as np
from sklearn.linear_model import LinearRegression
import sys


# utility function to load data
def dataLoad(filename):
    with open(filename, "r+") as f:
        data = json.load(f)
    return data

# load data
tr_en = dataLoad("eng_train.json")
tr_fr = dataLoad("fr_train.json")
tst_en = dataLoad("eng_test.json")
tst_fr = dataLoad("fr_test.json")

# strip training data to two lists of one-D values
vecs_en = list(map(lambda x: x[1], tr_en))
vecs_fr = list(map(lambda x: x[1], tr_fr))

models = []

# for each dimension
for i in range(300):

    # initialize scikit-learn's linear regression model
    reg = LinearRegression()

    # fetch i'th dimension of all vectors
    dim_fr = list(map(lambda x: [x[i]], vecs_fr))
    dim_en = list(map(lambda x: [x[i]], vecs_en))

    # train the model
    reg.fit(dim_fr, dim_en)

    # append the trained model to the list of models
    models.append(reg)


for i in range(300):
    print(models[i].coef_)
