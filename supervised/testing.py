from gensim.models import KeyedVectors
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
import sys


from training import dataLoad


from joblib import dump, load

models = load("model.joblib")

tst_en = dataLoad("eng_test.json")

# make a matrix of only the word embeddings
# and an associated list of vocabulary

en_vecs = []
en_words = []
for i in tst_en:
    en_words.append(i[0])
    en_vecs.append(i[1])



tst_fr = dataLoad("fr_test.json")

def find_transform(src):

    trans = []

    # loop to transform it
    for i in range(len(src)):
        a = models[i].coef_[0][0]
        b = models[i].intercept_[0]
        x = src[i]
        trans.append(a*x + b)

    return trans


def find_nn(trans):

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(en_vecs)
    distances, indices = nbrs.kneighbors([trans])

    index = indices[0][0]

    # remove that particular vector from the list
    # because the algorithm as current causes ALL french vectors
    # to translate to the english word 'collins'
    en_vecs.pop(index)

    word = en_words[index]
    return word

pairs = []

for i in tst_fr:
    pair = []

    word = i[0]
    vec = i[1]
    pair.append(word)

    # get the transform
    trans = find_transform(vec)
    # get the translated word
    translation = find_nn(trans)

    pair.append(translation)

    pairs.append(pair)

# now write it to file
with open("results.json", "w+") as f:
    json.dump(pairs, f)
