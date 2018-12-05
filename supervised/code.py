from gensim.models import KeyedVectors
import json
import numpy as np
import sys


train_file = "../data/train.txt"
test_file = "../data/test.txt"

x_model_vec = "../data/GoogleNews-vectors-negative300.bin"
y_model_vec = "../data/fr.vec"


def read_data(filename):

    with open(filename) as f:
        s = f.readlines()
    data = list(map(lambda x: x.rstrip("\n"), s))
    data2 = list(map(lambda x: x.split(" "), data))
    return data2

train_data = read_data(train_file)
test_data = read_data(test_file)
fr_model = KeyedVectors.load_word2vec_format(x_model_vec, binary=True)
en_model = KeyedVectors.load_word2vec_format(y_model_vec, binary=False)
def pairs2vec(ls):

    en_vecs = []
    fr_vecs = []
    for i in ls:
        try:
            fr_word = i[0]
            en_word = i[1]
            fr_vec = fr_model.wv[fr_word]
            en_vec = en_model.wv[en_word]
            en_vecs.append((en_word, en_vec.tolist()))
            fr_vecs.append((fr_word, fr_vec.tolist()))
        except KeyError:
            continue
    return en_vecs, fr_vecs

train_en_vecs, train_fr_vecs = pairs2vec(train_data)
test_en_vecs, test_fr_vecs = pairs2vec(test_data)


def save_pairs2vec(vecs, filename):
    with open(filename, "w+") as f:
        json.dump(vecs, f)

save_pairs2vec(train_en_vecs, "eng_train.json")
save_pairs2vec(train_fr_vecs, "fr_train.json")
save_pairs2vec(test_en_vecs, "eng_test.json")
save_pairs2vec(test_fr_vecs, "fr_test.json")
