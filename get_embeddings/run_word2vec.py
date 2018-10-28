from gensim.models import Word2Vec
import json
import numpy as np

def load_sents(filename, n):

    ls = []
    with open(filename, "r+") as f:
        for i in range(n):
            ls.append(f.readline().split())

    return ls


def make_model(sentences, savefile):
    # train model
    model = Word2Vec(sentences, min_count=1)

    # save model
    model.save(savefile)
    return model

def to_float(ls):

    fl = []
    for i in ls:
        fl.append(np.asscalar(i))
    return fl


if __name__ == "__main__":

    sents_en = load_sents("IITB.en-hi.en", 10000)
    sents_hi = load_sents("IITB.en-hi.hi", 10000)

    print("Sentences loaded.")

    model_en = make_model(sents_en, "model_en.bin")
    model_hi = make_model(sents_hi, "model_hi.bin")

    print("Model trained.")

    words_en = list(model_en.wv.vocab)
    words_hi = list(model_hi.wv.vocab)

    print("Vocabulary acquired")
    
    mat_en = []
    for word in words_en:
        mat_en.append(to_float(model_en[word]))
    mat_hi = []
    for word in words_hi:
        mat_hi.append(to_float(model_hi[word]))

    print("Representation matrices of both languages made.")

    d = {
            "vocab_english": words_en,
            "matrix_english": mat_en,
            "vocab_hindi": words_hi,
            "matrix_hindi": mat_hi
        }

    data = json.dumps(d)
    with open("data.json", "w+") as f:
        f.write(data)
    print("Data written to file.")





