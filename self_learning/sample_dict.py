"""Generate a sample dictionary for training"""

import json
import random

def random_dict():

    datafile = "/home/kw/Assignments/multilingual-word-embeddings/data.json"
    n = 100

    with open(datafile, "r+") as f:
        s = f.read()
        data = json.loads(s)

    vo_en = data["vocab_english"]
    vo_hi = data["vocab_hindi"]
    mat_en = data["matrix_english"]
    mat_hi = data["matrix_hindi"]

    vo_en =  vo_en[:n]
    vo_hi =  vo_hi[:n]
    mat_en =  mat_en[:n]
    mat_hi =  mat_hi[:n]

    d = []
    for i in range(n):
        subd = []
        val = random.randint(1, n)
        for j in range(n):
            if j == (val - 1):
                subd.append(1)
            else:
                subd.append(0)

        d.append(subd)

    return d, vo_en, vo_hi, mat_en, mat_hi




