"""Generate a sample dictionary for training"""

import json
import random

def get_dict():

    dictfile = "../init_dict.json"
    datafile = "../data.json"
    n = 2000

    with open(datafile, "r+") as f:
        s = f.read()
        data = json.loads(s)
    with open(dictfile, "r+") as f:
        s = f.read()
        d = json.loads(s)

    vo_en = data["vocab_english"]
    vo_hi = data["vocab_hindi"]
    mat_en = data["matrix_english"]
    mat_hi = data["matrix_hindi"]

    vo_en =  vo_en[:n]
    vo_hi =  vo_hi[:n]
    mat_en =  mat_en[:n]
    mat_hi =  mat_hi[:n]

    return d, vo_en, vo_hi, mat_en, mat_hi


