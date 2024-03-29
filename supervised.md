# Naive Approach to Cross-Lingual Mapping of Word Embeddings

## Logistics

First, we'll import Word2Vec from gensim, and also a few utility libraries.

```python
from gensim.models import KeyedVectors
import json
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
```

## Getting The Data

We're also going to assume that the train and test data are already here, along with pretrained models for English and French word embeddings, in a subdirectory of the main directory of this project.

I was originally going to do this only for Word2Vec because of lack of pretrained Fasttext models for a lot of languages, but then I discovered this glorious treasure trove of word embedding models in both Word2Vec and Fasttext:  https://github.com/Kyubyong/wordvectors

And also pretrained models for English here: https://github.com/3Top/word2vec-api, linked by the same glorious soul that published the above set of models.

So, we'll assume that the models- Word2Vec and fasttext- for English and French- are downloaded and in the =../data= directory. We're going to do this on the word2vec bin files first to see if they work. And like primitives, we hardcode the locations like so:


```python
train_file = "../data/train.txt"
test_file = "../data/test.txt"

x_model_vec = "../data/GoogleNews-vectors-negative300.bin"
y_model_vec = "../data/fr.vec"
```

** Reading

We read the data in like so: (this is the altered function, refer to the next subsection to see why I altered it)

#+NAME: read_in_data
#+BEGIN_SRC python

  def read_data(filename):

      with open(filename) as f:
          s = f.readlines()
      data = list(map(lambda x: x.rstrip("\n"), s))
      data2 = list(map(lambda x: x.split(" "), data))
      return data2

  train_data = read_data(train_file)
  test_data = read_data(test_file)
#+END_SRC

** Testing if it's read properly

I'm also going to make a small test print function to see if the data's read in properly. (In this post I'm exposing /everything/ about the way I work, including the small details like test prints. Orgmode allows me to add and remove code blocks at will, so I can just put this here and then remove it as I move on with writing the program.)

#+NAME: test_if_data_read
#+BEGIN_SRC python
  for i in range(3):
      print(train_data[i])
      print(test_data[i])
#+END_SRC

Aaaand the result is: 

#+BEGIN_QUOTE
['des', 'of\n']
['kiev', 'kyiv\n']
['les', 'the\n']
['kiev', 'kiev\n']
['est', 'is\n']
['sac', 'bag\n']
#+END_QUOTE

** Stable Time Loop From The Future

Whoops. I forgot to strip newlines. See, this is why I test print. Because everyone will inevitably forget details like this. So I go back and alter the function, forming a stable time loop (I just watched *Interstellar* okay), and...

#+BEGIN_QUOTE
['des', 'of']
['kiev', 'kyiv']
['les', 'the']
['kiev', 'kiev']
['est', 'is']
['sac', 'bag']
#+END_QUOTE

Yay. Now I snip the test function codeblock from my code (emacs magic), and move on...

* Words to Embeddings

This is the part where we, er, convert words to vectors.

** Loading Pretrained Models

Loading the pretrained models using gensim:

#+NAME: load_pretrained_models
#+BEGIN_SRC python
fr_model = KeyedVectors.load_word2vec_format(x_model_vec, binary=True)
en_model = KeyedVectors.load_word2vec_format(y_model_vec, binary=False)
#+END_SRC

Now to see if they loaded, let's try getting embeddings for a few sample words from both languages:

#+NAME: test_if_pretrained_models_loaded
#+BEGIN_SRC python
print(len(fr_model.wv["bag"]))
print((len(en_model.wv["sac"])))
#+END_SRC

** Word Pairs to Vector Pairs

A function to take a list of word pairs and convert them to a pair of (word, list) lists, put in a try/catch block because of words that may not be in the pretrained model's vocabulary throwing up errors:

#+NAME: word_to_vector_pairs
#+BEGIN_SRC python
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

#+END_SRC

And to test:

#+NAME: test_pairs2vec
#+BEGIN_SRC python

  print(len(train_en_vecs[0]))
  print(len(test_en_vecs[0]))

#+END_SRC

** Saving

To stop the time-consuming process of loading the entire pretrained model every time I run this code, I'm saving the list of vector pairs in four files: =eng_train.json=, =fr_train.json=, =eng_test.json=, =fr_test.json=. Each file will have a list of (word, vector) pairs in JSON format.

#+NAME: save_pairs2vec
#+BEGIN_SRC python

  def save_pairs2vec(vecs, filename):
      with open(filename, "w+") as f:
          json.dump(vecs, f)

  save_pairs2vec(train_en_vecs, "eng_train.json")
  save_pairs2vec(train_fr_vecs, "fr_train.json")
  save_pairs2vec(test_en_vecs, "eng_test.json")
  save_pairs2vec(test_fr_vecs, "fr_test.json")
#+END_SRC