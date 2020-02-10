import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize

import numpy as np
import tflearn
import tensorflow
import random
import json

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        words_temp = word_tokenize(pattern)
        words.extend(words_temp)
        docs_x.append(pattern)
        docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words))) #make sure there is no duplicates

labels = sorted(labels)

# Bag of Words (One-Hot Encoding)
training = []
output = []

out_empty = [0 for _ in range(len(classes))]

for x,doc in enumerate(doc_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append([bag])
    output.append(output_row)

training = np.array(training)
output = np.array(output)