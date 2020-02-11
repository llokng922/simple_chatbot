import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize

import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

stemmer = LancasterStemmer()
print("\x1b[0;37m")
with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle","rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            words_temp = word_tokenize(pattern)
            words.extend(words_temp)
            docs_x.append(words_temp)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words))) #make sure there is no duplicates

    labels = sorted(labels)

    # Bag of Words (One-Hot Encoding)
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x,doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)
    
    with open("data.pickle","wb") as f:
        pickle.dump((words, labels, training, output),f)



try:
    model.load("model.tflearn")
except:
    tf.reset_default_graph()

    net = tflearn.input_data(shape=[None,len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]),activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for s_word in s_words:
        for i,w in enumerate(words):
            if w == s_word:
                bag[i]=1

    return np.array(bag)


def chat():
    print("Start talking with the bot!")
    print("Type quit to stop!")
    while True:
        inp = input("\x1b[1;32m You: ")
        if inp.lower() == "quit":
            break
        
        results = model.predict([bag_of_words(inp, words)])[0]
        results_idx = np.argmax(results)
        tag = labels[results_idx]
        
        # Normal Response
        if results[results_idx]>0.8:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            
            print("\x1b[1;36m Crypto Guy: ", random.choice(responses))
        else:
            print("\x1b[1;38m Crypto Guy: I don't really get that, go ahead and ask another question.")

chat()
print("\x1b[0;37m")