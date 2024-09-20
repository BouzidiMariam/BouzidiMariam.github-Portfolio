#!/usr/bin/env python
# coding: utf-8

# In[13]:


import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
nltk.download('punkt')
import numpy
#import tflearn
import tensorflow
import random
import json
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


with open("intents.json") as file:
    data = json.load(file)
    print(data)


# In[15]:


try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        print(intent)
        for pattern in intent["patterns"]:
            print(pattern)

            wrds = nltk.word_tokenize(pattern)
            print(wrds)
            words.extend(wrds)
            print(words)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
            print(docs_y)

        if intent["tag"] not in labels:
            labels.append(intent["tag"])
            print(labels)

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    print(words)

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        print(bag)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)
    print(training)
    print(output)


# In[16]:


with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

model = Sequential([
Dense(8, input_shape=(len(training[0]),), activation='relu'),
Dense(8, activation='relu'),
Dense(len(output[0]), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

try:
 model.load("model.h5")
except:
 model.fit(training, output, epochs=1000, batch_size=8, verbose=1)
 model.save('my_model.keras')

