import random
import json
import pickle
import numpy as np
import warnings

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

lemmatizer = WordNetLemmatizer()  # Lemmatize the individual word
intents = json.loads(open('intents.json').read())  # Load json file and reading it as text and passing

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)  # tokenize the pattern
        words.extend(word_list)  # add the collection of tokenize words into the word []
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(words)
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
print(sorted(set(words)))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))  # save it into pickle
pickle.dump(classes, open('classes.pkl', 'wb'))

# Machine Learning Part We have a lot of characters and words etc. But that is not numerical value to feed into
# neural network. We need to represent this words as numerical values and use bag of word (bow) where we are going to
# set individual word values to either 0 or 1 depending on if it's occurring the particular pattern
training = []
output_empty = [0] * len(classes)  # Templates of zero, so we need as many zeros as classes

# Training the neural Network
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)  # To identify if it occurs in the pattern

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Neural Network Model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)
print("Done")

print(model.summary())

