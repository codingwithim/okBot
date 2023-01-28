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

# ignore warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

lemmatizer = WordNetLemmatizer()  # Lemmatize the individual word

# Load json file and reading it as text and passing
intents = json.loads(open('JSON_FILES/intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# tokenize the pattern and add the collection of tokenize words into the word []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# remove ignore letters and lemmatize the words
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

# sort classes
classes = sorted(set(classes))

# save words and classes in pickle
pickle.dump(words, open('pickle_files/words.pkl', 'wb'))
pickle.dump(classes, open('pickle_files/classes.pkl', 'wb'))

# Machine Learning Part
# We have a lot of characters and words etc. But that is not numerical value to feed into
# neural network. We need to represent this words as numerical values and use bag of word (bow) where we are going to
# set individual word values to either 0 or 1 depending on if it's occurring the particular pattern
training = []
output_empty = [0] * len(classes)  # Templates of zero, so we need as many zeros as classes

# Training the neural Network
for document in documents:
    bag = []
    word_patterns = document[0]
    # lemmatize words and convert to lower case
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        # To identify if it occurs in the pattern
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# shuffle training data
random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Neural Network Model
model = Sequential()
# add the first layer with 128 neurons and input shape as the length of train_x[0]. Activation function is relu
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# add dropout layer with dropout rate of 0.5
model.add(Dropout(0.5))
# add second layer with 64 neurons and relu activation function
model.add(Dense(64, activation='relu'))
# add dropout layer with dropout rate of 0.5
model.add(Dropout(0.5))
# add third layer with len(train_y[0]) neurons and softmax activation function
model.add(Dense(len(train_y[0]), activation='softmax'))

# create optimizer with learning rate of 0.01, decay of 1e-6, momentum of 0.9 and nesterov as True
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# compile the model with loss function as categorical_crossentropy, optimizer as sgd and metrics as accuracy
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fit the model on train_x and train_y for 1000 epochs with batch size of 5 and verbose as 1
hist = model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=5, verbose=1)
# save the model as chatbotmodel.h5
model.save('chatbotmodel.h5', hist)
print("Done")

# print the model summary
print(model.summary())


