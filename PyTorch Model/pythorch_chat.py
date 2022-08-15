import random
import json
import os
import sqlite3
import torch
import datetime

from pytorch_model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

print(model.eval())
bot_name = "okBot"
print("Let's chat! (type 'quit' to exit)")
if os.path.exists("chat.db"):
    conn = sqlite3.connect("chat.db")
    c = conn.cursor()

else:
    conn = sqlite3.connect("chat.db")
    c = conn.cursor()

    query = (''' CREATE TABLE IF NOT EXISTS CHAT
                (
                DATE SMALLDATETIME,
                INPUT VARCHAR(50) NOT NULL,
                OUTPUT VARCHAR(50) NOT NULL,
                INTENT VARCHAR(50) NOT NULL
                );
                ''')
    c.execute(query)

while True:
    # sentence = "do you use credit cards?"
    now = datetime.datetime.now()
    message = input("You: ")
    if message == "quit":
        break

    sentence = tokenize(message)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.6:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                x = random.choice(intent['responses'])
                print(f"{bot_name}:" + x)
                c.execute("INSERT INTO CHAT VALUES (?, ?, ?,?)", [now, message, x, tag])
                conn.commit()
