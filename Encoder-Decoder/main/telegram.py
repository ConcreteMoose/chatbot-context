#
import json 
import requests
import time
import urllib
import numpy as np
from numpy.random import choice
import re
import os

# imports for model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Masking
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.models import model_from_json 
from keras.layers.wrappers import Bidirectional
from attention_decoder import AttentionDecoder
import processing


TOKEN = "553617004:AAGFq_FMPlojaJcdn4dzrWgUmfnUU3gOyTs" 
URL = "https://api.telegram.org/bot{}/".format(TOKEN)

chats = []

#define model:
MAX_LENGTH   = 150
NR_WORDPIECE = 512
LATENT_DIM   = 256

model = Sequential()
model.add(Masking(mask_value=0.,input_shape=(MAX_LENGTH, NR_WORDPIECE)))
model.add(Bidirectional(LSTM(LATENT_DIM, return_sequences=True)))
model.add(AttentionDecoder(LATENT_DIM*2, NR_WORDPIECE))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

# Check for weights, and if not found return error
if os.path.isfile('../res/model.h5'):
    model.load_weights('../res/model.h5')
    print("Modelweights loaded")
else:
    raise FileNotFoundError("Could not find model weights!!!")


def get_url(url):
    response = requests.get(url)
    content = response.content.decode("utf8")
    return content


def get_json_from_url(url):
    content = get_url(url)
    js = json.loads(content)
    return js


def get_updates(offset=None):
    url = URL + "getUpdates?timeout=100"
    if offset:
        url += "&offset={}".format(offset)
    js = get_json_from_url(url)
    return js


def get_last_update_id(updates):
    update_ids = []
    for update in updates["result"]:
        update_ids.append(int(update["update_id"]))
    return max(update_ids)


# If input is received, predict output using model
def echo_all(updates):
    for update in updates["result"]:
        text = update["message"]["text"]
        chat = update["message"]["chat"]["id"]
        
        inputs = processing.sentence2onehot(text).reshape((1,150,512))
        output = model.predict(inputs).reshape((150,512))
        print(output.shape)
        for vec in output:
            print(np.argmax(vec))


        print("output:")
        send_message(processing.onehot2sentence(output), chat)


def get_last_chat_id_and_text(updates):
    num_updates = len(updates["result"])
    last_update = num_updates - 1
    text = updates["result"][last_update]["message"]["text"]
    chat_id = updates["result"][last_update]["message"]["chat"]["id"]
    return (text, chat_id)


def send_message(text, chat_id):
    text = urllib.parse.quote_plus(text)
    url = URL + "sendMessage?text={}&chat_id={}".format(text, chat_id)
    get_url(url)


def main():
    last_update_id = None
    while True:
        updates = get_updates(last_update_id)
        if len(updates["result"]) > 0:
            last_update_id = get_last_update_id(updates) + 1
            echo_all(updates)
        time.sleep(0.5)


if __name__ == '__main__':
    main()
