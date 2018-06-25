# basic telegram bot
# https://www.codementor.io/garethdwyer/building-a-telegram-bot-using-python-part-1-goi5fncay
# https://github.com/sixhobbits/python-telegram-tutorial/blob/master/part1/echobot.py

import json 
import requests
import time
import urllib
import numpy as np
#import database_parser
from numpy.random import choice
import re
import os
# python3: urllib.parse.quote_plus
# python2: urllib.pathname2url

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


TOKEN = "553617004:AAGFq_FMPlojaJcdn4dzrWgUmfnUU3gOyTs" # don't put this in your repo! (put in config, then import config)
URL = "https://api.telegram.org/bot{}/".format(TOKEN)

CHOICES = ["Waarom?","Ik snap het niet", "Hé?","Matthijs' haar zit altijd goed"]


#Q = database_parser.Querier('Filmquotes.xml')

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

def echo_all(updates):
    for update in updates["result"]:
        text = update["message"]["text"]
        chat = update["message"]["chat"]["id"]
        
        if True:
            inputs = processing.sentence2onehot(text).reshape((1,150,512))
            output = model.predict(inputs).reshape((150,512))
            print(output.shape)
            for vec in output:
                print(np.argmax(vec))


            print("output:")
            #print(output)
            send_message(processing.onehot2sentence(output), chat)

'''
def echo_all(updates):
    for update in updates["result"]:
        text = update["message"]["text"]
        chat = update["message"]["chat"]["id"]
        if not chat in chats:
            chats.append(chat)
            send_message("Hello, a pleasure as always. I am Brepo, the overly dramatic film bot. I don't like listening to people much, so excuse me if I ignore you and go on a rant. On the bright side, if you'd like to get some 'info' or a 'quote' from some movie and you're too lazy to search for it yourself, just ask me and I'll google it for you.",chat)
            return
        answer = ""
        if "quote" in text.lower():
            remq = re.compile(r'^quote\s+|\s+quote\s+')
            film = re.split(remq,text)[-1]
            answer = Q.returnQuote(remq.sub('',film.lower()))
        elif "info" in text.lower():
            rema = re.compile(r'^info\s+|\s+info\s+')
            film = re.split(rema,text)[-1]
            answer = Q.returnInfo(rema.sub('',film.lower()))
        else:
            answer = m.brepo_says(text)
        send_message(answer, chat)
'''


def get_last_chat_id_and_text(updates):
    num_updates = len(updates["result"])
    last_update = num_updates - 1
    text = updates["result"][last_update]["message"]["text"]
    chat_id = updates["result"][last_update]["message"]["chat"]["id"]
    return (text, chat_id)


def send_message(text, chat_id):
    text = urllib.parse.quote_plus(text) # urllib.parse.quote_plus(text) # (python3)
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
