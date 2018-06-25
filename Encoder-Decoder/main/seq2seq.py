from random import randint
from numpy  import array
from numpy  import argmax
from numpy  import array_equal
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
import csv
import multiprocessing
from multiprocessing import Pipe
from multiprocessing import Process
import time
import os
import json

FILE_PATH    = '../res/dialogueText_301.csv'	# Not included, too big for Github
MAX_LENGTH   = 150			# max length of sentence
NR_WORDPIECE = 512			# Number of unique wordpieces
TRAINING_TIME = 6*24*60*60		# Training time in seconds
LATENT_DIM   = 256			# dimensionality of output

    

#generate batches
def batch_generator(pipe):
    stop = False
    while not stop:
        with open(FILE_PATH,'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            conversation_current = '1.tsv'
            X = list()
            y = list()
            skip_first = True
            for cur_row in reader:
                if stop:
                    break                
                if not skip_first:
                    # if end of conversation, send batch and clear local mem
                    if cur_row[1] != conversation_current or len(y) >= 100:
                        X.pop()
                        X = array(X)
                        y = array(y)
                        if len(y) > 0:
                            pipe.send( (X,y) )
                            stop = pipe.recv()
                        X = list()
                        y = list()
                        conversation_current = cur_row[1]

                    #if beginning of conversation, store previous row
                    if len(X) == 0:
                        X.append( one_hot_encode( get_sequence( cur_row[-1],MAX_LENGTH), NR_WORDPIECE ))
                    else:
                        X.append( one_hot_encode( get_sequence( cur_row[-1],MAX_LENGTH), NR_WORDPIECE ))
                        y.append( one_hot_encode( get_sequence( cur_row[-1],MAX_LENGTH), NR_WORDPIECE ))
                else:
                    skip_first = False
                    

def get_sequence( sequence,  max_length):
    sequence = sequence.replace('u','')
    sequence = list(map(int,sequence.split(' ')))
    sequence += [-1 for _ in range( max_length - len(sequence))]
    sequence = sequence[:max_length]
    return sequence
    

# one hot encode sequence
def one_hot_encode(sequence, n_unique):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        if not value == -1:
            vector[value] = 1
        encoding.append(vector)
    return array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]


#define model
model = Sequential()
model.add(Masking(mask_value=0.,input_shape=(MAX_LENGTH, NR_WORDPIECE)))
model.add(Bidirectional(LSTM(LATENT_DIM, return_sequences=True)))
model.add(AttentionDecoder(LATENT_DIM*2, NR_WORDPIECE))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.summary()

# If weights are found, load them
if os.path.isfile('../res/model.h5'):
    model.load_weights('../res/model.h5')
    print("Modelweights loaded")


parent_conn, child_conn = Pipe()
batch_gen_process = Process(target=batch_generator, args=(child_conn,))
batch_gen_process.start()

stop = False
start_time = time.time()
save_time  = time.time()

# Start training
while not stop:
    stop = ( time.time() - start_time > TRAINING_TIME )
    X,y = parent_conn.recv()
    parent_conn.send( stop )
    model.fit(X,y,epochs=1, verbose=0)
    
    # save model weights every 15 min
    if time.time() - save_time > 15*60:
        save_time = time.time()
        model.save_weights( '../res/model.h5' )
        print("Model saved")
        score = model.evaluate(X,y,verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))


model.save_weights( '../res/model.h5' )
print("Model saved")

score = model.evaluate(X,y,verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))



