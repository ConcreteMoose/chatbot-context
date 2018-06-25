import multiprocessing as mp
from multiprocessing import Process, Pipe, Lock, Queue
from sklearn.cluster import KMeans
from sentence_parser import *
import random

import pickle
import os
import json
import numpy as np

"""
    run the chatbot without context clustering
"""

NR_OF_WORKERS = 20
NR_OF_WORDPIECES = 512
sp = SentenceParser()

# comparison of tf-idf vectors is split over CPU's for efficiency
def f(idx,pipe):
    with open('../../res/vectors/'+str(idx)+'.pkl','rb') as infile:
        vectors = pickle.load(infile)
    for vector in vectors['vecs']:
        vector['in'] = np.array(vector['in'])

    pipe.send( True )
    
    while True:
        task = pipe.recv() #recieve input sentence as tf-idf vector
        best_sent = ""
        best_sim  = None
        for i in range(len(vectors['vecs'])): #compare with all known vectors
            sim = 1./np.linalg.norm( vectors['vecs'][i]['in'] - task )
            if best_sim == None or sim > best_sim: #find the best one
                best_sim = sim
                best_sent = vectors['vecs'][i]['out']
        #send best one back to central worker
        pipe.send({'score':best_sim,'answer':best_sent})   
    


lock    = Lock()
workers = []
pipes   = []
for i in range(NR_OF_WORKERS): # init workers
    parent_conn, child_conn = Pipe()
    pipes.append(parent_conn)
    workers.append(Process(target=f, args=(i,child_conn)))
    workers[-1].start()

for pipe in pipes: # wait for workers to finish loading data
    pipe.recv()
print("Workers ready")

while True: # game-loop
    var = input("> ") #get input from user
    tfidf_vector = sp.sentence2tfidf(var) #convert input to tf-idf
    for pipe in pipes: #send tf-idf to all processes
        pipe.send(np.array(tfidf_vector))
    best_sent = "" #compare best results from all workers and select global best
    best_sim  = None
    for pipe in pipes:
        worker_best = pipe.recv()
        if best_sim == None or worker_best['score'] > best_sim:
            best_sim = worker_best['score']
            best_sent = worker_best['answer']
    print(sp.sequence2sentence(best_sent)) #convert best answer to readable sentence and show to user

