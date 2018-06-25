import multiprocessing as mp
from multiprocessing import Process, Pipe, Lock, Queue
from sklearn.cluster import KMeans
from sentence_parser import *
from scipy.spatial import distance as D
import random

import pickle
import os
import json
import numpy as np

NR_OF_WORKERS = 20
NR_OF_WORDPIECES = 512
sp = SentenceParser()


#analyze in parallel for speed-up
def f(idx, pipe):
    c_vecs = []
    #load data
    for folder in os.listdir('../../res/c_vectors'):
        with open('/'.join(['../../res/c_vectors',folder,str(idx)+'.pkl']),'rb') as infile:
            c_vecs.append(pickle.load(infile))
    for c_vec in c_vecs:
        for vec in c_vec['vecs']:
            vec['in'] = np.array(vec['in'])

    pipe.send( True )

    while True: 
        (cluster,sentence) = pipe.recv()
        if cluster == None: #if the cluster is not specified, search through all clusters
            best_sent = ""
            best_sim = None
            for c_vec in c_vecs:
                for i in range(len(c_vec['vecs'])):
                    sim = 1./np.linalg.norm( c_vec['vecs'][i]['in'] - sentence )
                    if best_sim == None or sim > best_sim:
                        best_sim = sim
                        best_sent = c_vec['vecs'][i]['out']
        else: #otherwise, search only in the specified cluster
            best_sent = ""
            best_sim = None
            for i in range(len(c_vecs[cluster]['vecs'])):
                sim = 1./np.linalg.norm( c_vecs[cluster]['vecs'][i]['in'] - sentence )
                if best_sim == None or sim > best_sim:
                    best_sim = sim
                    best_sent = c_vecs[cluster]['vecs'][i]['out']
        pipe.send({'score':best_sim,'answer':best_sent})

lock    = Lock
workers = []
pipes   = []
for i in range(NR_OF_WORKERS):
    parent_conn, child_conn = Pipe()
    pipes.append(parent_conn)
    workers.append(Process(target=f, args=(i,child_conn)))
    workers[-1].start()

for pipe in pipes: #wait for the processes to finish loading the data
    pipe.recv()
print("Hello, my name is Brepo. I'm a chatbot trained on the Ubuntu helpdesk dialogue corpus. How can I be of service?")

with open('../../res/cluster_centroids.pkl','rb') as f:
    centroids = pickle.load(f)

conversation = []
while True: #game loop
    var = input("> ") #get user input
    tfidf_vector = sp.sentence2tfidf(var)
    conversation.append(tfidf_vector) #add the input sentence to the conversation
    while len(conversation) > 7: #remember last 7 sentences in the conversation
        conversation.pop(0)
    context = np.mean(np.array(conversation),axis=0)
    context = context / np.linalg.norm(context) #determine context tf-idf vector
    cluster  = None
    distance = None
    for (idx,centroid) in enumerate(centroids): #compare context vector to centroids of context clusters
        centroid = centroid / np.linalg.norm(centroid)
        dist = np.linalg.norm(centroid - context)
        if distance == None or dist < distance:
            distance = dist
            cluster = idx
    for pipe in pipes: # send cluster id and tf-idf vector to all processes
        pipe.send((cluster,tfidf_vector))
    best_sent = ""
    best_sim = None
    for pipe in pipes: # select best result from results returned by processes
        worker_best = pipe.recv()
        if not worker_best['score'] == None:
            if best_sim == None or worker_best['score'] > best_sim:
                best_sim = worker_best['score']
                best_sent = worker_best['answer']
    conversation.append( sp.sentence2tfidf(best_sent)) #add response to conversation
    print(sp.sequence2sentence(best_sent)) # show response to user

