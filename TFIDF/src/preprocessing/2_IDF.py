import multiprocessing as mp
from multiprocessing import Process, Pipe, Lock, Queue
import math


import pickle
import os
import json

"""
    To construct TF-IDF vectors for conversations and sentences later, we have to determine
    IDF for all the wordpieces in the dataset first.
"""

NR_OF_WORKERS = 20

#split processing of the dataset over multiple workers for optimalization.
def f(idx,q,lock,pipe):
    idfs = {} #each process keeps its own idf counts
    stop = False
    while not stop:
        try:
            lock.acquire()
            filename = q.get(False) #all the conversations to analyze
            #print(idx,filename)
            lock.release()
        except:
            lock.release()
            stop = True
            break
        try:
            with open(filename,'r') as file:
                for line in file:
                    line = line.replace('\n','')
                    for wordpiece in line.split(' '):
                        try:
                            idfs[wordpiece] += 1
                        except KeyError:
                            idfs[wordpiece] = 1
        except:
            continue
    print(idx,'finished')
    pipe.send(idfs) #send counts to central worker

#init queue
queue = Queue()
for filename in os.listdir('../../res/conversations'):
    queue.put('../../res/conversations/' + filename)
NR_OF_DOCUMENTS = queue.qsize()

lock    = Lock()
workers = []
pipes   = []
for i in range(NR_OF_WORKERS): #create processes
    parent_conn, child_conn = Pipe()
    pipes.append(parent_conn)
    workers.append(Process(target=f, args=(i,queue,lock,child_conn)))
    workers[-1].start()

while queue.qsize() > 0:
    print(str(queue.qsize()) + '                 ', end='\r')

for w in workers: #wait for processes to finish
    w.join()

idfs = {}
for pipe in pipes: #merge all counts from the workers into one
    w_idfs = pipe.recv()
    for wordpiece in w_idfs:
        try:
            idfs[wordpiece] += w_idfs[wordpiece]
        except KeyError:
            idfs[wordpiece] =  w_idfs[wordpiece]

#turn counts into log idfs
for wordpiece in idfs:
    idfs[wordpiece] = 1. / (float(idfs[wordpiece]) / NR_OF_DOCUMENTS)

for key in idfs:
    idfs[key] = math.log(1. + idfs[key])

#store in file
with open('../../res/idf.json','w+') as json_out:
    json.dump(idfs,json_out,indent=2)
