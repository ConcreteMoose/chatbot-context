import multiprocessing as mp
from multiprocessing import Process, Pipe, Lock, Queue
import pickle
import os
import json

"""
    Turn all wordpiece-sequences in the conversations into tf-idf vectors
"""

NR_OF_WORKERS = 2
NR_OF_WORDPIECES = 512

#parallelization over 20 CPU cores
def f(idx,q,lock):
    lock.acquire()
    with open('../../res/idf.json','r') as idf_file:
        idf = json.load(idf_file)
    lock.release()
    while True:
        try:
            lock.acquire()
            folder = q.get(False)
            lock.release()
        except:
            lock.release()
            break
        vectors = []
        for filename in os.listdir(folder):
            with open( '/'.join( [folder, filename] ), 'r') as file:
                prev_line = None
                for line in file:
                    line = line.replace('\n','')
                    if not prev_line == None:
                        vectors.append( {'in':prev_line,'out':line} )
                    prev_line = [0.] * NR_OF_WORDPIECES
                    for wordpiece in line.split(' '):
                        prev_line[int(wordpiece.replace('u',''))] += idf[wordpiece]
        with open('../../res/c_vectors/' + folder.split('/')[-1] + '.pkl','wb+') as outfile:
            pickle.dump({'vecs':vectors},outfile,protocol=pickle.HIGHEST_PROTOCOL)

#initialize queue
queue = Queue()
for foldername in os.listdir('../../res/clusters'):
    queue.put('../../res/clusters/' + foldername)
NR_OF_CLUSTERS = queue.qsize()

lock = Lock()
workers = []
for i in range(NR_OF_WORKERS): #init workers
    workers.append(Process(target=f, args=(i,queue,lock)))
    workers[-1].start()
