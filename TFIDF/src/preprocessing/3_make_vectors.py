import multiprocessing as mp
from multiprocessing import Process, Pipe, Lock, Queue

import pickle
import os
import json
import numpy as np

"""
    convert documents of wordpiece sequences to TF-IDF vectors
"""

NR_OF_WORKERS = 20
NR_OF_WORDPIECES = 512

#split dataset over parallel workers for efficiency
def f(idx,q,lock,pipe):
    lock.acquire()
    with open('../../res/idf.json','r') as idf_file:
        idf = json.load(idf_file)
    lock.release()    
    vectors = []
    while True:
        try:
            lock.acquire() #get filename of conversation to analyze next
            filename = q.get(False)
            lock.release()
        except:
            lock.release()
            break
        with open(filename,'r') as file:
            prev_line = None
            for line in file:
                line = line.replace('\n','')
                if not prev_line == None: #store tf-idf vector of previous sentence and wordpiece sequence of current as answer
                    vectors.append( {'in':prev_line,'out':line} )
                prev_line = [0.] * NR_OF_WORDPIECES
                for wordpiece in line.split(' '):
                    try:
                        prev_line[int(wordpiece.replace('u',''))] += idf[wordpiece]
                    except:
                        print(wordpiece)
                        raise SyntaxError

    #temporarily store vectors somewhere. This datastructure is too big to send
    # over a pipe
    with open('../../res/' + str(idx) + '_vecs.pkl','wb+') as outfile:
        pickle.dump({'vecs':vectors},outfile,protocol=pickle.HIGHEST_PROTOCOL)


#init queue
queue = Queue()
for filename in os.listdir('../../res/conversations'):
    queue.put('../../res/conversations/' + filename)
NR_OF_DOCUMENTS = queue.qsize()

lock    = Lock()
workers = []
pipes   = []
for i in range(NR_OF_WORKERS): #init workers
    parent_conn, child_conn = Pipe()
    pipes.append(parent_conn)
    workers.append(Process(target=f, args=(i,queue,lock,child_conn)))
    workers[-1].start()

while queue.qsize() > 0:
    print(str(queue.qsize()) + '                 ', end='\r')


for w in workers:
    w.join()

vectors = {'vecs':[]} #merge vectors created by workers into single file
for idx in range(NR_OF_WORKERS):
    print("Saving worker",idx)
    with open('../../res/' + str(idx) + '_vecs.pkl','rb') as infile:
        w_vecs = pickle.load(infile)['vecs']
    for vec in w_vecs:
        vectors['vecs'].append(vec)

    #remove temporary file made by worker
    os.remove('../../res/' + str(idx) + '_vecs.pkl')

#store in file
with open('../../res/vectors.pkl','wb+') as outfile:
    pickle.dump(vectors,outfile,protocol=pickle.HIGHEST_PROTOCOL)
