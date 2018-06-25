from sentence_parser import *
import os
import multiprocessing
from multiprocessing import Queue, Lock, Process, Pipe
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
import shutil
import pickle

"""
    convert all conversations to tf-idf document-vectors
    cluster conversations into 'context'-clusters
    store all conversation documents in their assigned cluster
"""

NR_OF_WORKERS = 20
NR_OF_CLUSTERS = 3
sp = SentenceParser()

def f(idx,q,rq,lock,pipe):
    while True:
        try:
            lock.acquire()
            filename = q.get(False)
            lock.release()
        except:
            lock.release()
            break
        #print(idx,filename,"                           ",end='\r')
        with open(filename,'r') as file:
            lines = []
            for line in file:
                lines.append(np.array(sp.sentence2tfidf(line)))
            lines = np.array(lines)
            lines = np.mean(lines,axis=0)
            lock.acquire()
            rq.put((filename,lines))
            lock.release()
    print("Worker",idx,"finished.")
    pipe.send("Worker " + str(idx) + " finished.")
    return
                
queue = Queue()
for filename in os.listdir('../../res/conversations'):
    queue.put('../../res/conversations/' + filename)

results_queue = Queue()

print( len(os.listdir('../../res/conversations')))
print( queue.qsize() )

lock = Lock()
pipes = []
workers = []

for i in range(NR_OF_WORKERS):
    parent_conn, child_conn = Pipe()
    pipes.append(parent_conn)
    workers.append( Process( target=f, args=(i,queue,results_queue,lock,child_conn)))
    workers[-1].start()


for pipe in pipes:
    pipe.recv()

doc_vecs = []
filenames = []
while results_queue.qsize() > 0:
    (filename,lines) = results_queue.get()
    doc_vecs.append(lines / np.linalg.norm(lines))
    filenames.append(filename)
doc_vecs = np.array(doc_vecs)
#print(doc_vecs.shape)

# do k-means clustering
kmeans = KMeans(n_clusters=NR_OF_CLUSTERS,init='k-means++')
kmeans.fit(doc_vecs)

"""
    if we're working locally (4 CPU's in our case), perform PCA and show clustering
    in a plot for debugging.
"""
if multiprocessing.cpu_count() < 10:
    print(kmeans.cluster_centers_)
    print(kmeans.labels_)

    pca = PCA(n_components=3)
    pca.fit(doc_vecs)
    pca_docs = pca.transform(doc_vecs)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter( pca_docs[:,0], pca_docs[:,1], pca_docs[:,2], c=kmeans.labels_, cmap='rainbow' )
    plt.show()

#store each conversation in its assigned cluster
for i in range(NR_OF_CLUSTERS):
    if not os.path.exists('../../res/clusters/' + str(i).zfill(2)):
        os.makedirs('../../res/clusters/' + str(i).zfill(2))
with open('../../res/cluster_centroids.pkl','wb+') as f:
    pickle.dump(kmeans.cluster_centers_,f)
for i in range(len(filenames)):
    shutil.copy( filenames[i], '../../res/clusters/' + str( kmeans.labels_[i] ).zfill(2) + '/' + filenames[i].split('/')[-1] )
