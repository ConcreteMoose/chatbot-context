import pickle
import os

"""
    split datasets of tf-idf vectors into 20 partitions for efficient analysis
    during runtime
"""

i = 0
for filename in os.listdir('../../res/c_vectors'):
    with open('/'.join(['../../res/c_vectors',filename]),'rb') as f:
        whole = pickle.load(f)

    partitions = [{'vecs':[]} for _ in range(20)]
    j = 0
    p = 0
    for vec in whole['vecs']: #iteratively assign a vector to a partition
        print(i,j,end='\r')
        partitions[p]['vecs'].append(vec)
        j += 1
        p += 1
        p %= 20
    i += 1

    #save partitions to pickle files
    os.remove('/'.join(['../../res/c_vectors',filename]))
    folder = '/'.join(['../../res/c_vectors',filename.split('.')[0]])
    if not os.path.exists( folder ):
        os.makedirs( folder )
    for p in range(len(partitions)):
        with open( '/'.join([folder,str(p)+'.pkl']), 'wb+') as outfile:
            pickle.dump(partitions[p], outfile, protocol=pickle.HIGHEST_PROTOCOL)
