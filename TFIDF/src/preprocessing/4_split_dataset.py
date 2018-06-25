import pickle

"""
    Split dataset into 20 partitions. Analysis during conversations can be split over 20
    CPU cores, which significantly speeds up the process
"""

with open('../../res/vectors.pkl','rb') as f:
    all_vecs = pickle.load(f)

print(len(all_vecs['vecs']))

#make 20 different datastructures
documents = [{'vecs':[]} for _ in range(20)]
i = 0
j = 0
for vec in all_vecs['vecs']: #iteratively assign vector to document file
    print(j,end='\r')
    documents[i]['vecs'].append(vec)
    j += 1
    i += 1
    i %= 20

#store partitions on drive
for i in range(len(documents)):
    print(i,len(documents[i]['vecs']))
    with open('../../res/vectors/' + str(i) + '.pkl','wb+') as outfile:
        pickle.dump(documents[i],outfile,protocol=pickle.HIGHEST_PROTOCOL)
