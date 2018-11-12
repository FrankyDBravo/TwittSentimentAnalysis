from scipy.sparse import *
import numpy as np

#if we want all vocabulary: take n as len(occurences)
def create_vocab(occurences, n):
    vocab = [occurences[i][0] for i in range(n)]
    encode = dict(zip(vocab, range(n)))

    return (encode)

def create_cooc(tweets_data,vocab):
    # Create cooccurence matrix
    data, row, col = [], [], []
    counter = 1
    for line in tweets_data :

        tokens = [vocab.get(t, -1) for t in line]
        tokens = [t for t in tokens if t >= 0]
        
        for i in range(len(tokens)) :
            for j in range(len(tokens)) :
                if i==j:
                    continue      
                else:
                    if (tokens[i]==tokens[j]):
                        data.append(1/2)
                        row.append(tokens[i])
                        col.append(tokens[j])
                    else:
                        data.append(1)
                        row.append(tokens[i])
                        col.append(tokens[j])
        if counter % 10000 == 0:
            print(counter)
        counter += 1
    cooc = coo_matrix((data, (row, col)))
    print("summing duplicates (this can take a while)")
    cooc.sum_duplicates()
    #create standard matrix of sparse cooc
    cooc_m=cooc.toarray()
    print(cooc.shape)
    return cooc, cooc_m