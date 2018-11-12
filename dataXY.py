from scipy.sparse import *
import numpy as np
import pickle
import random


def create_alloutput(tweets_data_tot,name_file):
    w_emb=np.load('embeddings_nltk.npy')
    with open('vocab_nltk.pkl', 'rb') as f:
        tot_vocab = pickle.load(f)
    with open('cooc_nltk.pkl', 'rb') as f:
        cooc = pickle.load(f)
    #-------------X & Y for DNN
        # initialize the input/output arrays that will contain the feature representation of our training tweets
    X = []
    y = []
    index = 0

    # loop on all tweets
    for line in tweets_data_tot:

        # x contain all our word embedding representation (for words in our vocab) for a given tweet
        x = []
        for word in line :
            if word in tot_vocab.keys() : 
                word_emb = w_emb[tot_vocab[word]]
                x.append(word_emb)
            else : 
                continue    

        # test if there is no word in our vocab for a certain tweet
        if len(x) != 0 :     
            # append the average word embedding representation of all our words (in the vocab) for a tweet
            X.append(np.mean(x, axis = 0))

            # associate the sentiment for the tweet : 0 for negative and 1 for positive
            if index<len(tweets_data_tot)/2:
                y.append(0)
            else: y.append(1)
        index += 1
    X = np.array(X)
    y=np.array(y)
    
    np.save(name_file+'_dnn', X)
    np.save(name_file+'_sol', y)

    
    
    
    
    #-------------X for CNN (same Y)
    
    X = []
    index = 0
    for line in tweets_data_tot:
        x = []
        for word in line :
            if word in tot_vocab.keys() : 
                indice_word = tot_vocab[word]
                x.append(indice_word)
            else: 
                continue
        if len(x) != 0 :
            X.append(x)
        else:
            g = np.random.randint(len(tot_vocab),size=10)
            X.append(np.array(g))
        index += 1

    X = np.array(X)

    np.save(name_file+'_cnn', X)

def create_test(tweets_data_tot,name_file):

    w_emb=np.load('embeddings_nltk.npy')
    with open('vocab_nltk.pkl', 'rb') as f:
        tot_vocab = pickle.load(f)
    with open('cooc_nltk.pkl', 'rb') as f:
        cooc = pickle.load(f)
    #-------------X & Y for DNN
        # initialize the input/output arrays that will contain the feature representation of our training tweets
    X = []
    index = 0

    # loop on all tweets
    for line in tweets_data_tot:

        # x contain all our word embedding representation (for words in our vocab) for a given tweet
        x = []
        for word in line :
            if word in tot_vocab.keys() : 
                word_emb = w_emb[tot_vocab[word]]
                x.append(word_emb)
            else : 
                continue    

        # test if there is no word in our vocab for a certain tweet
        if len(x) == 0 :

            g = np.random.randint(len(tot_vocab))
            X.append(w_emb[g])


        else :      

            # append the average word embedding representation of all our words (in the vocab) for a tweet
            X.append(np.mean(x, axis = 0))

  
        index += 1
    X = np.array(X)
    
    np.save(name_file+'_test_dnn', X)
                
                
#-------------X for CNN (same Y)
    
    X = []
    index = 0
    for line in tweets_data_tot:
        x = []
        for word in line :
            if word in tot_vocab.keys() : 
                indice_word = tot_vocab[word]
                x.append(indice_word)
            else: 
                continue
        if len(x) != 0 :
            X.append(x)
        else:
            g = np.random.randint(len(tot_vocab),size=10)
            X.append(np.array(g))
        index += 1

    X = np.array(X)

    np.save(name_file+'_test_cnn', X)