import numpy as np
from itertools import combinations
import csv


# AVERAGING THE VECTOR REPRESENTATION OF TWEETS WORDS
#   Takes in input the vocabulary, the embedding and all the tweets. For eac tweet it takes the mean value of the vector representation of its words.
def build_training_set(tot_vocab , w_emb , tweets_data_tot):

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
        if len(x) > 0 :
            X.append(np.mean(x, axis = 0))
        
            # associate the sentiment for the tweet : 0 for negative and 1 for positive
            if index < len(tweets_data_tot)/2 :
                y.append(-1)
            else :
                y.append(1)
    
        index += 1
    X = np.array(X)
    y = np.array(y)
    return X,y


# AVERAGING THE VECTOR REPRESENTATION OF TWEETS WORDS
#   Takes in input the vocabulary, the embedding and all the tweets. For eac tweet it takes the mean value of the vector representation of its words.
def build_test_set(tot_vocab , w_emb , tweets_data_tot):

    X = []

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
        else:           
            # append the average word embedding representation of all our words (in the vocab) for a tweet
            X.append(np.mean(x, axis = 0))
            
    X = np.array(X)
    return X



# DATA NORMALIZATION
#    Takes as input a matrix NxD representing all the dataset. It computes the average observation array with dimension 1x and then the standard deviation vector. Finally it subtracts the mean vector (component-wise) to all the observations and divide (still component-wise) for all the standard deviations.

def normalizator(X):
    mean_X = np.mean(X,axis=0)
    std_X = np.std(X,axis=0)
    X_norm = (X - mean_X[:None])/std_X[:None]
    return X_norm



# CREATION OF SUBMISSION FILE
#    Creates an output file in csv format for submission to kaggle. It takes as arguments the prediction 'y_pred' and the output file name.

def create_csv_submission(y_pred, name):
    lung = len(y_pred)
    ids = [i+1 for i in range(lung)]
    for i in range(lung):
        if y_pred[i] > 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = -1
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
            
# POLINOMIAL


def build_poly(x, degree):
    [N,D] = x.shape
    phi_mat = []
    for i in range(D):
        col = x[:,i]
        power_col = []
        for j in range(degree):
            power_col.append(np.power(col,j+1))
        phi_mat.append(power_col)
    phi_mat = np.vstack(phi_mat).T
    return phi_mat


def build_poly_extended(x, degree):
    [N,D] = x.shape
    phi_mat = []
    # Single terms
    for i in range(D):
        col = x[:,i]
        power_col = []
        for j in range(degree):
            power_col.append(np.power(col,j+1))
        phi_mat.append(power_col)
    # Mixed terms
    col_ind = np.arange(D)
    combs = list(combinations(col_ind,degree))
    print(combs)
    for i in range(len(combs)):
        ind = combs[i]
        temp_mix = x[:,ind[0]]
        for j in range(degree-1):
            temp_mix = np.multiply(temp_mix , x[:,ind[j+1]])
        phi_mat.append(temp_mix)
    phi_mat = np.vstack(phi_mat).T
    return phi_mat
