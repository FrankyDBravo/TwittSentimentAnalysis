import numpy as np
from collections import Counter
import nltk
import pickle

def preprocess(tweets_data_path):

    # array to store all tweets
    tweets_data = []

    # open the file containing tweets
    tweets_file = open(tweets_data_path, "r")
    with open('stopwords.pkl', 'rb') as f:
        stopwords = pickle.load(f)
    # array of the words to remove
    to_remove =stopwords[0:109]

    # to count the occurence of each word
    count_test = Counter()
    # variable to get the total number of words in a positive tweet
    pos_total_words = 0

    # loop on each tweet
    for line in tweets_file:
        tweet_line = np.array(line.split(" "))

        # loop for each word on a tweet line
        for word in tweet_line :

            # don't store the word if it on the to_remove list (useless for our analysis)
            if word in to_remove :

                # delete the word from the tweet
                tweet_line = np.delete(tweet_line, np.where(tweet_line == word))


        # update the counter
        count_test.update(tweet_line)

        # add the array containing all thw words of a certain tweet to our all tweets array
        tweets_data.append(tweet_line)



    return tweets_data,count_test

def preprocess_nltk(tweets_data_path):

    # array to store all tweets
    tweets_data = []

    # open the file containing tweets
    tweets_file = open(tweets_data_path, "r")

    # get list of stopwords from the STOPWORD api
    with open('stopwords.pkl', 'rb') as f:
        stopwords = pickle.load(f)
    # array of the words to remove
    to_remove =stopwords

    # to count the occurence of each word
    count_test = Counter()
    # variable to get the total number of words in a positive tweet
    pos_total_words = 0

    # loop on each tweet
    for line in tweets_file:
        tweet_line = np.array(nltk.word_tokenize(line))

        # loop for each word on a tweet line
        for word in tweet_line :

            # don't store the word if it on the to_remove list (useless for our analysis)
            if word in to_remove :

                # delete the word from the tweet
                tweet_line = np.delete(tweet_line, np.where(tweet_line == word))


        # update the counter
        count_test.update(tweet_line)

        # add the array containing all thw words of a certain tweet to our all tweets array
        tweets_data.append(tweet_line)



    return tweets_data,count_test
def occurence(count_test_word,n):
    # get the total number of different words in the tweets
    nb_elem_word = len(count_test_word)

    # get the occurences of each words in the tweets
    word_occurences = count_test_word.most_common(nb_elem_word-1)

    # don't take the words that are repeated less than 5 times
    word_occurences = [word_occurences[i] for i in range(len(word_occurences)) if word_occurences[i][1] > n-1]

    # get the total words number of neg tweets that has 5 or more occurences
    total_words = sum([word_occurences[i][1] for i in range(len(word_occurences))])

    word_weighted = word_occurences.copy()

    # get a probability of occurences (or weight) of each word in a positive sentences
    for word in range(len(word_weighted)) :
        word_weighted[word] = (word_weighted[word][0] ,word_weighted[word][1]/total_words)

    #pos_weighted

    return word_occurences, word_weighted
