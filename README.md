- run the file run.py to get the result obtained on kaggle.

- We have four auxilliary python .py : 
            - twitter_processing.py : preprocess the given text files
            - coocme.py : compute the coocurence matrix 
            - glove_emb.py : create the vectors for word representation
            - dataXY.py : transform our data so we can use it for neural networks (CNN and DNN)
            
- We also have .npy files which stores the arrays from the methods we run so you can access directly test our data on models without recompiling everything
	-DataCreation: create the X and y for CNN and other classifier (2 different data matrix)
	-CNN : convolutional 
	-Other classifier: logistic , Kernel SVM, dense neural network 
	(Load the vector representation according to your desire)

WARNING:
———
Due to size limitation we were unable to load our X and y preparation (for Glove and W2v respectively) 
The code provided will create them but with a different name that the previous one used. CF data creation for the name of the output
——-

-DEPENDENCIES for python3:

TENSORFLOW
KERAS
SKLEARN
PANDAS
NLTK
GENSIM