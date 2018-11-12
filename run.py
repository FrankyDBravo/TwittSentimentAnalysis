import csv
from keras.models import model_from_json
from keras.models import Model, Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D
from keras.preprocessing import sequence
import numpy as np
import pickle



X_test=np.load('tweets_emb_nltk_conv_test.npy')
sequence_length=56
X2_test= sequence.pad_sequences(X_test,maxlen=sequence_length)

def load_model():
    # loading model
    model = model_from_json(open('model_architecture.json').read())
    model.load_weights('model_weights_76.h5')
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model




def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


model=load_model()
y_pred = model.predict(X2_test)
y_rendu=[]
for i in range(len(y_pred)):
    if y_pred[i]>= 0.5:
        y_rendu.append(1)
    else: y_rendu.append(-1)

OUTPUT_PATH = 'prediction.csv'
ids_test=[i+1 for i in range(len(y_rendu))]
create_csv_submission(ids_test, y_rendu, OUTPUT_PATH)
