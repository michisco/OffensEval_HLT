from keras.layers import Dropout, Dense, Embedding, LSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from tensorflow.keras import regularizers
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import shuffle
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
import warnings
import logging
logging.basicConfig(level=logging.INFO)

def prepare_model_test(X, MAX_NB_WORDS=75000,MAX_SEQUENCE_LENGTH=140):
    '''Prepare test data to feed model.'''
    np.random.seed(7)
    text = np.array(X)
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index
    text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    print('Found %s unique tokens.' % len(word_index))
    indices = np.arange(text.shape[0])
    text = text[indices]
    X_Glove = text
    
    return (X_Glove, word_index)

def prepare_model_dev(X_train, X_val,MAX_NB_WORDS=75000,MAX_SEQUENCE_LENGTH=140):
    '''Prepare development data to feed model.'''
    np.random.seed(7)
    text = np.concatenate((X_train, X_val), axis=0)
    text = np.array(text)
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index
    text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Found %s unique tokens.' % len(word_index))
    indices = np.arange(text.shape[0])

    text = text[indices]
    X_train = text[0:len(X_train), ]
    X_val = text[len(X_train):, ]
    
    return (X_train, X_val, word_index)

def prepare_embeddings(path_file = 'drive/MyDrive/HLT/GloVe/', glove_file = 'glove.twitter.27B.100d.txt'):
    '''Prepare embeddings using GloVe.'''
    np.random.seed(7)
    embeddings_dict = {}
    f = open(path_file+glove_file, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            pass
        embeddings_dict[word] = coefs
    f.close()
    print('Total %s word vectors.' % len(embeddings_dict))
    return embeddings_dict

def BiLSTM(word_index, embeddings_dict, hparams = None):
    '''Create bidirectional LSTM model.'''
    model = Sequential()
    # Make the embedding matrix using the embedding_dict
    embedding_matrix = np.random.random((len(word_index) + 1, hparams['embedding_dim']))
    for word, i in word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) != len(embedding_vector):
                print("could not broadcast input array from shape", str(len(embedding_matrix[i])),
                      "into shape", str(len(embedding_vector)), " Please make sure your"
                                                                " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
            embedding_matrix[i] = embedding_vector
            
    # Add embedding layer
    model.add(Embedding(len(word_index) + 1,
                                hparams['embedding_dim'],
                                weights=[embedding_matrix],
                                input_length=hparams['max_sequence_length'],
                                trainable=True))
                                
    # Add hidden layers 
    for i in range(0,hparams['num_hidden_layer']):
        # Add a bidirectional lstm layer
        model.add(Bidirectional(LSTM(hparams['num_lstm_nodes'], return_sequences=True, recurrent_dropout=0.2)))
        # Add a dropout layer after each lstm layer
        model.add(Dropout(hparams['dropout']))
        
    model.add(Bidirectional(LSTM(hparams['num_lstm_nodes'], recurrent_dropout=0.2)))
    model.add(Dropout(hparams['dropout']))
    
    # Add the fully connected layer with relu activation
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(hparams['L2_reg'])))
   
    # Add the output layer with softmax
    model.add(Dense(2, activation='softmax'))
    
    # Compile the model using sparse_categorical_crossentropy
    model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=hparams['optimizer'],
                      metrics=['accuracy'])
    return model