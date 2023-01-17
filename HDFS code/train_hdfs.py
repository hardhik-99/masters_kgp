# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:42:07 2023

@author: hardh
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tqdm import tqdm

with open('./data/hdfs_semantic_vec.json') as f:
    # Step-1 open file
    gdp_list = json.load(f)
    high_sem_vec = list(gdp_list.values())
    
    # Step-2 PCA: Dimensionality reduction to 20-dimensional data
    from sklearn.decomposition import PCA
    estimator = PCA(n_components=20)
    pca_result = estimator.fit_transform(high_sem_vec)

    # Step-3 PPA: De-averaged
    ppa_result = []
    result = pca_result - np.mean(pca_result)
    pca = PCA(n_components=20)
    pca_result = pca.fit_transform(result)
    U = pca.components_
    for i, x in enumerate(result):
        for u in U[0:7]:
            x = x - np.dot(u.transpose(), x) * u
        ppa_result.append(list(x))
    low_sem_vec = np.array(ppa_result)

low_dim_len = 20

def read_data(path, split = 0.8):
    
    logs_series = pd.read_csv(path)
    logs_series = logs_series.values
    label = logs_series[:,1]
    logs_data = logs_series[:,0]
    logs_data, label = shuffle(logs_data, label)
    for i in range(len(logs_data)):
        logs_data[i] = [x for x in logs_data[i].split(' ')]
    max_seq_len = max([len(x) for x in logs_data])
    log_seq = np.array(pad_sequences(logs_data, maxlen=max_seq_len, padding='pre'))
    log_seq = np.asarray(log_seq)
    
    total_log_count = logs_data.shape[0]
    split_boundary = int(total_log_count * split)
    
    logs = np.zeros((total_log_count, max_seq_len, low_dim_len))
    
    for i in range(total_log_count):
        for j in range(max_seq_len):
            logs[i, j, :] = low_sem_vec[log_seq[i, j]] 
    
    x_train = logs[:split_boundary,:,:]
    x_valid = logs[split_boundary:,:,:]
    y_train = label[:split_boundary]
    y_valid = label[split_boundary:]
    return x_train, y_train, x_valid, y_valid, max_seq_len
    
# Path
train_path = './data/log_train.csv'
# Training data and valid data
x_train, y_train, x_valid, y_valid, max_seq_len = read_data(train_path)

y_train = np.asarray(y_train).astype(np.int32)
y_valid = np.asarray(y_valid).astype(np.int32)

model = Sequential()
model.add(LSTM(64, input_shape=(max_seq_len, low_dim_len), return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(16))
model.add(Dense(1, activation='sigmoid'))

#adam = Adam(lr=0.01)
sgd = SGD(lr=0.02)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, epochs=10, \
                    validation_data=(x_valid, y_valid), verbose=1)

#Plot Model Accuracy

import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()
    
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

#Test Prediction
test_path = './data/log_test_2000.csv'
logs_series = pd.read_csv(test_path)
logs_series = logs_series.values
label_test = logs_series[:,1]
logs_data = logs_series[:,0]
logs_data, label = shuffle(logs_data, label_test)
for i in range(len(logs_data)):
    logs_data[i] = [x for x in logs_data[i].split(' ')]
max_seq_len = 269
log_seq = np.array(pad_sequences(logs_data, maxlen=max_seq_len, padding='pre'))
log_seq = np.asarray(log_seq)

total_log_count = logs_data.shape[0]

logs_test = np.zeros((total_log_count, max_seq_len, low_dim_len))
    
for i in range(total_log_count):
    for j in range(max_seq_len):
        logs_test[i, j, :] = low_sem_vec[log_seq[i, j]] 

x_test = logs_test
y_test = label_test
y_test = np.asarray(y_test).astype(np.int32)

y_pred = model.predict(x_test)
y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])
test_acc = np.sum(y_pred == y_test) / len(y_test)
print("Test accuracy: ", test_acc)

from sklearn.metrics import f1_score
print("F1 score: ", f1_score(y_test, y_pred))

