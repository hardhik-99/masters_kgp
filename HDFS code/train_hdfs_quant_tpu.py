# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:42:07 2023

@author: hardh
"""

import json
import time

import numpy as np
import pandas as pd
import time
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, Adagrad

#Load TFlite model
import tflite_runtime.interpreter as tflite

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
low_sem_vec[0, :] = 0

def read_data(path, split = 0.85):
    
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
    x_test = logs[split_boundary:,:,:]
    y_train = label[:split_boundary]
    y_test = label[split_boundary:]
    return x_train, y_train, x_test, y_test, max_seq_len
    
# Path
train_path = './data/log_train.csv'
# Training data and valid data
x_train, y_train, x_test, y_test, max_seq_len = read_data(train_path)

y_train = np.asarray(y_train).astype(np.int32)
y_test = np.asarray(y_test).astype(np.int32)

def load_tflite_model(modelpath):
    interpreter = tflite.Interpreter(model_path=modelpath,
                                     experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()
    return interpreter

# pred no quant
def convert_to_tflite_noquant(model, filename):
    # Convert the tensorflow model into a tflite file.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    # Save the model.
    with open(filename, 'wb') as f:
        f.write(tflite_model)

model_tflite_filename = "model_no_quant.tflite"
interpreter_noquant = load_tflite_model(model_tflite_filename)
interpreter_noquant.allocate_tensors()

y_pred = []

def tflite_predict(interpreter, data):
    input_data = data.reshape((1, max_seq_len, low_dim_len)).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

start_time = time.time()
for i in tqdm(range(x_test.shape[0])):
    x_test_sample = x_test[i]
    pred = tflite_predict(interpreter_noquant, x_test_sample)
    y_pred.append(pred[0][0])

print("---Pred time (noquant):  %s seconds ---" % (time.time() - start_time))
    
y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])
print("TPU accuracy (noquant): ", 100 * np.sum(y_pred == y_test) / len(y_pred), "%")
print("F1 score (noquant): ", f1_score(y_test, y_pred))

# pred hybrid quant

model_tflite_filename = "model_hybrid_quant.tflite"
interpreter_hybridquant = load_tflite_model(model_tflite_filename)
interpreter_hybridquant.allocate_tensors()

y_pred = []

def tflite_predict(interpreter, data):
    input_data = data.reshape((1, max_seq_len, low_dim_len)).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

start_time = time.time()
for i in tqdm(range(x_test.shape[0])):
    x_test_sample = x_test[i]
    pred = tflite_predict(interpreter_hybridquant, x_test_sample)
    y_pred.append(pred[0][0])

print("---Pred time (hybrid):  %s seconds ---" % (time.time() - start_time))
    
y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])
print("TPU accuracy (hybrid): ", 100 * np.sum(y_pred == y_test) / len(y_pred), "%")
print("F1 score (hybrid): ", f1_score(y_test, y_pred))

# pred int quant

model_tflite_filename = "model_int_quant.tflite"
interpreter_int = load_tflite_model(model_tflite_filename)
interpreter_int.allocate_tensors()

y_pred = []

def tflite_predict(interpreter, data):
    input_data = data.reshape((1, max_seq_len, low_dim_len)).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

start_time = time.time()
for i in tqdm(range(x_test.shape[0])):
    x_test_sample = x_test[i]
    pred = tflite_predict(interpreter_int, x_test_sample)
    y_pred.append(pred[0][0])

print("---Pred time (int quant):  %s seconds ---" % (time.time() - start_time))
    
y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])
print("TPU accuracy (int quant): ", 100 * np.sum(y_pred == y_test) / len(y_pred), "%")
print("F1 score (int quant): ", f1_score(y_test, y_pred))