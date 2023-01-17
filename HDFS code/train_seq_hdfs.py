# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 11:47:46 2023

@author: hardh
"""

import numpy as np
import pandas as pd
import time
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


split = 0.8

# Path
train_path = './data/log_train.csv'
    
logs_series = pd.read_csv(train_path)
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

x_train = log_seq[:split_boundary,:]
x_test = log_seq[split_boundary:,:]
y_train = label[:split_boundary]
y_test = label[split_boundary:]
    
y_train = np.asarray(y_train).astype(np.int32)
y_test = np.asarray(y_test).astype(np.int32)

#Training

total_log_keys = 29

x_train = x_train.reshape((split_boundary, max_seq_len, 1))
x_test = x_test.reshape((total_log_count - split_boundary, max_seq_len, 1))


model = Sequential()
model.add(LSTM(64, input_shape=(max_seq_len, 1), return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(16))
model.add(Dense(1, activation='sigmoid'))

adam = Adam(lr=0.01)
#sgd = SGD(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, epochs=10, verbose=1)  

#Plot Model Accuracy

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def plot_graphs(history, string):
    figure(figsize=(8, 6), dpi=80)
    plt.plot(np.arange(1,11), history.history[string], marker="x", linewidth=3, \
             markersize=10)
    plt.xlabel("Epochs", fontsize=20)
    plt.ylabel(string, fontsize=20)
    plt.xlim(1,10)
    plt.xticks(np.arange(1,11, step=1), size=15)
    plt.yticks(size=15)
    plt.grid()
    plt.show()
    
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

#Test Prediction

y_pred = model.predict(x_test)
y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])
test_acc = np.sum(y_pred == y_test) / len(y_test)
print("Test accuracy: ", test_acc)

from sklearn.metrics import f1_score
print("F1 score: ", f1_score(y_test, y_pred))

# pred no quant
def convert_to_tflite_noquant(model, filename):
    # Convert the tensorflow model into a tflite file.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    # Save the model.
    with open(filename, 'wb') as f:
        f.write(tflite_model)

model_tflite_filename = "model_no_quant.tflite"
convert_to_tflite_noquant(model, model_tflite_filename)


interpreter_noquant = tf.lite.Interpreter("model_no_quant.tflite")
interpreter_noquant.allocate_tensors()

y_pred_noquant = []

def tflite_predict(interpreter, data):
    input_data = data.reshape((1, max_seq_len, 1)).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

start_time = time.time()
for i in tqdm(range(x_test.shape[0])):
    x_test_sample = x_test[i]
    pred = tflite_predict(interpreter_noquant, x_test_sample)
    y_pred_noquant.append(pred[0][0])

print("---Pred time (noquant):  %s seconds ---" % (time.time() - start_time))
    
y_pred_noquant = np.array([1 if x > 0.5 else 0 for x in y_pred_noquant])
print("TPU accuracy (noquant): ", 100 * np.sum(y_pred_noquant == y_test) / len(y_pred_noquant), "%")
print("F1 score (noquant): ", f1_score(y_test, y_pred_noquant))

# pred hybrid quant
def convert_to_tflite_hybridquant(model, filename):
    # Convert the tensorflow model into a tflite file.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()

    # Save the model.
    with open(filename, 'wb') as f:
        f.write(tflite_model)

model_tflite_filename = "model_hybrid_quant.tflite"
convert_to_tflite_hybridquant(model, model_tflite_filename)


interpreter_hybridquant = tf.lite.Interpreter("model_hybrid_quant.tflite")
interpreter_hybridquant.allocate_tensors()

y_pred_hybrid = []

def tflite_predict(interpreter, data):
    input_data = data.reshape((1, max_seq_len, 1)).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

start_time = time.time()
for i in tqdm(range(x_test.shape[0])):
    x_test_sample = x_test[i]
    pred = tflite_predict(interpreter_hybridquant, x_test_sample)
    y_pred_hybrid.append(pred[0][0])

print("---Pred time (hybrid):  %s seconds ---" % (time.time() - start_time))
    
y_pred_hybrid = np.array([1 if x > 0.5 else 0 for x in y_pred_hybrid])
print("TPU accuracy (hybrid): ", 100 * np.sum(y_pred_hybrid == y_test) / len(y_pred_hybrid), "%")
print("F1 score (hybrid): ", f1_score(y_test, y_pred_hybrid))

#pred int quant
def representative_dataset():
  # To ensure full coverage of possible inputs, we use the whole train set
  for i in range(x_train.shape[0]):
    input_data = x_train[i, :, :].reshape((1, max_seq_len, 1))
    input_data = tf.dtypes.cast(input_data, tf.float32)
    yield [input_data]

batch_size = 1
model.input.set_shape((batch_size,) + model.input.shape[1:])

def convert_to_tflite_int(model, filename):
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # This sets the representative dataset for quantization
    converter.representative_dataset = representative_dataset
    
    # This ensures that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity
    converter.target_spec.supported_types = [tf.int8]
    """
    # These set the input and output tensors to int8
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    """
    tflite_model = converter.convert()

    # Save the model.
    with open(filename, 'wb') as f:
        f.write(tflite_model)

model_tflite_filename = "model_int_quant.tflite"
convert_to_tflite_int(model, model_tflite_filename)

#Pred hybrid
interpreter_int = tf.lite.Interpreter("model_int_quant.tflite")
interpreter_int.allocate_tensors()

y_pred_int = []

def tflite_predict(interpreter, data):
    input_data = data.reshape((1, max_seq_len, 1)).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

start_time = time.time()
for i in tqdm(range(x_test.shape[0])):
    x_test_sample = x_test[i]
    pred = tflite_predict(interpreter_noquant, x_test_sample)
    y_pred_int.append(pred[0][0])

print("---Pred time (int quant):  %s seconds ---" % (time.time() - start_time))
    
y_pred_int = np.array([1 if x > 0.5 else 0 for x in y_pred_int])
print("TPU accuracy (int quant): ", 100 * np.sum(y_pred_int == y_test) / len(y_pred_int), "%")
print("F1 score (int quant): ", f1_score(y_test, y_pred_int))
