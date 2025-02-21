# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 08:12:09 2020

@author: marks

Run this python file first, to create the functions and preprocess the data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D

from tensorflow.keras.models import model_from_json

import matplotlib.pyplot as plt

# Fix random seed for reproducibility
np.random.seed(1209)

# Function to take data samples
def takeDataSamples(data_in, cols, rows, sample_num, col_range, spacing):
  """
  Function to sample from the input data to create output data within the 
  specified column range (col_range) and spacing, with sample_num samples taken. 
  
  Input:
  data_in = input data
  cols = the pre-sampled columns
  rows = the pre-sampled rows
  sample_num = the number of sample observations to be taken
  col_range = the range of columns sampled from the input data
  spacing = only include samples at every 'nth' column in the output data

  Output:
  data_out = output (sampled) data
  """
  
  data_out = np.zeros([sample_num, int(np.ceil(col_range/spacing))])

  for i in range(sample_num):
    col_ind_sampled = np.arange(cols[i], cols[i]+col_range, spacing)
    data_out[i] = data_in[rows[i], col_ind_sampled]
  
  return data_out

# Function to preprocess the data (without creating a validation set)
def dataPreprocess(col_range, spacing, train_sample_num, test_sample_num, ph):
  # 1: Take Data Samples
  
  # Training Samples
  num_rows, num_cols = np.shape(glucose_train_long)
  train_cols = np.random.randint(0, num_cols-col_range, train_sample_num)
  train_rows = np.random.randint(0, num_rows, train_sample_num)

  glucose_train = takeDataSamples(glucose_train_long, train_cols, train_rows, train_sample_num, col_range, spacing)
  insulin_train = takeDataSamples(insulin_train_long, train_cols, train_rows, train_sample_num, col_range, spacing)
  carbs_train = takeDataSamples(carbs_train_long, train_cols, train_rows, train_sample_num, col_range, spacing)
  
  # Test Samples
  num_rows, num_cols = np.shape(glucose_test_long)
  test_cols = np.random.randint(0, num_cols-col_range, test_sample_num)
  test_rows = np.random.randint(0, num_rows, test_sample_num)
  
  glucose_test = takeDataSamples(glucose_test_long, test_cols, test_rows, test_sample_num, col_range, spacing)
  insulin_test = takeDataSamples(insulin_test_long, test_cols, test_rows, test_sample_num, col_range, spacing)
  carbs_test = takeDataSamples(carbs_test_long, test_cols, test_rows, test_sample_num, col_range, spacing)
  
  #----------------------------------------------------------------------------#
  # 2: Create Label
  
  ph_indx = int(np.ceil(ph/spacing))  # Use ceil to predict at least ph into the future

  # Training Set
  y_train = glucose_train[:,-ph_indx:]
  X_glucose_train_raw = glucose_train[:, :-ph_indx]
  X_insulin_train_raw = insulin_train[:, :-ph_indx]
  X_carbs_train_raw = carbs_train[:, :-ph_indx]

  # Test Set
  y_test = glucose_test[:, -ph_indx:]
  X_glucose_test_raw = glucose_test[:, :-ph_indx]
  X_insulin_test_raw = insulin_test[:, :-ph_indx]
  X_carbs_test_raw = carbs_test[:, :-ph_indx]
  
  #----------------------------------------------------------------------------#
  # 3: Normalize
  
  glucose_norm = StandardScaler()
  glucose_norm.fit(X_glucose_train_raw)
  X_glucose_train = glucose_norm.transform(X_glucose_train_raw)
  X_glucose_test = glucose_norm.transform(X_glucose_test_raw)
  
  insulin_norm = StandardScaler()
  insulin_norm.fit(X_insulin_train_raw)
  X_insulin_train = insulin_norm.transform(X_insulin_train_raw)
  X_insulin_test = insulin_norm.transform(X_insulin_test_raw)
  
  carbs_norm = MinMaxScaler()
  carbs_norm.fit(X_carbs_train_raw)
  X_carbs_train = carbs_norm.transform(X_carbs_train_raw)
  X_carbs_test = carbs_norm.transform(X_carbs_test_raw)
  
  #---------------------------------------------------------------------------#
  # 4: Concatenate
  
  # Concatenate the training set
  X_train = np.concatenate((tf.expand_dims(X_glucose_train, 2), tf.expand_dims(X_insulin_train, 2), tf.expand_dims(X_carbs_train, 2)), axis = 2)

  # Concatenate the test set
  X_test = np.concatenate((tf.expand_dims(X_glucose_test, 2), tf.expand_dims(X_insulin_test, 2), tf.expand_dims(X_carbs_test, 2)), axis = 2)
  
  return (X_train, X_test, y_train, y_test)

#=== Import Data =============================================================#

# Load in
glucose_train_df = pd.read_csv('./CGM_prediction_data-master/glucose_readings_train.csv', header = None)
glucose_test_df = pd.read_csv('./CGM_prediction_data-master/glucose_readings_test.csv', header = None)
insulin_train_df = pd.read_csv('./CGM_prediction_data-master/insulin_therapy_train.csv', header = None)
insulin_test_df = pd.read_csv('./CGM_prediction_data-master/insulin_therapy_test.csv', header = None)
carbs_train_df = pd.read_csv('./CGM_prediction_data-master/meals_carbs_train.csv', header = None)
carbs_test_df = pd.read_csv('./CGM_prediction_data-master/meals_carbs_test.csv', header = None)

# Convert to NumPy array
glucose_train_long = np.array(glucose_train_df)
glucose_test_long = np.array(glucose_test_df)
insulin_train_long = np.array(insulin_train_df)
insulin_test_long = np.array(insulin_test_df)
carbs_train_long = np.array(carbs_train_df)
carbs_test_long = np.array(carbs_test_df)

#=== Preprocess the data while creating a validation set =====================#

#=== Sample Data =============================================================#

# First, randomly sample data samples
# Pick a random row and column as the starting index of the sample and extract those data points (maybe only 2 hrs). Repeat 10,000 times. 

# Init
col_range = 150  # 2 hours, 30 minutes PH
spacing = 10


# Training Set:
train_sample_num = 10000
num_rows, num_cols = np.shape(glucose_train_long)

# Note: Could try sampling without replcement if can
train_cols = np.random.randint(0, num_cols-col_range, train_sample_num)
train_rows = np.random.randint(0, num_rows, train_sample_num)


# Test Set
test_sample_num = 5000
num_rows, num_cols = np.shape(glucose_test_long)

test_cols = np.random.randint(0, num_cols-col_range, test_sample_num)
test_rows = np.random.randint(0, num_rows, test_sample_num)

# Take Samples
glucose_train = takeDataSamples(glucose_train_long, train_cols, train_rows, train_sample_num, col_range, spacing)
insulin_train = takeDataSamples(insulin_train_long, train_cols, train_rows, train_sample_num, col_range, spacing)
carbs_train = takeDataSamples(carbs_train_long, train_cols, train_rows, train_sample_num, col_range, spacing)

glucose_test = takeDataSamples(glucose_test_long, test_cols, test_rows, test_sample_num, col_range, spacing)
insulin_test = takeDataSamples(insulin_test_long, test_cols, test_rows, test_sample_num, col_range, spacing)
carbs_test = takeDataSamples(carbs_test_long, test_cols, test_rows, test_sample_num, col_range, spacing)

#=== Split Dataset ===========================================================#

# Extract feature to predict, and split training set into training set proper and validation set

ph = 30  # prediction horizon
ph_indx = int(np.ceil(ph/spacing))  # Use ceil to predict at least ph into the future

# Training Set and Validation Set Split
valid_ratio = 0.3  # The ratio of the full training set to be split into the validation set
valid_size = int(valid_ratio*train_sample_num)

# Training Set
y_train = glucose_train[:-valid_size,-ph_indx:]
X_glucose_train_raw = glucose_train[:-valid_size, :-ph_indx]
X_insulin_train_raw = insulin_train[:-valid_size, :-ph_indx]
X_carbs_train_raw = carbs_train[:-valid_size, :-ph_indx]

# Validation Set
y_val = glucose_train[-valid_size:, -ph_indx:]
X_glucose_val_raw = glucose_train[-valid_size:, :-ph_indx]
X_insulin_val_raw = insulin_train[-valid_size:, :-ph_indx]
X_carbs_val_raw = carbs_train[-valid_size:, :-ph_indx]

# Test Set
y_test = glucose_test[:, -ph_indx:]
X_glucose_test_raw = glucose_test[:, :-ph_indx]
X_insulin_test_raw = insulin_test[:, :-ph_indx]
X_carbs_test_raw = carbs_test[:, :-ph_indx]

print(np.shape(y_train))
print(np.shape(X_glucose_train_raw))

#=== Normalize ===============================================================#

glucose_norm = StandardScaler()
glucose_norm.fit(X_glucose_train_raw)
X_glucose_train = glucose_norm.transform(X_glucose_train_raw)
X_glucose_val = glucose_norm.transform(X_glucose_val_raw)
X_glucose_test = glucose_norm.transform(X_glucose_test_raw)

insulin_norm = StandardScaler()
insulin_norm.fit(X_insulin_train_raw)
X_insulin_train = insulin_norm.transform(X_insulin_train_raw)
X_insulin_val = insulin_norm.transform(X_insulin_val_raw)
X_insulin_test = insulin_norm.transform(X_insulin_test_raw)

carbs_norm = MinMaxScaler()
carbs_norm.fit(X_carbs_train_raw)
X_carbs_train = carbs_norm.transform(X_carbs_train_raw)
X_carbs_val = carbs_norm.transform(X_carbs_val_raw)
X_carbs_test = carbs_norm.transform(X_carbs_test_raw)

#=== Concatenate the features of the data set ================================#

X_train = np.concatenate((tf.expand_dims(X_glucose_train, 2), tf.expand_dims(X_insulin_train, 2), tf.expand_dims(X_carbs_train, 2)), axis = 2)
print(np.shape(X_train))

# Concatenate the validation set
X_val = np.concatenate((tf.expand_dims(X_glucose_val, 2), tf.expand_dims(X_insulin_val, 2), tf.expand_dims(X_carbs_val, 2)), axis = 2)
print(np.shape(X_val))

# Concatenate the test set
X_test = np.concatenate((tf.expand_dims(X_glucose_test, 2), tf.expand_dims(X_insulin_test, 2), tf.expand_dims(X_carbs_test, 2)), axis = 2)
print(np.shape(X_test))


#=== Remove Insulin ==========================================================#

# Concatenate the training set
X_train_partial = np.concatenate((tf.expand_dims(X_glucose_train, 2), tf.expand_dims(X_carbs_train, 2)), axis = 2)
print(np.shape(X_train_partial))

# Concatenate the validation set
X_val_partial = np.concatenate((tf.expand_dims(X_glucose_val, 2), tf.expand_dims(X_carbs_val, 2)), axis = 2)
print(np.shape(X_val_partial))

# Concatenate the test set
X_test_partial = np.concatenate((tf.expand_dims(X_glucose_test, 2), tf.expand_dims(X_carbs_test, 2)), axis = 2)
print(np.shape(X_test_partial))
