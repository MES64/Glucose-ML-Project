# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:39:44 2020

@author: marks
"""

#=== CRNN Model ==============================================================#

# 1D temporal convolution, with 8, 16, 32, ... filters; kernel size is always 3; padding at both ends st. the size in = size out; input shape = 16 time steps, 3 features/channels
model_1=Sequential()
model_1.add(Conv1D(filters = 8, kernel_size = 3, padding = 'same', activation = 'relu', input_shape=(12,3)))
model_1.add(MaxPooling1D(pool_size = 2, strides = None, padding = 'same'))

model_1.add(Conv1D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu'))
model_1.add(MaxPooling1D(pool_size = 2, strides = None, padding = 'same'))

# Could try to reintroduce the original data here, on top of the features extracted above

model_1.add(LSTM(10))
model_1.add(Dense(ph_indx))

model_1.compile(loss = 'mean_squared_error', 
                optimizer = 'rmsprop', 
                metrics = [tf.keras.metrics.RootMeanSquaredError()])

history_1 = model_1.fit(X_train, y_train, batch_size = 64, epochs = 200, validation_data = (X_val, y_val))

#=== Save Model ==============================================================#

# Model to JSON
model_json = model_1.to_json()
with open("model_crnn.json", "w") as json_file:
    json_file.write(model_json)

# Weights to HDF5
model_1.save_weights("model_crnn.h5")