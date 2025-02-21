# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 08:52:55 2020

@author: marks
"""

#=== LSTM Model ==============================================================#

model_partial = Sequential()
model_partial.add(LSTM(10))
model_partial.add(Dense(ph_indx))

model_partial.compile(loss = 'mean_squared_error', 
                      optimizer = 'rmsprop', 
                      metrics = [tf.keras.metrics.RootMeanSquaredError()])

history = model_partial.fit(X_train_partial, y_train, batch_size = 64, epochs = 200, validation_data = (X_val_partial, y_val))

#=== Save Model ==============================================================#

# Model to JSON
model_json = model_partial.to_json()
with open("model_partial.json", "w") as json_file:
    json_file.write(model_json)

# Weights to HDF5
model_partial.save_weights("model_partial.h5")