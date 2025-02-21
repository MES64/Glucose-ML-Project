# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 20:00:34 2020

@author: marks
"""

#=== LSTM Model ==============================================================#

model = Sequential()
model.add(LSTM(10))
model.add(Dense(ph_indx))

model.compile(loss = 'mean_squared_error', 
              optimizer = 'rmsprop', 
              metrics = [tf.keras.metrics.RootMeanSquaredError()])

history = model.fit(X_train, y_train, batch_size = 64, epochs = 200, validation_data = (X_val, y_val))

#=== Save Model ==============================================================#

# Model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Weights to HDF5
model.save_weights("model.h5")