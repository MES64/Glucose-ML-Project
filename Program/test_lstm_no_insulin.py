# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:00:40 2020

@author: marks
"""

# Load in model
with open("model_partial.json", "r") as json_file:
    model_json = json_file.read()
model_partial = model_from_json(model_json)
model_partial.load_weights("model_partial.h5")

model_partial.compile(loss = 'mean_squared_error', 
              optimizer = 'rmsprop', 
              metrics = [tf.keras.metrics.RootMeanSquaredError()])

y_test_hat_partial = model_partial.predict(X_test_partial)
print(y_test_hat_partial[:12])
print(y_test[:12])

time_pnts = list(range(0, col_range, spacing))
print(time_pnts)

plt.figure(num=None, figsize=(20,28))
for i in range(1,9):
    plt.subplot(4,2,i)
    plt.plot(time_pnts[-ph_indx:], y_test_hat_partial[i-1])
    plt.plot(time_pnts, glucose_test[i-1])
    plt.ylim([4.0, 18.0])
    plt.xlabel('Timestep')
    plt.ylabel('Glucose Level')
    plt.legend(['prediction', 'label'])

model_partial.evaluate(X_test_partial, y_test)