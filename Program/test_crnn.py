# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:51:36 2020

@author: marks
"""

# Load in model
with open("model_crnn.json", "r") as json_file:
    model_json = json_file.read()
model_1 = model_from_json(model_json)
model_1.load_weights("model_crnn.h5")

model_1.compile(loss = 'mean_squared_error', 
                optimizer = 'rmsprop', 
                metrics = [tf.keras.metrics.RootMeanSquaredError()])

y1_test_hat = model_1.predict(X_test)
print(y1_test_hat[:12])
print(y_test[:12])

col_range = 150
time_pnts = list(range(0, col_range, spacing))
print(time_pnts)

plt.figure(num=None, figsize=(20,28))
for i in range(1,9):
    plt.subplot(4,2,i)
    plt.plot(time_pnts[-ph_indx:], y1_test_hat[i-1])
    plt.plot(time_pnts, glucose_test[i-1])
    plt.ylim([4.0, 18.0])
    plt.xlabel('Timestep')
    plt.ylabel('Glucose Level')
    plt.legend(['prediction', 'label'])

model_1.evaluate(X_test, y_test)