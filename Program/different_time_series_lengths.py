# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:35:38 2020

@author: marks
"""

# Initialize
timesteps_list = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210]  # Including ph
spacing = 10
train_sample_num = 10000
test_sample_num = 5000
ph = 30

ph_indx = int(np.ceil(ph/spacing))  # Use ceil to predict at least ph into the future

# Loop through each of the different number of timesteps
rmse_list = []
for col_range in timesteps_list:
  # Preprocess Data
  X_train_ts, X_test_ts, y_train_ts, y_test_ts = dataPreprocess(col_range, spacing, train_sample_num, test_sample_num, ph)
  
  # Create and Train Model
  model_ts = Sequential()
  model_ts.add(LSTM(10))
  model_ts.add(Dense(ph_indx))
  
  model_ts.compile(loss = 'mean_squared_error', 
                   optimizer = 'rmsprop', 
                   metrics = [tf.keras.metrics.RootMeanSquaredError()])
  
  model_ts.fit(X_train_ts, y_train_ts, batch_size = 64, epochs = 200, verbose = 0)
  
  # Evaluate the model (RMSE) for each number of timesteps used
  mse_rmse = model_ts.evaluate(X_test_ts, y_test_ts)
  rmse_list.append(mse_rmse[1])


# Plot
plt.plot(np.array(timesteps_list) - ph, rmse_list, '-x')
plt.xlabel('Input Timesteps')
plt.ylabel('RMSE')