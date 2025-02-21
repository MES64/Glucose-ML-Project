# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:03:59 2020

@author: marks

Run this file only when all the test files have been run, since the predictions
of all three models will be needed
"""

time_pnts = list(range(0, col_range, spacing))

plt.figure(num=None, figsize=(20,28))
for i in range(1,9):
    plt.subplot(4,2,i)
    plt.plot(time_pnts[-ph_indx:], y_test_hat[i-1])
    plt.plot(time_pnts[-ph_indx:], y_test_hat_partial[i-1])
    plt.plot(time_pnts[-ph_indx:], y1_test_hat[i-1])
    plt.plot(time_pnts[-ph_indx:], glucose_test[i-1][-ph_indx:])
    #plt.ylim([4.0, 18.0])
    plt.xlabel('Timestep')
    plt.ylabel('Glucose Level')
    plt.legend(['LSTM - insulin', 'LSTM - no insulin', 'CRNN', 'label'])
    
time_pnts = list(range(0, col_range, spacing))

plt.figure(num=None, figsize=(20,28))
for i in range(1,9):
    plt.subplot(4,2,i)
    plt.plot(time_pnts[-ph_indx:], y_test_hat[i-1])
    plt.plot(time_pnts[-ph_indx:], y_test_hat_partial[i-1])
    plt.plot(time_pnts[-ph_indx:], y1_test_hat[i-1])
    plt.plot(time_pnts, glucose_test[i-1])
    plt.ylim([4.0, 18.0])
    plt.xlabel('Timestep')
    plt.ylabel('Glucose Level')
    plt.legend(['LSTM - insulin', 'LSTM - no insulin', 'CRNN', 'label'])