"""
@Author = Ilias Chatzistefanidis
Date = 23 September 2022

This file includes all the functions of the notebook for easier usage.
"""
import os
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from random import random
from random import randint
from numpy import array
from numpy import zeros
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
import datetime
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import Activation
from keras.layers import RepeatVector
from keras.layers import Bidirectional
from keras.layers import GRU

from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import math
from sklearn.metrics import mean_squared_error
import datetime

def split_sequence(sequence, n_steps, pred_steps, mean_batch):
    """
    This function applies a filtering technique to reduce the data volume and 
    then applies the sliding window technique to create a supervised learning structure.
    
    Regarding the filtering, the algorithm filters the sequence by calculating the average
    value of every m values. E.g. the average of every 5 values. In this way, the sequence's
    size is reduced maintaing the data patterns. 
    
    Regarding the sliding window, the algorithm calculates input windows X_i and the respective
    y_i labels to be fed into the model. Each input window is slided by one value to the future
    to create multiple samples for the model. Each X has length L. Each y is the label and 
    represent the forecasting of the model. Each y is a single value that represents the average 
    of multiple values. The number of these value that are used to calculate each y is p.
    Importantly, these values of the y label are locating after the respective X values to
    represent the future.

    Params:

    - sequence: input sequence
    - n_steps (L): input window time-steps/length   (length of X)
          The number of values that will be used to form the input window of the model
    - pred_steps (p): number of time-steps used for output/label(y)
          The number of values that will be used to calculate an average value (mean).
          This average value will be the y label. 
            e.g. if pred_steps equals 5, the mean of 5 values will be the y label
    - mean_batch (m): filtering window length
          The number of values that will be used by the filtering window

    An example for better understanding is the following:

    Assume the normalized inputs:
    sequence = [0.93, 0.93, 0.80, 0.80, 0.67, 0.67, 0.60, 0.60, 0.50, 0.50, 0.40, 0.40]
    n_steps = 3
    pred_steps = 2
    mean_batch = 2

    Then the filtered sequence after applying the filtering with mean_batch = 2 is:
    filtered_seq = [0.93, 0.80, 0.67, 0.60, 0.50, 0.40]

    Then samples are created using the sliding window with n_steps = 3, pred_steps = 2. 

    X_1 = [0.93, 0.80, 0.67]
    and y_1 = the average of 0.60 and 0.50.
    Hence, y_1 = [0.55]

    Then, we slide by on value and
    X_2 = [0.80, 0.67, 0.60] and y_2 = [0.45] (mean of 0.50 and 0.40)
    """
    new_sequence = []

    ### Filtering
    temp_sum = 0
    # iterate through sequence
    for i,item in enumerate(sequence,start=1):
        temp_sum+=item
        # for every m values calculate the mean value and 
        # append it in the new sequence
        if i%mean_batch == 0:
            mean_temp = temp_sum/mean_batch
            temp_sum=0
            new_sequence.append(mean_temp)

    # work with the new filtered sequence
    sequence = new_sequence

    # adjust the params to the new sequence
    n_steps = int(n_steps/mean_batch)
    pred_steps = int(pred_steps/mean_batch)

    ### Sliding window technique
    X, y = list(), list()
    # iterate through sequence
    for i in range(len(sequence)):
        # for each iteration (i),
        # find the end of this pattern (X_i)
        end_ix = i + n_steps
        
        # check if X_i is beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # check if values for y_i go beyond the sequence
        pred_ix = end_ix + pred_steps
        if pred_ix > len(sequence)-1:
            break
            
        # compute the y label (mean of p values after X_i)
        mean_pred = np.mean(sequence[end_ix:pred_ix])
        
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], mean_pred

        # store sample
        X.append(seq_x)
        y.append(seq_y)
        
    return np.array(X), np.array(y)


def cross_validation_from_scratch(X, y, init_window=0, prediction_window=500):
    """
    This function applies the Time-series Cross Validation tecnique to efficiently 
    evaluate a model's predictive performance on multiple unseen data.

    Params:
    - X: Input X with samples
    - y: Input y with samples' labels
    - init_window: The initial offset of the training set
          This variable determines how many samples will be inserted in the training set
          before the beginning of the execution.
    - prediction_window: Number of samples on prediction/ length of folds
    """
    # init training sets
    X_splits = []
    y_splits = []

    # init validation sets
    X_pred_splits = []
    y_pred_splits = []

    # find the size of the data we need to split into folds
    # we split only the data after the initial offset
    size_to_split = X.shape[0] - init_window

    # determine num of splits/ number of folds
    n_splits = math.ceil(size_to_split/prediction_window)

    # iterate in each fold
    for i in range(n_splits):
        # compute X,y

        # X of training set
        X_split = X[: init_window + i * prediction_window ]
        # X of validation set
        X_pred_split = X[ init_window + i*prediction_window : init_window + (i+1)*prediction_window ]

        # y of training set
        y_split = y[: init_window + i * prediction_window ]
        # y of validation set
        y_pred_split = y[ init_window + i*prediction_window : init_window + (i+1)*prediction_window ]

        # store them
        X_splits.append(X_split)
        y_splits.append(y_split)

        X_pred_splits.append(X_pred_split)
        y_pred_splits.append(y_pred_split)

    # test that everything is ok
    if  len(X_splits) == len(y_splits) == len(X_pred_splits) == len(y_pred_splits):
        print("[INFO]: Num of Folds:",len(X_splits))
    else:
        print('[ERROR]: Error occured!')
        return

    return X_splits, y_splits, X_pred_splits, y_pred_splits


def collect_predictions(model, X_test, y_test, n_steps, mean_batch, n_features, scaler, verbose=0):
    """
    This function utilizes the pretrained model and collects the predictions on 
    the desired test set.

    Params:
    - model: the variable of the pre-trained model
    - X_test: the X samples of the test set
    - y_test: the y labels of the X samples
    - n_steps, mean_batch, n_features, scaler: variables utilized earlier in the notebook (see notebook)
    - verbose: the verbose parameter to be used by the predict() function of the model
    """
    
    # init 2 sets for predictions and real values (labels)
    all_preds_test = []
    all_real_test = []

    # iterate through the validation samples
    for i in range(X_test.shape[0]):
        # define model's input window
        x_input = X_test[i]

        # reshape appropriately
        x_input = x_input.reshape((1, int(n_steps/mean_batch), n_features))

        # predict
        yhat = model.predict(x_input, verbose=0)

        # inverse transform of the normalized scale back to the original scale of
        # CQI data [0,15]    
        y_pred = yhat
        y_real = y_test[i]
        y_pred_inversed = scaler.inverse_transform(y_pred)
        y_real_inversed = scaler.inverse_transform(np.array(y_real).reshape(-1,1))

        # store predictions and real values (labels)
        all_preds_test.append(y_pred_inversed)
        all_real_test.append(y_real_inversed)

    return all_preds_test,all_real_test


def validate_model(epochs, X_train, y_train, X_test, y_test,init_w=40500,pred_w=500):
    """
    This function utilizes two more funtions:
        - cross_validation_from_scratch()
        - collect_predictions()
    It is designed to evaluate the model using two steps:
        1) Apply the Time-Series Cross Validation technique to the dataset
        2) Train on the complete dataset and validate on the experimental data

    Params:

    - epochs: The epochs to train the model
    - X_train, y_train: The samples to be used in the first step.
    - X_test, y_test: The samples to be used in the seconds step.
    - init_w: The offset of the time-series cross validation technique.
    - pred_w: The length of the folds in the time-series cross validation technique.
    """
    # Create Folds
    print("[INFO]: Split Dataset for Time-series CV")
    X_splits, y_splits, X_pred_splits, y_pred_splits = cross_validation_from_scratch(X_train,y_train,init_window=init_w, prediction_window=pred_w)

    # init lists for metrics 
    MAE_hist = []
    RMSE_hist = []
    time_in_folds = []

    # counter of for loop
    counter=0   

    # utilize the folds to apply time-series cross validation
    print("[INFO]: Begin Time-series CV")
    for X_temp, y_temp, X_pred_temp, y_pred_temp in zip(X_splits, y_splits, X_pred_splits, y_pred_splits):
        counter+=1
        print("[INFO]: Fold ",counter)

        # define model
        model = keras.Sequential()
        model.add(Bidirectional(LSTM(25, activation='relu',return_sequences=True), input_shape=(int(n_steps/mean_batch), n_features)    ))
        model.add(Bidirectional(LSTM(25, activation='relu') ))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # start counting training time
        a = datetime.datetime.now()

        # fit model
        print("[INFO]: Train Model on Training Set")
        model.fit(X_temp, y_temp, epochs=epochs, verbose=1, batch_size=2**6)

        # calculate training time
        b = datetime.datetime.now()
        diff = b-a
        diff_secs = diff.total_seconds()

        # store training time
        time_in_folds.append(diff_secs)

        # make predictions 
        print("[INFO]: Collect Predictions on Validation Set",counter)
        all_preds_test,all_real_test = collect_predictions(model, X_pred_temp, y_pred_temp)

        # reshape for evaluation
        all_preds_test = np.array(all_preds_test).reshape(-1)
        all_real_test = np.array(all_real_test).reshape(-1)

        # evaluate
        MAE_current = mean_absolute_error(all_preds_test, all_real_test)
        RMSE_current = mean_squared_error(all_preds_test, all_real_test)

        # save stats
        MAE_hist.append(MAE_current)
        RMSE_hist.append(RMSE_current)

        print("[RESULTS]: ",counter,"/",len(X_splits)," Time:",round(diff_secs,1), "MAE_current:",round(MAE_current,2), "RMSE_current",round(RMSE_current,2)) 
        print()

    print("[INFO]: End of Time-Series CV")
    print("[INFO]: Train on whole training set and evaluate on experimental real data (excluded from training set)")
    
    # define model
    model = keras.Sequential()
    model.add(Bidirectional(LSTM(25, activation='relu',return_sequences=True), input_shape=(int(n_steps/mean_batch), n_features)    ))
    model.add(Bidirectional(LSTM(25, activation='relu') ))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # start counting training time
    a = datetime.datetime.now()

    # fit model
    print("[INFO]: Train Model on the whole training dataset")
    model.fit(X_train, y_train, epochs=epochs, verbose=1, batch_size=2**6)

    # calculate training time
    b = datetime.datetime.now()
    diff = b-a
    diff_secs = diff.total_seconds()

    # make predictions
    print("[INFO]: Collect Predictions on experimental data")
    all_preds_test,all_real_test = collect_predictions(model, X_test, y_test)

    # reshape for evaluation
    all_preds_test = np.array(all_preds_test).reshape(-1)
    all_real_test = np.array(all_real_test).reshape(-1)

    # plot forecasting vs actual values
    print("[INFO]: Plot the forecasting on experimental data:")
    plt.figure(figsize=[10,5])
    plt.plot(all_preds_test,'orange',label="yhat")
    plt.plot(all_real_test,'blue',label="y")
    plt.ylim([0,15])
    plt.legend()
    plt.show()

    # evaluate with MAE and RMSE metrics
    MAE_real_data = mean_absolute_error(all_preds_test, all_real_test)
    RMSE_real_data = mean_squared_error(all_preds_test, all_real_test)

    # print results
    print("[RESULTS]: Overall Fold MAE:   ",round(np.mean(MAE_hist),2),'(mean)  ',round(np.median(MAE_hist),2),'(median)' )
    print("[RESULTS]: Overall Fold RMSE:  ",round(np.mean(RMSE_hist),2),'(mean)  ',round(np.median(RMSE_hist),2),'(median)'  )
    print("[RESULTS]: Mean Time in Folds:",round(np.mean(time_in_folds) ,2))
    print("[RESULTS]: Tesing MAE (real data): ",round(MAE_real_data,2))
    print("[RESULTS]: Tesing RMSE (real data):",round(RMSE_real_data,2))
    print("[RESULTS]: Time in real data:",round(diff_secs,2))

    return MAE_hist, RMSE_hist, MAE_real_data, RMSE_real_data