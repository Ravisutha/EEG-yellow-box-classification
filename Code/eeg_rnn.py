# Recurrent Neural Network

from __future__ import print_function
import sys
import pydot
sys.path.append('/home/ravi/Class/Thesis/Data/Code')
import reading_edf as data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.utils import plot_model
#RMSE
import math
from sklearn.metrics import mean_squared_error
# Command Line Arguments
import argparse
# Load modle
from keras.models import load_model
import pickle
# Random
import random

#-----------------------------
# Part 1 - Data Pre-processing
#-----------------------------

# Importing the training set
def import_training_data (idea=1, num_samples=3):
    """
    Give the training data based on which idea you are implementing.
    Input:
        idea: Idea number
        Idea #1:
            - Train with normal signal and observe what happens when you show "yellow-boxes" to RNN.
        Idea #2:
            - You know when yellow boxes start and end. See when you give yellow box input, it suggests the presence of "abnormal activity"
    Output:
        Matrix of training
    """

    a = data.edf_convert()
    if (idea == 1):
        M = np.zeros ((num_samples, 100))
        for i in range (num_samples):
            montage = a.file[i][3]
            M[i] = data.edf_convert ().get_normal_signal (num_samples=1, sample_num=i, truncate=True, montage=montage, random=True)

        training_set = np.array (M).reshape (-1, 1)
        points = -1

    elif (idea == 2):
        M = np.zeros ((num_samples*2, 100))
        s = []
        points = np.zeros ((num_samples * 2, 1))
        for i in range (num_samples):
            montage = a.file[i][3]
            i = i * 2
            M[i] = data.edf_convert ().get_normal_signal (num_samples=1, sample_num=i, truncate=True, montage=montage, random=True)
            points[i, 0] = 1
            s.append ((M[i], points[i, 0]))
            (M[i+1], details) = data.edf_convert ().get_yellow_signals (num_samples=1, sample_num=i, truncate=True, random=False)
            points[i+1, 0] = -1
            s.append ((M[i+1], points[i+1, 0]))

        random.shuffle (s)

        for x in s:
            M[i] = x[0]
            points[i, 0] = x[1]

        #(M, details) = data.edf_convert ().get_yellow_signals (num_samples=1)
        training_set = np.array (M).reshape (-1, 1)
        print (points)
        #points = (float(details[0][1]), float(details[0][2]))

    return (training_set, points)

# Feature Scaling
def scale_data (training_set):
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform (training_set)
    return (sc, training_set_scaled)

#Creating a datastructure with 60 timesteps (window) and 1 output
def split_data (training_set_scaled, points=-1, idea=1):
    if idea == 1:
        n_timesteps = 30
    elif idea == 2:
        n_timesteps = 100

    n_output = 1
    upper_bound = training_set_scaled.shape[0]
    X_train = []
    Y_train = []

    for i in range (n_timesteps, upper_bound):
      init  = i - n_timesteps
      X_train.append (training_set_scaled[init:i, 0])
      if idea == 1:
          Y_train.append (training_set_scaled[i, 0])
      elif idea == 2:
          if (init % 100 == 0):
              Y_train.append (1)
          else:
              Y_train.append (-1)

    X_train, Y_train = np.array (X_train), np.array (Y_train)

    #Reshaping, keras needs 3D tensor: (batch_size, timesteps, inputdim)
    X_train = X_train.reshape (X_train.shape[0], X_train.shape[1], 1)

    return (X_train, Y_train)

#-----------------------------
#Part 2 - Building the RNN
#-----------------------------

# Initializing the RNN
def RNN(X_train, n_output=1):
    regressor = Sequential ()

    #Adding the first LSTM layer with Dropout regularisation
    regressor.add (LSTM (units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add (Dropout (0.2)) 

    #Adding the second LSTM layer with Dropout regularisation
    regressor.add (LSTM (units=100, return_sequences=True))
    regressor.add (Dropout (0.2)) 

    #Adding the third LSTM layer with Dropout regularisation
    regressor.add (LSTM (units=100, return_sequences=True))
    regressor.add (Dropout (0.2)) 

    #Adding the fourth LSTM layer with Dropout regularisation
    regressor.add (LSTM (units=100))
    regressor.add (Dropout (0.2)) 

    #Ouput Layer
    regressor.add (Dense (n_output))

    return regressor

#Compiling the keras neural network
def compile(regressor):
    regressor.compile (optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
def train (regressor, X_train, Y_train):
    regressor.fit (X_train, Y_train, epochs=30, batch_size=32)

#-----------------------------------------------------------
#Part 3 - Making the predictions and visualising the results
#-----------------------------------------------------------

def import_test_data (sc, num, n_timesteps=100, idea=1):
    if idea==1:
        # Get EEG data
        if (num < 0):
            M = data.edf_convert ().get_normal_signal (num_samples=1, sample_num=-num, random=True, truncate=True)
            points = -1
        else:
            (M, details) = data.edf_convert ().get_yellow_signals (num_samples=1, sample_num=num, random=False, truncate=True)
            print ("Start of abnormal activity: {}".format (details[0][1]))
            print ("End of abnormal activity: {}".format (details[0][2]))
            print ("Montage: {}".format (details[0][3]))
            points = (float (details[0][1]), float (details[0][2]))
    elif idea==2:
        if (num < 0):
            M = data.edf_convert ().get_normal_signal (num_samples=1, sample_num=-num, random=True, truncate=True)
            points = -1
        else:
            (M, details) = data.edf_convert ().get_yellow_signals (num_samples=1, sample_num=num, random=True, truncate=True)
            print ("Start of abnormal activity: {}".format (details[0][1]))
            print ("End of abnormal activity: {}".format (details[0][2]))
            print ("Montage: {}".format (details[0][3]))
            points = (float (details[0][1]), float (details[0][2]))

    test_set = np.array(M).reshape (-1, 1)

    inputs = test_set

    # Scaled inputs and outputs
    inputs = sc.transform (inputs)

    upper_bound = inputs.shape[0]
    X_test = []
    Y_test = []

    for i in range (n_timesteps, upper_bound):
      init  = i - n_timesteps
      print (init)
      print (i)
      print (upper_bound)
      X_test.append (inputs[init:i, 0])
      if idea == 1:
          Y_test.append (inputs[i, 0])
      elif idea == 2:
          if num < 0:
              Y_test.append (1)
          else:
              Y_test.append (-1)

    if n_timesteps == upper_bound:
        X_test.append (inputs[0:100, 0])
        if num < 0:
          Y_test.append (1)
        else:
          Y_test.append (-1)

    X_test = np.array(X_test)
    print (X_test.shape[0])
    print (X_test.shape[1])
    X_test = X_test.reshape (X_test.shape[0], X_test.shape[1], 1)

    Y_test = np.array(Y_test).reshape(-1, 1)
    if idea == 1:
        Y_test = sc.inverse_transform (Y_test)

    return (X_test, Y_test, points)

# Predict
def predict (sc, regressor, X_test, idea=1):
    predicted_values = regressor.predict (X_test)
    if (idea == 1):
        predicted_values = sc.inverse_transform (predicted_values) # Inverse scaling
    return (predicted_values)

#-----------------------------------------------------------------------
# Part  4: Other useful features
#-----------------------------------------------------------------------

# Visualising the results
def visualize (test_set, predicted_values, signal_type, points=-1, idea=1, save=False):

    plt.figure ()
    if (idea == 2):
        plt.subplot (211)
    plt.plot (test_set, color='red', label='Real EEG signal')
    if (idea == 2):
        plt.subplot (212)
    plt.plot (predicted_values, color='blue', label='Fitted EEG signal')

    # Plot a yellow box
    if (points != -1):
        start = points[0]
        end = points[1]
        #start = int (float (start) * 250)
        #end = int (float (end) * 250)
        start = 0
        end = 100

        m1 = max (test_set[:, 0])
        m2 = min (test_set[:, 0])

        x2 = np.array ([start, start, end, end, start])
        y2 = np.array ([m1, m2, m2, m1, m1])

        #plt.plot (x2, y2, color="yellow")

    if (signal_type < 0):
        signal_type = "normal_" + str (-signal_type)
    else:
        signal_type = "yelow_" + str (signal_type)

    plt.title ('Real vs Predicted: ' + signal_type)
    plt.xlabel ('Time')
    plt.ylabel ('EEG signal')
    plt.legend()

    if save:
        plt.savefig ("/home/ravi/Class/ANN/Takehome2/Output/Temp4/compare_idea_" + str(idea) + "_" + signal_type + ".pdf", bbox_inches="tight")

    #plt.show()

# RMSE (Root Mean Squared Error)
def compute_rmse (test_set, predicted_values):
    rmse = math.sqrt (mean_squared_error (test_set, predicted_values))
    return rmse

# Analyze
def analyze (Y_test, predicted_values, signal_type, n_timesteps=60, points=-1, save=False, plot=True, idea=1):
    if idea == 1:
        upper_bound = Y_test.shape[0]
        r = []

        n_timesteps = 30
        for i in range (0, upper_bound, n_timesteps):
            r.append (compute_rmse (Y_test[i:i+n_timesteps, 0], predicted_values[i:i+n_timesteps, 0]))

        if plot:
            plt.title ('RMSE')
            x =  np.arange(len(r))
            plt.figure ()
            plt.plot (x, r, label="RMSE")

            # Plot the box
            if (points != -1):
                start = float (0 * 250 / n_timesteps)
                end = float ( 100 / n_timesteps)
                m1 = max (r)
                m2 = min (r)
                x2 = np.array ([start, start, end, end, start])
                y2 = np.array ([m1, m2, m2, m1, m1])

            plt.xlabel ('Time in seconds')
            plt.ylabel ('RMSE')
            plt.legend ()

            if save:
                if (signal_type < 0):
                    signal_type = "normal_" + str (-signal_type)
                else:
                    #plt.show()
                    signal_type = "yelow_" + str (signal_type)
                plt.savefig ("/home/ravi/Class/ANN/Takehome2/Output/Temp3/analysis_idea_" + signal_type + ".pdf", bbox_inches="tight")

        return (sum (r)/len(r))
    else:
        return (sum(Y_test))

def compute_confusion_matrix (pos_confusion_matrix, neg_confusion_matrix, idea=1):
    if idea == 1:
        neg_max = max (neg_confusion_matrix)

        false_positive = 0;
        for c in pos_confusion_matrix:
            if c < neg_max:
                false_positive += 1

        true_negative = len (pos_confusion_matrix) - false_positive

        print ("False Positive\t|\tTrue Negative")
        print ("{}            \t|\t{}".format (false_positive, true_negative))

        neg_max = sum (neg_confusion_matrix) // len (neg_confusion_matrix)

        false_positive = 0;
        for c in pos_confusion_matrix:
            if c <= neg_max:
                false_positive += 1

        true_negative = len (pos_confusion_matrix) - false_positive

        print ("False Positive\t|\tTrue Negative")
        print ("{}            \t|\t{}".format (false_positive, true_negative))

    else:
        true_negative = 0
        true_positive = 0
        false_negative = 0
        false_positive = 0
        for c in pos_confusion_matrix:
            if c > 0:
                false_positive += 1
            else:
                true_negative += 1

        for c in neg_confusion_matrix:
            if c < 0:
                false_negative += 1
            else:
                true_positive += 1

# Save model
def my_save_model (model):
    model.save('my_model5.h5')  # creates a HDF5 file 'my_model.h5'

# Load model
def my_load_model ():
    model = load_model('my_model5.h5')
    return (model)

# Save weights
def my_save_weights (model):
    model.save_weights('my_weights5.h5')

# Load weights
def my_load_weights (model):
    model.load_weights('my_weights5.h5')

#-------------------------------------------------
# Part 5: Call the functions
#-------------------------------------------------

# Main function
def train_model (idea=1, load=False, save=False, num_samples=3):

    if (load == True):
        model = my_load_model ()

        # Train
        my_load_weights (model)

        # Load sc
        sc = pickle.load (open ("sc.p", "rb"))

    else:
        # Get data
        (train_data, points) = import_training_data (idea=idea, num_samples=num_samples)

        # Scale data
        sc, train_data_scaled = scale_data (train_data)

        # Split data into input and output signals
        (X_train, Y_train) = split_data (train_data_scaled, idea=idea, points=points)

        # Define
        model = RNN (X_train)

        # Compile
        compile (model)

        # Train
        train (model, X_train, Y_train)

    # Save the trained model
    if (save == True):
        my_save_model (model)
        my_save_weights (model)
        pickle.dump (sc, open ("sc.p", "wb"))

    return (model, sc)

def test_model (model, sc, idea=1, save=False, plot=False):
    # Use dark background
    plt.style.use('dark_background')

    # Signal Types
    signal_type = list (range (-80, 80))

    pos_confusion_matrix = []
    neg_confusion_matrix = []

    for i in signal_type:
        # Get test data
        (X_test, Y_test, points) = import_test_data (sc, i, idea=idea)

        # Predict
        predicted_values = predict (sc, model, X_test, idea=idea)

        # Visualize
        if plot:
            visualize (Y_test, predicted_values, i, points=points, idea=idea, save=save)

        # Analyze
        max_rmse = analyze (Y_test, predicted_values, i, points=points, save=save, idea=idea)
        
        if (i < 0):
            neg_confusion_matrix.append (max_rmse)
        else:
            pos_confusion_matrix.append (max_rmse)

    print ("______________________________________________")
    print ("----------------------------------------------")
    print ("Yellow box signal output: (Expecting all -1s)")
    print (pos_confusion_matrix) # Expecting all -1
    print ("______________________________________________")
    print ("----------------------------------------------")
    print ("Normal signal output: (Expecting all +1s)")
    print (neg_confusion_matrix) # Expecting all 1
    print ("______________________________________________")
    print ("----------------------------------------------")

    # Find out true negatives and false positives
    compute_confusion_matrix (pos_confusion_matrix, neg_confusion_matrix, idea=idea)

if __name__ == "__main__":

    # Train the model
    #(model, sc) = train_model (idea=2, load=False, save=True, num_samples=100)
    (model, sc) = train_model (idea=2, load=True, save=False, num_samples=1)

    #plot_model(model, to_file='model.png', show_shapes=True)

    # Test it
    test_model (model, sc, idea=2, save=False, plot=True)
