import numpy as np
import math
from q4_train import q4_train
from q4_predict import q4_predict
from q4_mse import q4_mse


def q4_cross_validation_error(X, Y, lambdavec, mode, N):
# Calculates the cross-validation errors for different values of lambdavec, given the
# training set X, Y.
#
# ** Implementation notes **
# - As discussed in class, you should first randomly permute the examples, before starting the
#   cross-validation stage. Here we did it for you: we created Xr and Yr which are obtained from
#   X and Y by permuting examples. You should use Xr and Yr in your code (not X and Y)
# - Do not change/initialize/reset the Python pseudo-number generator.
#
# INPUT
#  X: a numpy.ndarray matrix of size [m x d] and type 'float' where each row 
#     is a d-dimensional input example
#  Y: a numpy.ndarray vector of size [m x 1] and type 'float', where the 
#     i-th element is the correct output value for the i-th input example. 
#  lambdavec: a numpy.ndarray vector of size [k x 1] and type 'float'
#             containing the set of regularization hyperparameter values
#  mode: specifies the type of features; 
#        it is a 'str' that can be either 'linear' or 'quadratic'.
#  N: `int' representing the number of folds for the cross-validation stage
#
# OUTPUT
#  error: a numpy.ndarray vector of size [k x 1] and type 'float'
#         containing the cross-validation error (i.e., the average of the mean 
#         squared errors over the N validation sets) for each value in lambdavec.
#


# ********  DO NOT TOUCH THE FOLLOWING 5 LINES  ********************
    np.random.seed(0)
    m = X.shape[0]
    idxperm = np.random.permutation(m)-1
    Xr = X[idxperm,:]
    Yr = Y[idxperm]
# ******************************************************************
   
    error = []
    subarrays_of_X = np.split(Xr, N)
    subarrays_of_Y =  np.split(Yr, N)
    
    for i in range(0, lambdavec.shape[0]):
        lambda_ = lambdavec[i]
        errors_N = []
        for n in range(0, N):
            # Create test and train sets
            X_test_set = subarrays_of_X[n]
            Y_test_set = subarrays_of_Y[n]
            X_training_set = list(subarrays_of_X)
            Y_training_set = list(subarrays_of_Y)

            #Delete the test sets from the training sets
            del X_training_set[n]
            del Y_training_set[n]
            
            training_X = []
            training_X = X_training_set[0]
            for i in range(0, len(X_training_set)-1):
                   training_X = np.concatenate((training_X, X_training_set[i+1]), axis=0) 
            training_Y = np.array(Y_training_set).ravel()

            # Calculating theta, predicted Y and the mean squared error.
            theta = q4_train(training_X, training_Y, lambda_, mode)
            predicted_Y = q4_predict(theta, X_test_set, mode)
            err = q4_mse(predicted_Y, Y_test_set)
            errors_N.append(err)
            
        error.append(np.mean(errors_N))
           
    return error
