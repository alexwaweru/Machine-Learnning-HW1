import numpy as np
from q4_features import q4_features
import scipy.io as spio
def q4_train(X, Y, lambdaval, mode):

# Trains the regularized least squares regression model using the closed form 
# solution given the training data X, Y.
#
# INPUT:
#  X: a numpy.ndarray matrix of size [m x d] and type 'float' where each row 
#     is a d-dimensional input example
#  Y: a numpy.ndarray vector of size [m x 1] and type 'float', where the 
#     i-th element is the correct output value for the i-th input example. 
#  lambda: 'float' regularization hyperparameter
#  mode: specifies the type of features; 
#        it is a 'str' that can be either 'linear' or 'quadratic'.
#
# OUTPUT:
#  theta: a numpy.ndarray vector of size [n x 1] and type 'float'
#         containing the learned model parameters.
#

    #get B
    B = q4_features(X, mode)

    #get B.BT
    BTB= np.dot(B.T, B)

    #get BT.Y
    BTY = np.dot(B.T, Y)

    #create U
    dimension = max(BTB.shape)
    U = np.eye(dimension)
    U[0,0] = 0

    #Solve system of linear equation
    theta = np.linalg.solve((BTB+lambdaval*(U)), (BTY))
    theta = np.reshape(theta, (theta.shape[0], 1)) 

    return theta
