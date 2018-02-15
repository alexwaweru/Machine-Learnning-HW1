import numpy as np
import scipy.io as spio

def q4_mse(pred_Y, correct_Y):
# This function calculates the Mean Squared Error given two sets of output
# values, one set corresponding to the correct values, the other set
# representing the output values predicted by a regression model
# INPUT:
#  pred_Y: a numpy.ndarray vector of type 'float' containing m predicted values
#  correct_Y: a numpy.ndarray vector of type 'float' containing m correct values
#
# OUTPUT:
#  err: 'float' representing the Mean Squared Error
#

      sum_sqrd_err = 0
      M = min((correct_Y.shape)[0], (pred_Y.shape)[0])
      
      for i in range(M):
            sum_sqrd_err += (correct_Y[i] - pred_Y[i])**2

      err = sum_sqrd_err/M
      
      return err




