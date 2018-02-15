import numpy as np
    

def q4_features(X, mode):
# Given the data matrix X (where each row X[i,:] is an example), the function
# computes the feature matrix B, where row B[i,:] represents the feature vector 
# associated to example X[i,:]. The features should be either linear or quadratic
# functions of the inputs, depending on the value of the input argument 'mode'.
# Please make sure to implement the features according to the *exact* order
# specified in the text of the homework assignment.
#
# INPUT:
#  X: a numpy.ndarray matrix of size [m x d] and type 'float' where each row 
#     is a d-dimensional input example
#  mode: specifies the type of features; 
#        it is a 'str' that can be either 'linear' or 'quadratic'.
#
# OUTPUT:
#  B: a numpy.ndarray matrix of size [m x n] and type 'float', with each row 
#     containing the feature vector of an example

    B = []
    if mode == 'linear':
        B = np.insert(X, 0, 1, axis= 1)
        
    elif mode == 'quadratic':
        r = []
        array_ = []
        for  i, x in enumerate(X):
            r = [1]
            r.extend(x)
            for k in range(len(x)):
                for j in range(len(x)):
                    r.append(x[j] * x[k])
            array_.append(r)
        B = np.array(array_)
                
        
    else:
        print('Error, only linear and quadratic forms are supported');
        return []

    return B

    
    

