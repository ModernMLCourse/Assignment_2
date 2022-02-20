"""
Utility functions
PLEASE read function notes carefully
==========================
Author: Muhammad Alrabeiah
Date: Feb. 2022
"""
import numpy as np


def extract_feat(X,
                 mu=np.zeros((2,1)),
                 var = 1
                 ):
    '''
    A function extracting engineered features from data matrix.
    It implements a Gaussian basis function
    :param X: Data matrix with dimensions (N,U) --> N is # of observed variables, U is # of samples
    :param mu_1: center for basis function with dimensions (2,1)
    :param var_1: spread of basis function (scalar)
    :return Y: A feature matrix
    '''
    # TODO implement Gaussian basis function
    h = # Basis function
    return h

def log_reg( params, H):
    '''
    Defining the logistic regression model
    :param params: parameter vector with dimensions (N+1,1) --> see "extract_feat" for N
    :param H: Design matrix (N+1,U) --> for dimensions see "extract_feat"
    :return y: predictions in the range (0,1)
    '''
    # TODO implement logistic regression model
    v = # Linear model
    y = # Sigmoid using v
    return [y,v]

def cross_entropy(pred, T):
    '''
    Implementation of cross entropy loss (Not the best way to implement it)
    Note please that this function hard-codes "log(0)=0"
    ARGUMENTS--------------
    :param pred: predictions with dimensions (U,)
    :param T: labels with dimesnions (U,)
    OUTPUTS----------------
    :return L: loss value
    '''
    beh_idx = np.logical_and( pred > 0.001, pred <= 0.99 ) # Exclude values that could cause the case of "log(0)"
    beh_pred = pred[beh_idx]
    T = T[beh_idx]
    # Computing cross entropy
    l1 = T*np.log(beh_pred)
    l2 = (1-T)*np.log((1-beh_pred))
    L = -1*np.sum(l1+l2)/pred.shape[0]
    return L

def derivative(X, y, t):
    '''
    Derivative of loss function w.r.t. parameters
    ARGUMENTS--------------
    :param X: design matrix with dimensions (N+1, U)
    :param y: predictions with dimensions (U,)
    :param t: labels with dimensions (U,)
    OUTPUTS----------------
    :return D: derivative
    '''
    # TODO implement derivative of loss
    D = # Derivative of cros entropy
    return D

def trn_algrm(X,
              T,
              model,
              params,
              criterion,
              deri,
              num_itr,
              ):
    '''
    A script implementing a training algorithm for a classification model
    using gradient descent
    ARGUMENTS---------------
    :param X: design matrix with dimensions (N+1, U) --> see first function above
    :param T: labels with dimensions (U,)
    :param model: model to train
    :param params: parameter vector of the model with dimensions (N+1,1)
    :param criterion: definition of the loss function
    :param deri: definition of the derivative of the loss function
    :param num_itr: total number of training iterations
    OUTPUTS------------------
    :return params: trained parameter vector
    '''
    count = 0 # A counter for the while loop
    lr = 1 # Learning rate
    flag = 0 # A flag to break the while loop
    max_itr = num_itr # Max number of while loop iteration (very good practice to guarantee no infinite loops!)
    thrshold = 0.005 # Threshold to stop training
    print('--------- Commence training ----------')
    while (flag == 0 and count < max_itr):
        count += 1
        [y, logits] = # TODO Pass data and get predictions
        loss = criterion(y,T) # Measure error
        print('Itr {0:}, lr {1:}, and Loss {2:}'.format(count, lr, loss))
        if loss < thrshold: # Check if target loss is reached
            flag  = 1
        d = # TODO Compute derivative
        # Update weights
        params = params - (lr * d) # Implementing gradient descent update rule

    return params








