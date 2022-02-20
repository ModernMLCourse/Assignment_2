"""
EE468: Neural Networks and Deep Learning class.
Second assignment-- Classification
===============================================
Instructor notes:
-----------------
This script is meant as a guid for students to learn Python and do
Assignment 2 of the course. It has more missing lines than those of
previous assignment, so that students can be more independent when
developing a code

Please note the following:
1) Data is ALREADY RANDOMLY SHUFFLED
2) DO NOT change the structure of the code. Just fill in the
missing line
3) FILL IN the missing lines in both "main.py" and "utils.py"
4) The script must type out the training record for each model.
5) Print validation loss of both models.
6) Make sure the visualization section works. It must produce
four figures.

Happy coding... ;-)
============================
Author: Muhammad Alrabeiah
Date: Feb. 2022
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# This is how to include Python functions from a separate file
from utils import extract_feat, log_reg, trn_algrm, derivative, cross_entropy


def main(): # Your main code should go here
    fig_c = 0 # Initialize figure counter
    # Prepare training and validation data
    data_path =
    D = sio.loadmat(data_path)
    X = D['c']
    T = D['labels']
    X_trn = # TODO Read 600 training samples
    T_trn = T[0,:600].astype(np.float32) - 1 # Labels need to be 0 or 1
    X_val = # TODO Read 200 val samples
    T_val = # TODO Read 200 val samples

    # Feature Extraction------------------
    # hyper-params for 1st basis
    mu_1 = np.array([[1],[1]])
    var_1 = 2
    # hyper-params for 2nd basis
    mu_2 = np.array([[5],[5]])
    var_2 = 2

    # Transform training data
    h1 = # TODO use "extract_feat" function to get first transformed dimension (use 1st basis)
    h2 = # TODO use "extract_feat" function to get second transformed dimension (use 2ndd basis)
    H_trn = # TODO concatenate h1 and h2 in one matrix of dimension (2, 600)

    # Transform validation data
    h1 = # TODO similar to above but for validation data
    h2 = # TODO similar to above but for validation data
    H_val = # TODO concatenate h1 and h2 in one matrix of dimension (2, 200)

    # Model Training --------------------
    # Log. Reg Model
    w_init = np.random.randn(2,) # Initialize weights
    pred = log_reg # Define the model --> Read about assigning a python function to a variable
    D = # TODO Define the derivative function
    loss = # TODO Define the loss function

    # 1) Train on original data
    dsg_mtx_1 = # TODO form design  matrix using original data in X_trn
    params = np.concatenate((w_init, np.ones((1,)))) # Adding one for intercept (bais) parameter
    w_star1 = trn_algrm(X=dsg_mtx_1,
                        T=T_trn,
                        model=pred,
                        params=params,
                        criterion=loss,
                        deri=D,
                        num_itr=10000)

    # 2) Train on transformed data
    dsg_mtx_2 = # TODO form design matrix using transformed data in H_trn
    params = # TODO prepare parameter vector --> param above
    w_star2 = # TODO train your second model


    # Test Trained Model --------------------

    # 1) Test model 1 on original data
    dsg_mtx_val_1 = # TODO design matrix using X_val
    pred, _ = # TODO get predictions from "log_reg". NOTES: 1) use learned parameters and 2) "_" means unwanted value
    val_loss_1 = # TODO measure validation loss

    # 2) Test model 2 on transformed data
    dsg_mtx_val_2 = # TODO design matrix using X_val
    pred, _ = # TODO get predictions from "log_reg".
    val_loss_2 = # TODO measure validation loss
    print('Model 1 validation loss {0:}, and Model 2 validation loss {1:}'.format(val_loss_1, val_loss_2))

    # Visualizations
    # 1) visualize original data
    fig_c += 1
    plt.figure(fig_c)
    plt.scatter(X_trn[0, :], X_trn[1, :])
    plt.grid()

    # 2) visualize transformed data
    fig_c += 1
    plt.figure(fig_c)
    plt.scatter(H_trn[0,:], H_trn[1,:])
    plt.grid()

    # 3) visualize classification results on original data
    pred, _ = log_reg(w_star1, dsg_mtx_1)  # Predicted labels
    class_1 = X_trn[:, np.around(pred) == 0]  # Round down probabilities < 0.5
    class_2 = X_trn[:, np.around(pred) == 1]  # Round up probabilities > 0.5
    fig_c += 1
    plt.figure(fig_c)
    plt.scatter(class_1[0, :], class_1[1, :], label='class 1')
    plt.scatter(class_2[0, :], class_2[1, :], label='class 2')
    plt.grid()
    plt.legend()

    # 4) visualize classification results on transformed data
    pred, _ = log_reg(w_star2, dsg_mtx_2) # Predicted labels
    class_1 = X_trn[:, np.around(pred) == 0] # Round down probabilities < 0.5
    class_2 = X_trn[:, np.around(pred) == 1] # Round up probabilities > 0.5
    fig_c += 1
    plt.figure(fig_c)
    plt.scatter(class_1[0, :], class_1[1, :],label='class 1')
    plt.scatter(class_2[0, :], class_2[1, :],label='class 2')
    plt.grid()
    plt.legend()
    plt.show()



if __name__ == '__main__': # Read about this "if" statment. It is important when you want your script readable from another script
    main()

