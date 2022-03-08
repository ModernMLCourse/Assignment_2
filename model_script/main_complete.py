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

Happy coding... ;-)
============================
Author: Muhammad Alrabeiah
Date: Feb. 2022
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from utils_complete import extract_feat, log_reg, trn_algrm, derivative, cross_entropy


def main(): # Your main code should go here
    fig_c = 0
    # Prepare training and validation data
    data_path = './data/dataset.mat'
    D = sio.loadmat(data_path)
    X = D['c']
    T = D['labels']
    X_trn = X[:,:600]
    T_trn = T[0,:600].astype(np.float32) - 1 # Labels need to be 0 or 1
    X_val = X[:,600:]
    T_val = T[0,600:].astype(np.float32) - 1

    # Feature Extraction------------------
    mu_1 = np.array([[1],[1]])
    var_1 = 2
    mu_2 = np.array([[5],[5]])
    var_2 = 2
    h1 = extract_feat(X_trn, mu_1, var_1)
    h2 = extract_feat(X_trn, mu_2, var_2)
    H_trn = np.concatenate((h1.reshape(1,h1.shape[0]),h2.reshape(1,h2.shape[0])),axis=0) # Training feature matrix

    h1 = extract_feat(X_val, mu_1, var_1)
    h2 = extract_feat(X_val, mu_2, var_2)
    H_val = np.concatenate((h1.reshape(1, h1.shape[0]), h2.reshape(1, h2.shape[0])), axis=0) # Validatioon feature matrix

    # Model Training --------------------
    # Log. Reg Model
    w_init = np.random.randn(2,) # Initialize weights
    pred = log_reg # Define the model
    D = derivative # Define the derivative function
    loss = cross_entropy # Define the loss function

    # 1) Train on original data
    dsg_mtx_1 = np.concatenate((X_trn, np.ones((1, X_trn.shape[1]))), axis=0)
    params = np.concatenate((w_init, np.ones((1,))))
    w_star1 = trn_algrm(X=dsg_mtx_1,
                        T=T_trn,
                        model=pred,
                        params=params,
                        criterion=loss,
                        deri=D,
                        num_itr=10000)

    # 2) Train on transformed data
    dsg_mtx_2 = np.concatenate((H_trn,np.ones((1,H_trn.shape[1]))), axis=0)
    params = np.concatenate((w_init, np.ones((1,))))
    w_star2 = trn_algrm(X=dsg_mtx_2,
                       T=T_trn,
                       model=pred,
                       params=params,
                       criterion=loss,
                       deri=D,
                       num_itr=10000)


    # Test Trained Model --------------------

    # 1) Test model 1
    dsg_mtx_val_1 = np.concatenate((X_val, np.ones((1, X_val.shape[1]))), axis=0)
    pred, _ = log_reg(w_star1, dsg_mtx_val_1)
    val_loss_1 = loss(pred, T_val)

    # 2) Test model 2
    dsg_mtx_val_2 = np.concatenate((H_val,np.ones((1,H_val.shape[1]))), axis=0)
    pred, _ = log_reg(w_star2, dsg_mtx_val_2)
    val_loss_2 = loss(pred, T_val)
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

