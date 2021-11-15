import pickle

import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import fabs, sqrt
import matplotlib.pyplot as plt
import pandas as pnd
import time


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return (1.0 / (1.0 + np.exp(-z)))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples.
    # Your code here.
    test = []
    train = []
    for i in range(10):
        id_train = f'train{i}'
        id_test = f'test{i}'
        mat_test = mat[id_test]
        mat_train = mat[id_train]
        label_test = np.full((mat_test.shape[0], 1), i)
        label_train = np.full((mat_train.shape[0], 1), i)
        labelled_train = np.concatenate((mat_train, label_train), axis=1)
        labelled_test = np.concatenate((mat_test, label_test), axis=1)
        test.append(labelled_test)
        train.append(labelled_train)
    train_all = np.concatenate((train[0], train[1], train[2], train[3],
                                train[4], train[5], train[6], train[7], train[8], train[9]), axis=0)
    test_all = np.concatenate(
        (test[0], test[1], test[2], test[3], test[4],
         test[5], test[6], test[7], test[8], test[9]),
        axis=0)
    np.random.shuffle(train_all)
    train_final = train_all[0:50000,:]
    train_data = train_final[:, 0:784]
    train_label = train_final[:, 784]
    validation_final = train_all[50000:60000,:]
    validation_data = validation_final[:, 0:784]
    validation_label = validation_final[:, 784]
    test_data = test_all[:, 0:784]
    test_label = test_all[:, 784]
    test_data = test_data / 255.0
    validation_data = validation_data / 255.0
    train_data = train_data / 255.0
    # Feature selection
    # Your code here.
    all = np.concatenate((train_data, validation_data), axis=0)
    ref = all[0, :]
    redundant_vals = np.all(all == ref, axis=0)

    count = 0
    global selected_indicies
    for i in range(len(redundant_vals)):
        if redundant_vals[i] == False:
            count += 1
            selected_indicies.append(i)
            print(i, end=" ")
    print(" ")
    print(f"Total Selected Features-->{count}")

    final_all = all[:, ~redundant_vals]
    train_row = train_data.shape[0]

    train_data = final_all[0:train_row, :]
    validation_data = final_all[train_row:, :]
    test_data = test_data[:, ~redundant_vals]
    print('preprocess done')
    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    train_rows = training_data.shape[0]

    # Your code here
    # Input to Hidden Layer
    biases_1 = np.full((train_rows, 1), 1)
    training_data_with_biases = np.concatenate(
        (biases_1, training_data), axis=1)
    bj = np.dot(training_data_with_biases, np.transpose(w1))
    sigma_bj = sigmoid(bj)

    # Hidden to Output
    sigma_bj_rows = sigma_bj.shape[0]
    biases_2 = np.full((sigma_bj_rows, 1), 1)
    sigma_bj_with_biase = np.concatenate((biases_2, sigma_bj), axis=1)
    bz = np.dot(sigma_bj_with_biase, np.transpose(w2))
    sigma_bz = sigmoid(bz)

    # Error Calculation thru Error Function
    ground_truth = np.full((train_rows, n_class), 0)
    for i in range(train_rows):
        label = training_label[i]
        ground_truth[i][label] = 1

    ground_truth_prime = (1.0 - ground_truth)
    sigma_bz_prime = (1.0 - sigma_bz)
    lg_sigma_bz = np.log(sigma_bz)
    lg_sigma_bz_prime = np.log(sigma_bz_prime)

    err = np.sum(np.multiply(ground_truth, lg_sigma_bz) +
                 np.multiply(ground_truth_prime, lg_sigma_bz_prime))
    err = (err / ((-1)*train_rows))

    # Gradient Calculation for BP
    delta = sigma_bz - ground_truth
    grad_w2 = np.dot(delta.T, sigma_bj_with_biase)

    t1 = np.dot(delta,w2)
    t1 = t1 * (sigma_bj_with_biase*(1.0-sigma_bj_with_biase))

    grad_w1 = (np.dot(np.transpose(t1), training_data_with_biases))[1:, :]

    # Calculate Regularization

    reg_para = lambdaval * (np.sum(np.square(w1))+np.sum(np.square(w2)))/(2*train_rows)
    obj_val = err + reg_para

    grad_w1_reg = ((lambdaval*w1)+grad_w1)/train_rows
    grad_w2_reg = ((lambdaval*w2)+grad_w2)/train_rows

    obj_grad = np.concatenate((grad_w1_reg.flatten(),grad_w2_reg.flatten()),0)

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    # obj_grad = np.array([])

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image

    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    train_rows = data.shape[0]
    biase1 = np.full((train_rows,1), 1)

    data_wit_biases = np.concatenate((biase1,data), axis=1)
    bj = sigmoid(np.dot(data_wit_biases, w1.T))

    biase2 = np.full((bj.shape[0],1),1)
    data2 = np.concatenate((biase2,bj),axis=1)
    cj = sigmoid(np.dot(data2,w2.T))
    labels = np.argmax(cj, axis=1)
    return labels


"""**************Neural Network Script Starts here********************************"""
selected_indicies = []
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
exec_time = []
train_acc = []
test_acc = []
validation_acc = []
lambdas = []
hidden = []

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in output unit
n_class = 10

lambada_vals = np.arange(0, 70, 10)
n_hidden_vals = np.arange(4, 24, 4)


for lambda_val in lambada_vals:
    for n_hidden in n_hidden_vals:

        start = time.time()

        #  Train Neural Network
        # initialize the weights into some random matrices
        initial_w1 = initializeWeights(n_input, n_hidden)
        initial_w2 = initializeWeights(n_hidden, n_class)

        # unroll 2 weight matrices into single column vector
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

        args = (n_input, n_hidden, n_class, train_data, train_label, lambda_val)

        # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

        opts = {'maxiter': 50}  # Preferred value.

        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

        # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
        # and nnObjGradient. Check documentation for this function before you proceed.
        # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


        # Reshape nnParams from 1D vector into w1 and w2 matrices
        w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
        w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

        # Test the computed parameters

        predicted_label = nnPredict(w1, w2, train_data)

        # find the accuracy on Training Dataset
        training_accuracy = str(100 * np.mean((predicted_label == train_label).astype(float)))

        print('Training set Accuracy:' + training_accuracy + '%', end=" ")

        predicted_label = nnPredict(w1, w2, validation_data)

        validation_accuracy = str(100 * np.mean((predicted_label == validation_label).astype(float)))
        # find the accuracy on Validation Dataset
        print('|| Validation set Accuracy:' + validation_accuracy + '%', end=" ")

        predicted_label = nnPredict(w1, w2, test_data)

        # find the accuracy on Validation Dataset
        test_accuracy = str(100 * np.mean((predicted_label == test_label).astype(float)))
        print('|| Test set Accuracy:' + test_accuracy + '%', end=" ")

        end = time.time()

        exec_time.append(end - start)
        lambdas.append(lambda_val)
        hidden.append(n_hidden)
        test_acc.append(float(test_accuracy))
        train_acc.append(float(training_accuracy))
        validation_acc.append(float(validation_accuracy))
        print(' || time=',(end-start), end=" ")
        print(' || n_hidden=', n_hidden, end=" ")
        print(' || λ=', lambda_val)

results = pnd.DataFrame(np.column_stack([lambdas, hidden, train_acc, validation_acc, test_acc, exec_time]),
                       columns=['λ', 'm', 'Train_Accuracy', 'Validation_Accuracy', 'Test_Accuracy', 'Training_Time'])
results = results.sort_values(by=['Test_Accuracy'], ascending=False)


results.head(10)

optimal_lambda = results.iloc[0,0]
optimal_m = results.iloc[0,1]

print("Optimal Lambda :", optimal_lambda)
print("Optimal hidden units :", optimal_m)

parameters = [selected_indicies, int(optimal_m), w1, w2, int(optimal_lambda)]
pickle.dump(parameters, open('params.pickle', 'wb'))