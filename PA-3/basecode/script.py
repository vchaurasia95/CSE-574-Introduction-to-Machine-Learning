import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def preprocess():
    """ 
     Input:
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
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    # Step 1: add Bias Term
    X = np.concatenate((np.full((n_data, 1), 1), train_data), axis=1)

    W = initialWeights.reshape(n_features + 1, 1)

    theta = sigmoid(np.dot(X, W))

    # Step 2: calculate error:

    part_a = np.multiply(labeli, np.log(theta))

    part_b = np.multiply(np.subtract(1, labeli), np.log(np.subtract(1, theta)))

    error = np.multiply(-1, np.divide(np.sum(part_a + part_b), n_data))

    # Step 3: calculate error_grad

    error_grad = np.divide(np.sum(np.multiply(np.subtract(theta, labeli), X), axis=0), n_data)

    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    # Add Bias Term

    rows = data.shape[0]

    bias_arr = np.full((rows, 1), 1)
    X = np.concatenate((bias_arr, data), axis=1)

    probablity = sigmoid(np.dot(X, W))

    label = np.argmax(probablity, axis=1)
    label = label.reshape((rows, 1))

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    # Step 1: Add bias
    X = np.concatenate((np.full((n_data, 1), 1), train_data), axis=1)

    W = params.reshape(n_feature + 1, n_class)

    part_a = np.exp(np.dot(X, W))

    part_b_1 = np.sum(part_a, axis=1)
    part_b_2 = part_b_1.reshape(part_b_1.shape[0], 1)

    theta = np.divide(part_a, part_b_2)

    sum_1 = np.sum(np.multiply(labeli, np.log(theta)))
    error = -1 * np.sum(sum_1)

    error_grad = np.dot(X.T, np.subtract(theta, labeli))

    error_grad = error_grad.ravel()

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))
    rows = data.shape[0]

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    X = np.concatenate((np.full((rows, 1), 1), data), axis=1)

    tmp = np.sum(np.exp(np.dot(X, W)), axis=1)
    tmp = tmp.reshape(tmp.shape[0], 1)
    theta = np.divide(np.exp(np.dot(X, W)), tmp)

    label = np.argmax(theta, axis=1)

    return label.reshape(rows, 1)


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('\n--------------SVM-------------------\n')
##################
# YOUR CODE HERE #
##################
index = np.random.randint(50000, size=10000)
tr_svm_data = train_data[index, :]
tr_svm_label = train_label[index, :]

print('-----linear kernel------')
svm_linear = SVC(kernel='linear')
svm_linear.fit(tr_svm_data, tr_svm_label.ravel())
print('\nTraining set Accuracy:' + str(100 * svm_linear.score(tr_svm_data, tr_svm_label)))
print('\nValidation set Accuracy:' + str(100 * svm_linear.score(validation_data, validation_label)))
print('\nTesting set Accuracy:' + str(100 * svm_linear.score(test_data, test_label)))

print('\n-----SVM with RBF for gamma = 1------')
svm_rbf = SVC(kernel='rbf', gamma=1.0)
svm_rbf.fit(tr_svm_data, tr_svm_label.ravel())
print('\nTraining set Accuracy:' + str(100 * svm_rbf.score(tr_svm_data, tr_svm_label)))
print('\nValidation set Accuracy:' + str(100 * svm_rbf.score(validation_data, validation_label)))
print('\nTesting set Accuracy:' + str(100 * svm_rbf.score(test_data, test_label)))

print('\n------SVM with RBF for gamma = 0-------')
svm_default = SVC(kernel='rbf', gamma='auto')
svm_default.fit(tr_svm_data, tr_svm_label.ravel())
print('\nTraining set Accuracy:' + str(100 * svm_default.score(tr_svm_data, tr_svm_label)))
print('\nValidation set Accuracy:' + str(100 * svm_default.score(validation_data, validation_label)))
print('\nTesting set Accuracy:' + str(100 * svm_default.score(test_data, test_label)))

print('\n------SVM with RBF for different values of C------')
c_values = [1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
validation_accuracy = np.zeros(11)
training_accuracy = np.zeros(11)
testing_accuracy = np.zeros(11)

for index in range(11):
    print('\nC: ' + str(c_values[index]))
    svm_c = SVC(C=c_values[index], kernel='rbf')
    svm_c.fit(tr_svm_data, tr_svm_label.ravel())
    training_accuracy[index] = 100 * svm_c.score(tr_svm_data, tr_svm_label)
    print('\nTraining set Accuracy:' + str(training_accuracy[index]))
    validation_accuracy[index] = 100 * svm_c.score(validation_data, validation_label)
    print('\nValidation set Accuracy:' + str(validation_accuracy[index]))
    testing_accuracy[index] = 100 * svm_c.score(test_data, test_label)
    print('\nTesting set Accuracy:' + str(testing_accuracy[index]))

svm_full = SVC(kernel='rbf', gamma='auto', C=70)
svm_full.fit(train_data, train_label.ravel())

print('----------\nRBF with FULL training set with best C :\n------------')
print('\nTraining set Accuracy:' + str(100 * svm_full.score(train_data, train_label)) + '%')
print('\nValidation set Accuracy:' + str(100 * svm_full.score(validation_data, validation_label)) + '%')
print('\nTesting set Accuracy:' + str(100 * svm_full.score(test_data, test_label)) + '%')

"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
